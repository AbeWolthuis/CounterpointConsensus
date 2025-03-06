#!/usr/bin/env python3

import argparse
import os
import sys
import re
from pathlib import Path

DESCRIPTION = """
Merge multiple Python files into one normal, standalone Python script.

Overall Merge Order:
  1) Shebang & short comment
  2) Deduplicated global imports (non-local)
  3) Top-level global constants (in ALL_CAPS), including multi-line definitions
  4) Top-level class definitions
  5) Non-class code
  6) def main(...) + if __name__ == '__main__': main()

Local imports referencing merged modules are removed. 
Any if __name__ == '__main__': blocks are removed to avoid duplicates.
Any lines containing 'importlib.reload' or old in-memory loader references are removed.

Multi-line constants are now handled in a block style, so the last brace won't be lost if it's at column 0.
"""

# -----------------------------------------------------------------------------
# Locate and identify files
# -----------------------------------------------------------------------------

def locate_main_file(main_arg: str = None) -> Path:
    """
    If main_arg is None, default to 'main.py'.
    Otherwise, use the given string:
      - if absolute, ensure file exists
      - if relative, check '.' then '..'
    """
    if main_arg is None:
        main_arg = "main.py"

    candidate = Path(main_arg)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
        else:
            print(f"[ERROR] Main file does not exist: {candidate}")
            sys.exit(1)
    else:
        here = Path('.').resolve()
        candidate_current = here / candidate
        if candidate_current.exists():
            return candidate_current.resolve()

        parent_candidate = here.parent / candidate
        if parent_candidate.exists():
            return parent_candidate.resolve()

        print(f"[ERROR] Could not locate '{main_arg}' in '.' or '..'")
        sys.exit(1)

def guess_module_name(filepath: Path) -> str:
    """Return the stem of a .py file, e.g. 'note.py' -> 'note'."""
    return filepath.stem

# -----------------------------------------------------------------------------
# Remove local imports, if __name__ blocks
# -----------------------------------------------------------------------------

def remove_local_imports(lines, local_modules):
    """
    Remove import lines referencing local modules (in 'local_modules').
    Also remove lines containing 'importlib.reload', 'InMemoryLoader', etc.
    """
    new_lines = []
    for line in lines:
        if ("importlib.reload" in line or
            "InMemoryLoader"  in line or
            "InMemoryFinder"  in line or
            "sys.meta_path"   in line):
            continue

        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            tokens = stripped.split()
            if tokens[0] == "import":
                module_part = tokens[1]  # e.g. "X" or "X.Y"
                parts = module_part.split(".")
                if any(part in local_modules for part in parts):
                    continue  # skip local import
            elif tokens[0] == "from":
                module_part = tokens[1]  # e.g. "X" or "X.Y.Z"
                parts = module_part.split(".")
                if any(part in local_modules for part in parts):
                    continue  # skip local import

        new_lines.append(line)

    return new_lines

def remove_if_name_main_block(lines):
    """
    Remove any top-level block that starts with `if __name__ == '__main__':`.
    Skips that line + all indented lines until we reach a new top-level statement.
    """
    new_lines = []
    skipping = False
    for line in lines:
        if not skipping:
            # Check if line is top-level and has `if __name__` in it
            if line.lstrip() == line and "if __name__" in line and "__main__" in line:
                skipping = True
            else:
                new_lines.append(line)
        else:
            # We are skipping lines until we hit a new top-level statement
            if line.lstrip() == line:
                # This is a new top-level line
                # check if it is another if __name__ block or something else
                if ("if __name__" in line and "__main__" in line):
                    # remain skipping
                    skipping = True
                else:
                    skipping = False
                    new_lines.append(line)
            # else keep skipping
    return new_lines

# -----------------------------------------------------------------------------
# Extract global imports, constants, classes, etc.
# -----------------------------------------------------------------------------

def extract_global_imports(lines):
    """
    Extract any remaining import lines (standard/3rd-party) to place them at
    the top of the final file. Return (import_lines, new_lines).
    """
    import_lines = []
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(line)
        else:
            new_lines.append(line)
    return import_lines, new_lines

CONSTANT_START_REGEX = re.compile(
    r'^([A-Z_][A-Z0-9_]*)'       # top-level name in ALL CAPS, possibly underscores/numbers
    r'(?:\s*:\s*[^=]+)?'         # optional type annotation (e.g. : int)
    r'\s*='                      # equals sign
)

def extract_top_level_constants(lines):
    """
    Extract all top-level "constant" blocks in ALL_CAPS. Similar to how we do for classes:
      - If a line is top-level and matches CONSTANT_START_REGEX, it's the start of a constant block.
      - We gather lines until we see a new top-level constant, or a top-level def/class, or EOF.

    Returns (constant_blocks, remainder).

    Each block is a list of lines for that constant definition (which can be multi-line).
    """
    constant_blocks = []
    remainder = []

    in_constant = False
    current_const = []

    def is_top_level(line):
        return (line.lstrip() == line)

    def start_new_constant(line: str) -> bool:
        # top-level + match pattern
        return is_top_level(line) and bool(CONSTANT_START_REGEX.match(line.strip()))

    def start_new_def(line: str) -> bool:
        return is_top_level(line) and line.strip().startswith("def ")

    def start_new_class(line: str) -> bool:
        return is_top_level(line) and line.strip().startswith("class ")

    idx = 0
    n = len(lines)
    while idx < n:
        line = lines[idx]
        if not in_constant:
            # check if this line starts a new constant
            if start_new_constant(line):
                in_constant = True
                current_const = [line]
            else:
                remainder.append(line)
        else:
            # we are inside a constant block
            # if we see a new top-level constant, or a top-level def, or a top-level class,
            # that means the current constant block ends
            if (start_new_constant(line) or start_new_def(line) or start_new_class(line)):
                # finalize the current const block
                constant_blocks.append(current_const)
                current_const = []
                in_constant = False
                # we re-check this line in the next iteration, so we decrement idx by 1
                # so that we re-process this line as the start of next block or remainder
                idx -= 1
            else:
                current_const.append(line)
        idx += 1

    # if we ended while still in_constant
    if in_constant and current_const:
        constant_blocks.append(current_const)

    return constant_blocks, remainder

def extract_main_function(lines):
    """
    Look for a top-level 'def main(' block. Extract that function body.
    Return (main_def_lines, other_lines).
    """
    main_def_lines = []
    other_lines = []
    in_main_def = False

    for line in lines:
        if not in_main_def and line.strip().startswith("def main(") and line.lstrip() == line:
            in_main_def = True
            main_def_lines.append(line)
            continue

        if in_main_def:
            if (line.strip().startswith("def ") and line.lstrip() == line) or \
               (line.strip().startswith("class ") and line.lstrip() == line):
                in_main_def = False
                other_lines.append(line)
            else:
                main_def_lines.append(line)
        else:
            other_lines.append(line)

    return main_def_lines, other_lines

def extract_top_level_classes(lines):
    """
    Extract all top-level class definitions. For each class, capture until next
    top-level 'class' or 'def' or EOF. Return (class_blocks, remainder).
    """
    class_blocks = []
    remainder = []
    in_class = False
    current_class = []

    def start_new_class(line):
        return line.strip().startswith("class ") and line.lstrip() == line

    def start_new_def(line):
        return line.strip().startswith("def ") and line.lstrip() == line

    idx = 0
    n = len(lines)
    while idx < n:
        line = lines[idx]
        if not in_class:
            if start_new_class(line):
                in_class = True
                current_class = [line]
            else:
                remainder.append(line)
        else:
            # we are inside a class block
            if start_new_class(line) or start_new_def(line):
                # close current class
                class_blocks.append(current_class)
                current_class = []
                in_class = False
                # re-check this line next iteration
                idx -= 1
            else:
                current_class.append(line)
        idx += 1

    if in_class and current_class:
        class_blocks.append(current_class)

    return class_blocks, remainder

# -----------------------------------------------------------------------------
# Parse each file
# -----------------------------------------------------------------------------

def parse_file(file_path: Path, local_modules):
    """
    Parse a single .py file in stages:
      1) Remove local imports + if __name__ blocks
      2) Extract leftover imports (global_imports)
      3) Extract top-level constants in ALL_CAPS
      4) Extract top-level classes
      5) Remainder is non-class code

    Return:
      (
        global_imports: list[str],
        constant_blocks: list[list[str]],
        class_blocks: list[list[str]],
        remainder: list[str]
      )
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1) Remove local imports & if __name__ blocks
    lines = remove_local_imports(lines, local_modules)
    lines = remove_if_name_main_block(lines)

    # 2) Extract global imports
    global_imports, lines2 = extract_global_imports(lines)

    # 3) Extract top-level constants (multi-line blocks)
    constant_blocks, lines3 = extract_top_level_constants(lines2)

    # 4) Extract classes
    class_blocks, lines4 = extract_top_level_classes(lines3)

    # lines4 is the non-class code
    remainder = lines4

    return (global_imports, constant_blocks, class_blocks, remainder)

# -----------------------------------------------------------------------------
# Merge everything
# -----------------------------------------------------------------------------

def merge_files(main_file: Path, other_files: list[Path], output_file: Path):
    """
    1) Identify local modules
    2) Parse each non-main file => gather (imports, const blocks, class blocks, remainder)
    3) Parse main file => plus extract main()
    4) Merge in final order:
       (A) Shebang + short comment
       (B) deduplicated global imports
       (C) all global constants (non-main first, then main)
       (D) classes from non-main
       (E) classes from main
       (F) non-class code from non-main
       (G) non-class code from main
       (H) main(...) + if __name__ == '__main__': main()
    """
    all_modules = {guess_module_name(main_file)} | {guess_module_name(f) for f in other_files}

    # Parse non-main
    non_main_imports = []
    non_main_constants = []
    non_main_classes = []
    non_main_code = []
    for fpath in other_files:
        (f_imports, f_const_blocks, f_class_blocks, f_remainder) = parse_file(fpath, all_modules)
        non_main_imports.extend(f_imports)
        non_main_constants.append((fpath.name, f_const_blocks))
        non_main_classes.append((fpath.name, f_class_blocks))
        non_main_code.append((fpath.name, f_remainder))

    # Parse main
    (main_imports, main_const_blocks, main_class_blocks, main_remainder) = parse_file(main_file, all_modules)
    main_def, main_remainder2 = extract_main_function(main_remainder)

    # Deduplicate imports
    combined_imports = []
    seen_imports = set()
    for imp_line in (non_main_imports + main_imports):
        stripped = imp_line.strip()
        if stripped not in seen_imports:
            seen_imports.add(stripped)
            combined_imports.append(imp_line)

    # Build final lines
    final_lines = []
    # (A) Shebang & comment
    final_lines.append("#!/usr/bin/env python3\n")
    final_lines.append("# Auto-generated merged script\n\n")

    # (B) imports
    for imp_line in combined_imports:
        final_lines.append(imp_line)

    # (C) global constants: non-main first, then main
    for filename, const_blocks in non_main_constants:
        for block in const_blocks:
            final_lines.append(f"\n## ========== START of {filename} (global constants) ========== ##\n")
            final_lines.extend(block)
            final_lines.append(f"## =========== END of {filename} (global constants) =========== ##\n")

    if main_const_blocks:
        fn = main_file.name
        for block in main_const_blocks:
            final_lines.append(f"\n## ========== START of {fn} (global constants) ========== ##\n")
            final_lines.extend(block)
            final_lines.append(f"## =========== END of {fn} (global constants) =========== ##\n")

    # (D) classes from non-main
    for filename, cls_blocks in non_main_classes:
        if cls_blocks:
            final_lines.append(f"\n## ========== START of {filename} (class definitions) ========== ##\n")
            for block in cls_blocks:
                final_lines.extend(block)
            final_lines.append(f"\n## =========== END of {filename} (class definitions) =========== ##\n")

    # (E) classes from main
    if main_class_blocks:
        fn = main_file.name
        final_lines.append(f"\n## ========== START of {fn} (class definitions) ========== ##\n")
        for block in main_class_blocks:
            final_lines.extend(block)
        final_lines.append(f"\n## =========== END of {fn} (class definitions) =========== ##\n")

    # (F) non-class code from non-main
    for filename, code_lines in non_main_code:
        final_lines.append(f"\n## ========== START of {filename} (non-class code) ========== ##\n")
        final_lines.extend(code_lines)
        final_lines.append(f"\n## =========== END of {filename} (non-class code) =========== ##\n")

    # (G) non-class code from main
    fn = main_file.name
    final_lines.append(f"\n## ========== START of {fn} (non-class code) ========== ##\n")
    final_lines.extend(main_remainder2)
    final_lines.append(f"\n## =========== END of {fn} (non-class code) =========== ##\n")

    # (H) main() if found
    if main_def:
        final_lines.append(f"\n## ========== START of {fn} (main function) ========== ##\n")
        final_lines.extend(main_def)
        final_lines.append(f"\n## =========== END of {fn} (main function) =========== ##\n")
        final_lines.append("\nif __name__ == '__main__':\n")
        final_lines.append("    main()\n")
    else:
        final_lines.append("\n# [INFO] No 'def main(...)' found; nothing to run automatically.\n")

    # Write out
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.writelines(final_lines)

    print(f"[INFO] Wrote merged file to {output_file}")

# -----------------------------------------------------------------------------
# CLI Entry
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--main",
        help="Optional path to the main Python file. Defaults to 'main.py' if omitted."
    )
    parser.add_argument(
        "--output", "-o",
        default="merged.py",
        help="Name of the merged script (placed in 'merged' folder). Default = merged.py"
    )
    args = parser.parse_args()

    main_file = locate_main_file(args.main)
    output_dir = Path("merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output

    here = Path('.').resolve()
    all_python_files = list(here.glob("*.py"))

    # Filter out main file, output file, or this script
    other_files = []
    for fpath in all_python_files:
        if fpath.resolve() == main_file.resolve():
            continue
        if fpath.resolve() == output_file.resolve():
            continue
        if fpath.resolve() == Path(__file__).resolve():
            continue
        other_files.append(fpath)

    merge_files(main_file, other_files, output_file)

    print("[INFO] Done. You can run your merged script with:")
    print(f"  python {output_file}")
    print("or make it executable and do:")
    print(f"  ./{output_file}")


if __name__ == "__main__":
    main()

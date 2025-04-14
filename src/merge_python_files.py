#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

DESCRIPTION = """
Merge multiple Python files into one normal, standalone Python script.
 - If you don't specify --main, the script looks for 'main.py' in the current directory or parent directory.
 - If you do specify --main (relative or absolute), the script tries to find it accordingly.
 - All other .py files in the current directory are merged, removing references to each other's imports.
 - A def main(...) block is moved to the end, then called in if __name__ == '__main__': main().
 - The final script is written to 'merged/merged.py' (or a custom --output name).
"""

def locate_main_file(main_arg: str = None) -> Path:
    """
    If main_arg is None, we default to 'main.py'.
    Otherwise, use the user's value.
      - If it's an absolute path, check if it exists.
      - If it's relative, check '.' then '..'.
    Return the resolved Path if found; otherwise sys.exit(1).
    """
    if main_arg is None:
        main_arg = "main.py"  # default if --main not provided

    candidate = Path(main_arg)
    if candidate.is_absolute():
        # If user gave an absolute path
        if candidate.exists():
            return candidate.resolve()
        else:
            print(f"[ERROR] Main file does not exist: {candidate}")
            sys.exit(1)
    else:
        # Check current dir
        here = Path('.').resolve()
        candidate_current = here / candidate
        if candidate_current.exists():
            return candidate_current.resolve()
        # Check parent dir
        parent_candidate = here.parent / candidate
        if parent_candidate.exists():
            return parent_candidate.resolve()

        print(f"[ERROR] Could not locate '{main_arg}' in '.' or '..'")
        sys.exit(1)

def guess_module_name(filepath: Path) -> str:
    """Return the stem of the .py file, e.g., utils.py -> 'utils'."""
    return filepath.stem

def remove_local_imports(lines, local_modules):
    """
    Remove import lines that reference any local module in 'local_modules'.
    Keep other import lines.
    This is a naive text-based approach:
      - if line starts with "import X" or "from X import ..."
      - where X is in local_modules, skip that line.
    Also remove references to old import-hook classes (InMemoryLoader, etc.) if any remain.
    """
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            tokens = stripped.split()
            if tokens[0] == "import":
                possible_module = tokens[1].split(".")[0]
                if possible_module in local_modules:
                    continue  # skip
            elif tokens[0] == "from":
                possible_module = tokens[1].split(".")[0]
                if possible_module in local_modules:
                    continue  # skip

        if "InMemoryLoader" in line or "InMemoryFinder" in line or "sys.meta_path" in line:
            continue

        new_lines.append(line)
    return new_lines

def extract_main_function(lines):
    """
    Look for a top-level 'def main(' block in the lines.
    Extract that entire function definition (until the next top-level def/class).
    Return (main_def_lines, other_lines).
    If there's no main(), main_def_lines will be empty.
    """
    main_def_lines = []
    other_lines = []
    in_main_def = False

    for line in lines:
        # Start capturing if we see 'def main(' at the top level
        if (not in_main_def) and line.strip().startswith("def main(") and line.lstrip() == line:
            in_main_def = True
            main_def_lines.append(line)
            continue

        if in_main_def:
            # If we see another top-level def or class, that means main() ended
            if (line.strip().startswith("def ") and line.lstrip() == line) or \
               (line.strip().startswith("class ") and line.lstrip() == line):
                in_main_def = False
                other_lines.append(line)  # put this new def/class line back in other_lines
            else:
                main_def_lines.append(line)
        else:
            other_lines.append(line)

    return main_def_lines, other_lines

def merge_files(main_file: Path, other_files: list[Path], output_file: Path):
    """
    1) Identify local module names from main_file + other_files.
    2) Remove local imports from each file, then concatenate them, each wrapped in block comments.
    3) Extract def main(...) from main_file to place last, then call it.
    """
    all_modules = {guess_module_name(main_file)} | {guess_module_name(f) for f in other_files}

    merged_lines = []

    # First merge the non-main files:
    for fpath in other_files:
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = remove_local_imports(lines, all_modules)

        # Wrap in block comments
        merged_lines.append(f"\n## ========== START of {fpath.name} ========== ##\n")
        merged_lines.extend(lines)
        merged_lines.append(f"\n## =========== END of {fpath.name} =========== ##\n")

    # Now handle the main file:
    with open(main_file, "r", encoding="utf-8") as f:
        main_lines = f.readlines()

    main_lines = remove_local_imports(main_lines, all_modules)
    main_def, other_main = extract_main_function(main_lines)

    # Insert a block comment for the non-main part
    merged_lines.append(f"\n## ========== START of {main_file.name} (non-main code) ========== ##\n")
    merged_lines.extend(other_main)
    merged_lines.append(f"\n## =========== END of {main_file.name} (non-main code) =========== ##\n")

    # If there's a def main(...), put it next
    if main_def:
        merged_lines.append(f"\n## ========== START of {main_file.name} (main function) ========== ##\n")
        merged_lines.extend(main_def)
        merged_lines.append(f"\n## =========== END of {main_file.name} (main function) =========== ##\n")

        # Add the call
        merged_lines.append("\nif __name__ == '__main__':\n")
        merged_lines.append("    main()\n")
    else:
        merged_lines.append("\n# [INFO] No 'def main(...)' found; nothing to run automatically.\n")

    # Write final merged lines
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.writelines(merged_lines)

    print(f"[INFO] Wrote merged file to {output_file}")

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--main",
        help="Optional path to the main Python file. If omitted, looks for 'main.py' in '.' or '..'."
    )
    parser.add_argument(
        "--output", "-o",
        default="merged.py",
        help="Name of the merged script (placed in 'merged' folder). Default = merged.py"
    )
    args = parser.parse_args()

    # Locate main file, either from --main or default 'main.py'
    main_file = locate_main_file(args.main)

    # Output folder 'merged'
    output_dir = Path("merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output

    # Gather all .py files in current dir, skipping main file, output file, or this script
    here = Path('.').resolve()
    all_python_files = list(here.glob("*.py"))

    other_files = []
    for fpath in all_python_files:
        if fpath.resolve() == main_file.resolve():
            continue
        if fpath.resolve() == output_file.resolve():
            continue
        if fpath.resolve() == Path(__file__).resolve():
            continue
        other_files.append(fpath)

    # Merge them
    merge_files(main_file, other_files, output_file)

    print("[INFO] Done. You can run your merged script with:")
    print(f"  python {output_file}")
    print("or make it executable and do:")
    print(f"  ./{output_file}")

if __name__ == "__main__":
    main()

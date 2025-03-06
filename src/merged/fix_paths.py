import argparse
import re
import sys
from pathlib import Path

DESCRIPTION = r"""
A helper script that looks for single-line calls to os.path.<function>(...) in .py files.
If the first argument is '..' or 'current_dir' (quoted or unquoted), it inserts another
'..' immediately after that argument, effectively going "one more level up."

Examples:

  # Original
  filepath = os.path.join("..", "data", "test")
  => 
  filepath = os.path.join("..", "..", "data", "test")

  # Original
  filepath2 = os.path.join(current_dir, "..", "data")
  =>
  filepath2 = os.path.join(current_dir, '..', "..", "data")

Usage:
  python fix_paths.py
    # Reads 'merged.py' from current directory
    # Writes 'merged_fixed_paths.py'

Optional:
  --input  FILE  (default merged.py)
  --output FILE  (default merged_fixed_paths.py)

Caveats:
  - Only single-line calls like os.path.join("..", "data").
  - Naively splits arguments on commas, so won't handle advanced code.
  - The first argument is checked; if it is '..' or 'current_dir', 
    we insert '..' right after it in the final call.
"""

# Regex capturing:
#   os.path.<function>(  ... )
#   group(1) = "os.path.join("
#   group(2) = the arguments inside (no parentheses)
#   group(3) = closing ")"
PATTERN = re.compile(
    r'(os\.path\.\w+\()\s*([^)]*)(\))'
)

def normalize_arg(arg_str: str) -> str:
    """
    Attempt to strip matching quotes from the start/end if present.
    This helps compare against '..' or 'current_dir' regardless of quotes.
    Example: '".."' -> '..'
             'current_dir' -> 'current_dir'
    """
    arg_str = arg_str.strip()
    if len(arg_str) >= 2 and arg_str[0] in ("'", '"') and arg_str[-1] == arg_str[0]:
        # e.g. "data" or 'data'
        return arg_str[1:-1]  # remove quotes
    return arg_str

def fix_first_argument(args_str: str) -> str:
    """
    Naively split the argument string by commas.
    If the *first argument* is '..' or 'current_dir' (with or without quotes),
    insert another '..' right after it.
    """
    # Split by commas (naive approach)
    raw_args = [a.strip() for a in args_str.split(",")]

    # If no arguments, do nothing
    if not raw_args:
        return args_str

    first_arg = raw_args[0]
    normalized = normalize_arg(first_arg)  # e.g. ".." or current_dir, etc.

    # If it is '..' or 'current_dir', we insert "'..'" after it
    if normalized == '..' or normalized == 'current_dir':
        # We'll insert '..' with single quotes
        raw_args.insert(1, "'..'")  
        # e.g. ["'..'", "'data'"] => ["'..'", "'..'", "'data'"]

    # Rebuild
    return ", ".join(raw_args)

def fix_line(line: str) -> str:
    """
    For each single-line call to os.path.<function>(args),
    if the first argument is '..' or 'current_dir', 
    add another '..' after it.
    """
    def replacer(match):
        start = match.group(1)  # e.g. "os.path.join("
        args = match.group(2)   # e.g. '"..", "data"'
        end = match.group(3)    # e.g. ")"

        fixed_args = fix_first_argument(args)
        return f"{start}{fixed_args}{end}"

    return re.sub(PATTERN, replacer, line)

def fix_file_lines(lines):
    """Apply fix_line to each line in the file."""
    new_lines = []
    for line in lines:
        new_lines.append(fix_line(line))
    return new_lines

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--input", "-i",
        default="merged.py",
        help="Path to input Python file (default: merged.py)"
    )
    parser.add_argument(
        "--output", "-o",
        default="merged_fixed_paths.py",
        help="Path to write the fixed file (default: merged_fixed_paths.py)"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        print(f"[ERROR] The input file does not exist: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Fix lines
    fixed_lines = fix_file_lines(lines)

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.writelines(fixed_lines)

    print(f"[INFO] Done. Output file:", output_path)
    print("Example transform: os.path.join('..', 'data') -> os.path.join('..', '..', 'data')")

if __name__ == "__main__":
    main()

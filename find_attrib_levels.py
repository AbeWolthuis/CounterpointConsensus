import os
import re
from collections import defaultdict

def find_attribution_levels(dataset_path: str) -> dict:
    """
    Find all unique attribution level lines for Jos files in the dataset.
    
    Returns:
        dict: {attribution_line: [list_of_files_with_this_line]}
    """
    attribution_pattern = re.compile(r'^!!attribution-level@Jos:\s*.*')
    attribution_lines = defaultdict(list)
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            #print(file)
            if file.startswith('Jos') and file.endswith('.krn'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if attribution_pattern.match(line):
                                attribution_lines[line].append(file)
                                break  # Assume only one attribution line per file
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    
    return dict(attribution_lines)

def print_attribution_summary(dataset_path: str):
    """Print a summary of all attribution levels found."""
    attribution_data = find_attribution_levels(dataset_path)
    
    print("=" * 80)
    print("ATTRIBUTION LEVELS FOR JOS FILES")
    print("=" * 80)
    
    if not attribution_data:
        print("No attribution-level lines found for Jos files.")
        return
    
    # Sort by attribution line for consistent output
    for attribution_line in sorted(attribution_data.keys()):
        files = attribution_data[attribution_line]
        print(f"\n{attribution_line}")
        # print(f"  Files: {len(files)}")
        # print(f"  Examples: {', '.join(files[:5])}")
        # if len(files) > 5:
        #     print(f"  ... and {len(files) - 5} more")

if __name__ == "__main__":
    # Fix the path construction - script is in project root, not src/
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))  # This is the project root
    DATASET_PATH = r'C:\Users\T460\Documents\Uni_spul\Jaar_7\Scriptie\CounterpointConsensus2\CounterpointConsensus\data\full\more_than_10\SELECTED'


    print(f"Looking for files in: {DATASET_PATH}")
    print(f"Path exists: {os.path.exists(DATASET_PATH)}")
    
    # List some directories to verify path
    print(os.listdir(DATASET_PATH)[:10])  # List first 10 items in the dataset path
    
    print_attribution_summary(DATASET_PATH)

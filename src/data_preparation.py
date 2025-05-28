import inspect
import os


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from counterpoint_rules import RuleViolation, CounterpointRules

# Data preparation functions loading the JRP files
def find_jrp_files(root_dir, valid_files=None, invalid_files=None, anonymous_mode='skip'):
    """
    Recursively find all .krn files in root_dir, filter by valid_files/invalid_files.
    Returns a list of filepaths.
    """

    krn_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.krn'):
                # Assumes the file name format is consistent with JRP IDs, e.g., "COM_01-12345-01.krn"
                # Note, the composer-abbreviation is the first 3 characters of the filename.
                original_fname = fname  # Preserve original filename
                jrp_id = fname.split('-')[0]
                
                if valid_files is not None and jrp_id not in valid_files:
                    continue
                if invalid_files is not None and jrp_id in invalid_files:
                    continue
                    
                # Check for Ano files if ano_mode is 'skip' or 'prefix'
                if jrp_id.startswith('Ano'):
                    if anonymous_mode == 'skip':
                        continue
                    elif anonymous_mode == 'prefix':
                        # Get parent folder name (assume immediate parent is the composer code)
                        parent_folder = os.path.basename(os.path.dirname(dirpath))
                        jrp_id = f"{parent_folder}_{jrp_id}"
                        # Don't modify fname - keep original filename for file system access

                krn_files.append(os.path.join(dirpath, original_fname))  # Use original filename
    return krn_files



# Data preparation functions for counterpoint violations




def extract_rule_id_from_column(column_name: str) -> str:
    """
    Extract rule ID from column name in format 'rule_name, rule_id'.
    Returns the rule_id part, or empty string if not found.
    """
    if ', ' in column_name:
        parts = column_name.split(', ')
        if len(parts) >= 2:
            return parts[-1]  # Take the last part as rule_id
    return ""

def sort_df_by_rule_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort DataFrame columns by rule ID. Columns without rule IDs are placed at the end.
    Special handling for 'composer' column to keep it last.
    """
    if df.empty:
        return df
    
    # Separate columns into categories
    rule_columns = []
    other_columns = []
    composer_column = None
    
    for col in df.columns:
        if col.lower() == 'composer':
            composer_column = col
        else:
            rule_id = extract_rule_id_from_column(col)
            if rule_id:
                rule_columns.append((col, rule_id))
            else:
                other_columns.append(col)
    
    # Sort rule columns by rule_id (case-insensitive)
    rule_columns.sort(key=lambda x: x[1].lower())
    
    # Sort other columns alphabetically
    other_columns.sort()
    
    # Build final column order
    sorted_columns = [col for col, _ in rule_columns] + other_columns
    
    # Add composer column at the end if it exists
    if composer_column:
        sorted_columns.append(composer_column)
    
    # Return DataFrame with reordered columns
    return df[sorted_columns]

def feature_counts(violations: dict[str, list[RuleViolation]]) -> dict[str, int]:
    '''
    The structured of the returned dictionary:
        keys: rule names (e.g., "Parallel_fifths", "Uncompensated_leap")
        values: number of violations for each rule
        e.g., {"Parallel_fifths": 2, "Uncompensated_leap": 3} 
    '''
    counts = {rule: len(violation_list) for rule, violation_list in violations.items()}
    return counts

def get_rule_id_from_method(rule_name: str) -> str:
    """
    Extract rule_id from the source code of a counterpoint rule method.
    """
    try:
        # Get the method from CounterpointRules class
        rule_method = getattr(CounterpointRules, rule_name)
        
        # Get the source code lines
        source_lines = inspect.getsource(rule_method).split('\n')
        
        # Look for the rule_id assignment
        for line in source_lines:
            line = line.strip()
            if line.startswith('rule_id ='):
                # Extract the value between quotes
                if "'" in line:
                    rule_id = line.split("'")[1]
                    return rule_id
                elif '"' in line:
                    rule_id = line.split('"')[1]
                    return rule_id
        
        # If rule_id not found, return empty string
        return ""
    except:
        # If any error occurs, return empty string
        return ""


def violations_to_df(violations: dict[str, list[RuleViolation]], metadata) -> pd.DataFrame:
    violation_counts = feature_counts(violations)
    # Add the composer from metadata (defaulting to "Unknown" if not present)
    composer = metadata.get("COM", "Unknown")
    # Add the composer to the violation_counts dictionary
    violation_counts["composer"] = composer

    # Create df (with dict keys as columns), and set composer as last column
    df = pd.DataFrame([violation_counts])
    cols = [col for col in df.columns if col != "composer"] + ["composer"]
    df = df[cols]

    # Now rename ALL columns to include rule IDs by inspecting the method source
    column_name_map = {}
    for rule_name in violations.keys():
        rule_id = get_rule_id_from_method(rule_name)
        if rule_id:  # Only rename if we successfully extracted a rule_id
            column_name_map[rule_name] = f"{rule_name}, {rule_id}"
    
    # Apply the renaming but leave 'composer' unchanged
    df = df.rename(columns=column_name_map)
    
    return df








if __name__ == "__main__":
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))  # This is src/
    PROJECT_ROOT = os.path.dirname(ROOT_PATH)               # Go up one level to CounterpointConsensus/
    DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "full", "more_than_10", "SELECTED")

    # Quick test - find ANY .krn files without filtering
    test_files = []
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith('.krn'):
                test_files.append(os.path.join(root, file))
                print(f"Found: {file}")
    
    print(f"Total .krn files found without filtering: {len(test_files)}")
    
    # Now test with your function
    filepaths = find_jrp_files(DATASET_PATH, None, None, anonymous_mode='skip')
    print(f"Files found with find_jrp_files: {len(filepaths)}")
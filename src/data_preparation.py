import inspect
import os


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np



from counterpoint_rules_base import RuleViolation, CounterpointRulesBase

DEBUG = False

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
                filename_base = fname.split('-')[0]
                
                if valid_files is not None and filename_base not in valid_files:
                    continue
                if invalid_files is not None and filename_base in invalid_files:
                    continue

                if filename_base.startswith('Jos'):
                    certain_attribution = check_josquin_attribution_level(os.path.join(dirpath, original_fname))
                    if not certain_attribution:
                        if DEBUG: print(f"Skipping {original_fname} due to low attribution level.")
                        continue
                # Check for Ano files if ano_mode is 'skip' or 'prefix'
                elif filename_base.startswith('Ano'):
                    if anonymous_mode == 'skip':
                        continue
                    elif anonymous_mode == 'prefix':
                        # Get parent folder name (assume immediate parent is the composer code)
                        parent_folder = os.path.basename(os.path.dirname(dirpath))
                        filename_base = f"{parent_folder}_{filename_base}"
                        # Don't modify fname - keep original filename for file system access

                krn_files.append(os.path.join(dirpath, original_fname))  # Use original filename
    return krn_files

def check_josquin_attribution_level(filepath: str) -> bool:
        """Check if file has low attribution level (4a or 4b) and is by Jos."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('!!attribution-level@Jos:'):
                    if line.startswith('!!attribution-level@Jos: 3') or line.startswith('!!attribution-level@Jos: 4a') or line.startswith('!!attribution-level@Jos: 4b'):
                        return False

        return True  # Include if we can't determine attribution or it's not 4a or 4b


# --- Data preparation functions for counterpoint violations ---




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
    Special handling for 'jrpid' and 'composer' columns to keep them first.
    """
    if df.empty:
        return df
    
    # Separate columns into categories
    rule_columns = []
    other_columns = []
    jrpid_column = None
    composer_column = None
    
    for col in df.columns:
        if col.lower() == 'composer':
            composer_column = col
        elif col.lower() == 'jrpid':
            jrpid_column = col
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
    
    # Build final column order: jrpid first, composer second, then rule columns, then other columns
    sorted_columns = []
    
    # Add jrpid and composer first if they exist
    if jrpid_column:
        sorted_columns.append(jrpid_column)
    if composer_column:
        sorted_columns.append(composer_column)
    
    # Add rule columns sorted by rule_id
    sorted_columns.extend([col for col, _ in rule_columns])
    
    # Add other columns
    sorted_columns.extend(other_columns)
    
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
        # Try to get the method from multiple possible classes
        rule_method = None
        possible_classes = [CounterpointRulesBase]
        
        # Import the other classes
        from counterpoint_rules_most import CounterpointRulesMost
        from counterpoint_rules_motion import CounterpointRulesMotion
        from counterpoint_rules_normalization import CounterpointRulesNormalization
        
        possible_classes.extend([CounterpointRulesMost, CounterpointRulesMotion, CounterpointRulesNormalization])
        
        # Try to find the method in any of the classes
        for cls in possible_classes:
            if hasattr(cls, rule_name):
                rule_method = getattr(cls, rule_name)
                break
        
        if rule_method is None:
            return ""
        
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
    except:
        return ""

def violations_to_df(violations: dict[str, list[RuleViolation]], metadata) -> pd.DataFrame:
    """
    Convert violations dictionary to DataFrame with rule counts, composer, and JRP ID.
    
    Args:
        violations: Dictionary mapping rule names to lists of RuleViolation objects
        metadata: Metadata dictionary containing composer and JRP ID information
        
    Returns:
        DataFrame with columns for each rule (with rule IDs), composer, and jrpid
    """
    violation_counts = feature_counts(violations)
    
    # Add the composer from metadata (defaulting to "Unknown" if not present)
    composer = metadata.get("COM", "Unknown")
    violation_counts["composer"] = composer
    
    # Add the JRP ID from metadata (defaulting to "Unknown" if not present)
    jrpid = metadata.get("jrpid", "Unknown")
    violation_counts["jrpid"] = jrpid

    # Create df (with dict keys as columns)
    df = pd.DataFrame([violation_counts])
    
    # Reorder columns: composer first, jrpid second, rule columns last
    rule_columns = [col for col in df.columns if col not in ["composer", "jrpid"]]
    cols = ["composer", "jrpid"] + rule_columns 
    df = df[cols]

    # Now rename ALL rule columns to include rule IDs by inspecting the method source
    column_name_map = {}
    for rule_name in violations.keys():
        rule_id = get_rule_id_from_method(rule_name)
        if rule_id:  # Only rename if we successfully extracted a rule_id
            column_name_map[rule_name] = f"{rule_name}, {rule_id}"

    # Apply the renaming but leave 'composer' and 'jrpid' unchanged
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
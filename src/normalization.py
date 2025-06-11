
import pandas as pd

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize counterpoint rule features using rule-specific normalization functions.
    
    Args:
        df: DataFrame with rule columns and normalization columns
        
    Returns:
        DataFrame with normalized rule features
    """
    df_normalized = df.copy()
    
    # Map the rule_id and norm_id to their column in the DF
    rule_columns_by_rule_id = {}
    norm_columns_by_norm_id = {}

    for col in df.columns:
        if ', ' in col:
            parts = col.split(', ')
            if len(parts) >= 2:
                rule_id = parts[-1]
                if col.startswith('norm_'):
                    norm_columns_by_norm_id[rule_id] = col
                else:
                    rule_columns_by_rule_id[rule_id] = col
    
    # Loop over all rule columns and apply normalization
    for rule_id, rule_col in rule_columns_by_rule_id.items():
        normalizer_func = NORMALIZATION_FUNCTIONS[rule_id]
        try:
            df_normalized[rule_col] = normalizer_func(df, rule_col, norm_columns_by_norm_id)
        except Exception as e:
            print(f"Warning: Failed to normalize rule {rule_id}: {e}")

    
    # Rename scaled columns to have a prefix 'scaled_'
    rename_mapping = {col: f"scaled_{col}" for col in rule_columns_by_rule_id.values() if col in df_normalized.columns}
    df_normalized.rename(columns=rename_mapping, inplace=True)

    
    return df_normalized

# --------------- NORMALIZATION FUNCTIONS ---------------

def normalize_rule_22(df: pd.DataFrame, rule_col: str, norm_columns_by_norm_id: dict) -> pd.Series:
    """Normalize rule 22, divide by total leaps."""
    norm_col = norm_columns_by_norm_id['N22']
    # Normalization logic
    return df[rule_col] / df[norm_col]

def normalize_rule_23(df: pd.DataFrame, rule_col: str, norm_columns_by_norm_id: dict) -> pd.Series:
    """Normalize rule 23, divide by total successive leaps."""
    norm_col = norm_columns_by_norm_id['N25']
    # Normalization logic
    return df[rule_col] / df[norm_col] 

def normalize_rule_25(df: pd.DataFrame, rule_col: str, norm_columns_by_norm_id: dict) -> pd.Series:
    """Normalize rule 25, divide by total successive leaps."""
    norm_col = norm_columns_by_norm_id['N25']

    # Normalization logic
    return df[rule_col] / df[norm_col]

def normalize_rule_27(df: pd.DataFrame, rule_col: str, norm_columns_by_norm_id: dict) -> pd.Series:
    """Normalize rule 26, divide by total successive leaps."""
    norm_col = norm_columns_by_norm_id['N25']

    # Normalization logic
    return df[rule_col] / df[norm_col]


NORMALIZATION_FUNCTIONS = {
    '22': normalize_rule_22,
    '23': normalize_rule_23,
    '25': normalize_rule_25,
    '27': normalize_rule_27,
}
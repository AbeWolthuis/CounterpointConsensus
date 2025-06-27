import pandas as pd

def normalize_dataframe(df: pd.DataFrame, normalize_everything_by_notecount: bool = False) -> pd.DataFrame:
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

    # Track which columns we actually scaled
    scaled_cols: list[str] = []
    
    # Loop over all rule columns and apply normalization
    for rule_id, rule_col in rule_columns_by_rule_id.items():
        if normalize_everything_by_notecount:
            # Check if note count column exists
            if 'N00' not in norm_columns_by_norm_id:
                print(f"Warning: Note count normalization (N00) not found. Cannot normalize {rule_id}")
                continue

            norm_col = norm_columns_by_norm_id['N00']
            df_normalized[rule_col] = df[rule_col] / df[norm_col]


            # Check if this is one of the voices we normalize by count of possible voice-combinations
            voice_combination_count_rules = ['41', '42', '43', '44', '48']
            if rule_id in voice_combination_count_rules:
                norm_col_voicecount = norm_columns_by_norm_id['N41']
                df_normalized[rule_col] = df_normalized[rule_col] / df[norm_col_voicecount]


            scaled_cols.append(rule_col) 


        else:
            # Otherwise, use the specific normalization function for each this rule
            try:
                normalizer_func = NORMALIZATION_FUNCTIONS[rule_id]
            except KeyError:
                print(f"Warning: No normalization function for rule {rule_id}. Skipping normalization.")
                continue
            try:
                df_normalized[rule_col] = normalizer_func(df, rule_col, norm_columns_by_norm_id)
                scaled_cols.append(rule_col)
            except Exception as e:
                print(f"Warning: Failed to normalize rule {rule_id}: {e}")

    
    # Rename only the columns we scaled
    rename_mapping = {col: f"scaled_{col}" for col in scaled_cols}
    df_normalized.rename(columns=rename_mapping, inplace=True)

    
    return df_normalized

# --------------- NORMALIZATION FUNCTIONS ---------------

def normalize_rule_22(df: pd.DataFrame, rule_col: str, norm_columns_by_norm_id: dict) -> pd.Series:
    """Normalize rule 22, divide by total leaps."""
    norm_col = norm_columns_by_norm_id['N22']
    # Normalization logic
    return df[rule_col] / df[norm_col]

def normalize_rule_23(df: pd.DataFrame, rule_col: str, norm_columns_by_norm_id: dict) -> pd.Series:
    """Normalize rule 23, divide by leaps approached and followed by a note."""
    norm_col = norm_columns_by_norm_id['N23']
    # Normalization logic
    return df[rule_col] / df[norm_col] 

def normalize_rule_25(df: pd.DataFrame, rule_col: str, norm_columns_by_norm_id: dict) -> pd.Series:
    """Normalize rule 25, divide by total successive leaps."""
    norm_col = norm_columns_by_norm_id['N25']

    # Normalization logic
    return df[rule_col] / df[norm_col]

def normalize_rule_26(df: pd.DataFrame, rule_col: str, norm_columns_by_norm_id: dict) -> pd.Series:
    """Normalize rule 26, divide by total successive leaps."""
    norm_col = norm_columns_by_norm_id['N25']

    # Normalization logic
    return df[rule_col] / df[norm_col]

def normalize_rule_27(df: pd.DataFrame, rule_col: str, norm_columns_by_norm_id: dict) -> pd.Series:
    """Normalize rule 27, divide by new occurrences on strong beats"""
    norm_col = norm_columns_by_norm_id['N27']

    # Normalization logic
    return df[rule_col] / df[norm_col]


NORMALIZATION_FUNCTIONS = {
    '22': normalize_rule_22,
    '23': normalize_rule_23,
    '25': normalize_rule_25,
    '26': normalize_rule_26,
    '27': normalize_rule_27,
    
}
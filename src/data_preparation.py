import pandas as pd

from typing import Dict, List

from counterpoint_rules import RuleViolation


def violations_to_df(violations: Dict[str, List[RuleViolation]], metadata) -> pd.DataFrame:
    violation_counts = feature_counts(violations)
    # Create DataFrame with one row, where keys become column names
    df = pd.DataFrame([violation_counts])
    
    return df

def feature_counts(violations: Dict[str, List[RuleViolation]]) -> Dict[str, int]:
    # Count the number of violations for each rule
    counts = {rule: len(violation_list) for rule, violation_list in violations.items()}
    return counts

#
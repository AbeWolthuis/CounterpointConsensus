import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from typing import Dict, List

from counterpoint_rules import RuleViolation



def violations_to_df(violations: Dict[str, List[RuleViolation]], metadata) -> pd.DataFrame:
    violation_counts = feature_counts(violations)
    # Add the composer from metadata (defaulting to "Unknown" if not present)
    composer = metadata.get("COM", "Unknown")
    # Add the composer to the violation_counts dictionary
    violation_counts["composer"] = composer

    # Create df (with dict keys as columns), and set composer as last column
    df = pd.DataFrame([violation_counts])
    cols = [col for col in df.columns if col != "composer"] + ["composer"]
    df = df[cols]

    return df

def feature_counts(violations: Dict[str, List[RuleViolation]]) -> Dict[str, int]:
    '''
        keys: rule names (e.g., "Parallel_fifths", "Uncompensated_leap")
        values: number of violations for each rule
        e.g., {"Parallel_fifths": 2, "Uncompensated_leap": 3} 
    '''
    counts = {rule: len(violation_list) for rule, violation_list in violations.items()}
    return counts


if __name__ == "__main__":
    SAVE_FIG = False

    # Example of final DF
    sample_data = [
        {"Parallel_fifths": 1, "Uncompensated_leap": 3, "Wrong_resolution_of_suspension": 0, "Composer": "des Prez, Josquin"},
        {"Parallel_fifths": 0, "Uncompensated_leap": 1, "Wrong_resolution_of_suspension": 1, "Composer": "des Prez, Josquin"},
        {"Parallel_fifths": 2, "Uncompensated_leap": 0, "Wrong_resolution_of_suspension": 3, "Composer": "la Rue, Pierre de"},
    ]
    
    # Create DataFrame from sample data
    sample_df = pd.DataFrame(sample_data)
    
    # Reorder columns to put composer at the end
    cols = [col for col in sample_df.columns if col != "Composer"] + ["Composer"]
    sample_df = sample_df[cols]
    
    print("Sample DataFrame structure:")
    print(sample_df.to_string(index=False))
    
    # Create a figure and axis with transparent background
    fig = plt.figure(figsize=(10, 4), dpi=150, facecolor='none', edgecolor='none')
    ax = plt.subplot(111)
    
    # Hide axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Hide frame
    plt.box(on=None)
    
    # Set axis background to transparent
    ax.patch.set_visible(False)
    
    # Create table plot
    table = plt.table(
        cellText=sample_df.values,
        colLabels=sample_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color the header row
    for i, key in enumerate(sample_df.columns):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4472C4')
    
    # Add alternating row colors for better readability
    for i in range(len(sample_df)):
        row_color = '#E6F0FF' if i % 2 == 0 else 'white'
        for j in range(len(sample_df.columns)):
            cell = table[(i+1, j)]
            cell.set_facecolor(row_color)
    
    # Save the table as a PNG file with transparency
    if SAVE_FIG:
        plt.savefig('counterpoint_violations.png', 
                    bbox_inches='tight', 
                    pad_inches=0.05, 
                    transparent=True)
        print("Table saved as 'counterpoint_violations.png'")
    else:
        # Optional: display the plot (if running in interactive environment)
        plt.show()
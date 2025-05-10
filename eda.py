import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormaps
import numpy as np # Import numpy for linspace
import seaborn as sns # Import seaborn


def find_jrp_files(root_dir, valid_files=None, invalid_files=None, ano_mode='skip'):

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
                jrp_id = fname.split('-')[0]
                if valid_files is not None and jrp_id not in valid_files:
                    continue
                if invalid_files is not None and jrp_id in invalid_files:
                    continue
                # Check for Ano files if ano_mode is 'skip' or 'prefix'
                if jrp_id.startswith('Ano'):
                    if ano_mode == 'skip':
                        continue
                    elif ano_mode == 'prefix':
                        # TODO: get the composer from another source, e.g. one of the !!! record fields in the JRP data

                        # Get parent folder name (assume immediate parent is the composer code)
                        parent_folder = os.path.basename(os.path.dirname(dirpath))
                        jrp_id = f"{parent_folder}_{jrp_id}"
                        # Update the filename to include the parent folder
                        fname = f"{parent_folder}_{fname}"

                krn_files.append(os.path.join(dirpath, fname))
    return krn_files

def count_slices(filepath):
    """
    Counts the number of lines in a Kern file that do not start with '=', '!', or '*'.
    These lines typically represent musical data slices.
    """
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(('=', '!', '*')):
                    count += 1
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return 0 # Return 0 or None, or raise exception as appropriate
    return count

def get_voice_count(filepath):
    """
    Counts the number of voices (**kern spines) in a Kern file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('**kern'):
                    # Split the line by tabs and count '**kern' occurrences
                    spines = line.split('\t')
                    voice_count = spines.count('**kern')
                    return voice_count
        # If no **kern line is found
        print(f"Warning: No '**kern' line found in {filepath}")
        return 0
    except Exception as e:
        print(f"Error processing file {filepath} for voice count: {e}")
        return 0 # Return 0 or None, or raise exception as appropriate

def jrp_data_eda(root_dir, valid_files=None, invalid_files=None, ano_mode='skip'):
    """
    Load JRP data from **kern files in root_dir, filtered by valid_files/invalid_files.
    Counts slices and voices for each file.
    Returns a pandas DataFrame with 'jrp_id', 'composer', 'slice_count', 'voice_count', and 'filepath'.
    ano_mode: 'skip' to exclude Ano files, 'prefix' to prefix with folder code.
    """
    files = find_jrp_files(root_dir, valid_files, invalid_files, ano_mode)
    if not files:
        print("No valid files found.")
        return pd.DataFrame()  # Return an empty DataFrame if no files are found

    records = []
    for f in files:
        fname = os.path.basename(f)
        jrp_id = fname.split('-')[0]
        composer = jrp_id[:3]
        # Handle Ano files
        if jrp_id.startswith('Ano'):
            if ano_mode == 'skip':
                continue
            elif ano_mode == 'prefix':
                # Get parent folder name (assume immediate parent is the composer code)
                parent_folder = os.path.basename(os.path.dirname(f))
                jrp_id = f"{parent_folder}_{jrp_id}"
                composer = parent_folder
        slice_count = count_slices(f)
        voice_count = get_voice_count(f)
        records.append({
            'jrp_id': jrp_id,
            'composer': composer,
            'slice_count': slice_count,
            'voice_count': voice_count,
            'filepath': f
        })
    df = pd.DataFrame(records)
    return df

from matplotlib.colors import LogNorm, ListedColormap

class ZeroLogNorm(LogNorm):
    def __call__(self, value, clip=None):
        arr = np.array(value)
        mask = arr == 0
        arr = np.where(mask, 1, arr)  # Temporarily set zeros to 1 for LogNorm
        normed = super().__call__(arr, clip)
        normed = np.where(mask, 0.0, normed)  # Set zeros to 0.0 (first color)
        return normed

def plot_slice_distribution(df):
    """
    Plots distributions and counts related to composers in a 2x2 grid:
    - Top-Left: Number of pieces per composer (bar, sorted by count).
    - Top-Right: Mean slice count per composer (bar, alphabetical).
    - Bottom-Left: Voice count distribution per composer (heatmap).
    - Bottom-Right: Slice count distribution per composer (violin, alphabetical).
    Uses consistent colors per composer for top plots, custom palette for voice count plot.
    """
    if df.empty:
        print("DataFrame is empty. Cannot generate plot.")
        return

    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 16)) # Adjusted figure size for 2x2

    # --- Determine Color Order based on Piece Count ---
    # Calculate piece counts and sort composers by count (descending)
    composer_counts = df['composer'].value_counts().sort_values(ascending=False)
    composers_sorted_by_count = composer_counts.index.tolist()

    # Create a consistent color map based on piece count frequency for composer-level plots
    composer_colors = dict(zip(composers_sorted_by_count, sns.color_palette('pastel', n_colors=len(composers_sorted_by_count))))

    # Get composers sorted alphabetically for consistent x-axis ordering in some plots
    sorted_composers_alpha = sorted(df['composer'].unique())


    # --- Bar Graph - Piece Count (Top-Left) ---
    sns.barplot(ax=axs[0, 0], x=composer_counts.index, y=composer_counts.values, palette=composer_colors, order=composers_sorted_by_count)
    axs[0, 0].set_title('Number of Pieces per Composer')
    axs[0, 0].set_xlabel(None) # Remove x-label
    axs[0, 0].set_ylabel(None) # Remove y-label
    axs[0, 0].tick_params(axis='x', rotation=45)

    # --- Bar Graph - Mean Slice Count (Top-Right) ---
    mean_slice_counts = df.groupby('composer')['slice_count'].mean().sort_index() # Group, mean, sort alphabetically
    sns.barplot(ax=axs[0, 1], x=mean_slice_counts.index, y=mean_slice_counts.values, palette=composer_colors, order=sorted_composers_alpha) # Order alphabetically for x-axis
    axs[0, 1].set_title('Mean Slice Count per Composer')
    axs[0, 1].set_xlabel(None) # Remove x-label
    axs[0, 1].set_ylabel(None) # Remove y-label
    axs[0, 1].tick_params(axis='x', rotation=45)

    # --- Heatmap - Voice Count Distribution (Bottom-Left) ---
    # Remove 1-voice bin and remap all pieces with more than 6 voices to a "6+" bin
    df_heat = df.copy()
    df_heat = df_heat[df_heat['voice_count'] > 1]  # Remove 1-voice pieces if present

    df_heat['voice_count_binned'] = df_heat['voice_count'].apply(lambda v: v if 2 <= v <= 6 else '6+')

    # Ensure the y-axis order: 2, 3, 4, 5, 6, '6+' (with 2 at the bottom)
    y_order = [2, 3, 4, 5, 6, '6+']
    df_heat['voice_count_binned'] = pd.Categorical(df_heat['voice_count_binned'], categories=y_order, ordered=True)

    voice_composer_counts = (
        df_heat.groupby(['voice_count_binned', 'composer'])
        .size()
        .unstack(fill_value=0)
        .reindex(index=y_order)
    )

    

    # Prepare annotation matrix: show count except for zeros (show empty string)
    annot_matrix = voice_composer_counts.astype(str)
    annot_matrix[voice_composer_counts == 0] = ""

    vmax = voice_composer_counts.values.max() if voice_composer_counts.values.max() > 1 else 2

    # Use viridis colormap, but set the first color (for 0) to the lowest color in the colormap (not white)
    base_cmap = plt.get_cmap("viridis")
    colors = [base_cmap(0.0)] + [base_cmap(i) for i in np.linspace(0, 1, 255)[1:]]
    cmap = ListedColormap(colors)

    

    hm = sns.heatmap(
        voice_composer_counts,
        ax=axs[1, 0],
        cmap=cmap,
        annot=annot_matrix,
        fmt="s",
        cbar_kws={},
        norm=ZeroLogNorm(vmin=1, vmax=vmax)
    )
    axs[1, 0].set_title('Voice Count Distribution per Composer')
    axs[1, 0].set_xlabel(None)  # Remove x-label
    axs[1, 0].set_ylabel(None)  # Remove y-label
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].invert_yaxis()  # So that 2 is at the bottom

    # Set colorbar ticks to 1, 10, 100, and max (if distinct)
    cbar = hm.collections[0].colorbar
    cbar.set_label('')  # Remove colorbar label

    # Set colorbar ticks to 1, 10, 100, and max (if distinct)
    import matplotlib.ticker as mticker
    ticks = [1, 3, 10, 33, 100]
    if vmax not in ticks:
        ticks.append(vmax)
    ticks = [t for t in ticks if t <= vmax]
    ticks = sorted(set(ticks))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
    cbar.ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    cbar.ax.yaxis.get_offset_text().set_visible(False)

    # --- Violin Plot - Slice Count (Bottom-Right) ---
    # Plot ordered alphabetically, but colored by frequency map
    sns.violinplot(ax=axs[1, 1], x='composer', y='slice_count', data=df, palette=composer_colors, inner='quartile', order=sorted_composers_alpha) # Use original df
    axs[1, 1].set_title('Distribution of Slice Counts per Composer') # Updated title
    axs[1, 1].set_xlabel(None) # Remove x-label
    axs[1, 1].set_ylabel(None) # Remove y-label explicitly
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].set_ylim(bottom=0) # Ensure y-axis starts at 0


    plt.tight_layout(pad=5.0, h_pad=5.0) # Adjust layout with horizontal and vertical padding
    plt.subplots_adjust(hspace=0.35)  # Increase vertical space between rows

    plt.show()

if __name__ == '__main__':

    ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(ROOT_DIR, "data")
    DATASET_PATH = os.path.join(DATA_DIR, "full", "more_than_10", "SELECTED")

    valid_files = None
    invalid_files = ['Bus3064', 'Bus3078', 'Com1002a', 'Com1002b', 'Com1002c', 'Com1002d', 'Com1002e', 'Duf2027a', 'Duf3015', 'Duf3080', 'Gas0204c', 'Gas0503', 'Gas0504', 'Jos0302e', 'Jos0303c', 'Jos0303e', 'Jos0304c', 'Jos0402d', 'Jos0402e', 'Jos0602e', 'Jos0603d', 'Jos0901e', 'Jos0902a', 'Jos0902b', 'Jos0902c', 'Jos0902d', 'Jos0902e', 'Jos0904a', 'Jos0904b', 'Jos0904d', 'Jos0904e', 'Jos1302', 'Jos1501', 'Jos1610', 'Jos1706', 'Jos1802', 'Jos2015', 'Jos2102', 'Jos2602', 'Jos3004', 'Jos9901a', 'Jos9901e', 'Jos9905', 'Jos9906', 'Jos9907a', 'Jos9907b', 'Jos9908', 'Jos9909', 'Jos9910', 'Jos9911', 'Jos9912', 'Jos9914', 'Jos9923', 'Mar1003c', 'Mar3040', 'Oke1003a', 'Oke1003b', 'Oke1003c', 'Oke1003d', 'Oke1003e', 'Oke1005a', 'Oke1005b', 'Oke1005c', 'Oke1005d', 'Oke1005e', 'Oke1010a', 'Oke1010b', 'Oke1010c', 'Oke1010d', 'Oke1010e', 'Oke1011d', 
'Oke3025', 'Ort2005', 'Rue1007a', 'Rue1007b', 'Rue1007c', 'Rue1007d', 'Rue1007e', 'Rue1029a', 'Rue1029b', 'Rue1029c', 'Rue1029d', 'Rue1029e', 'Rue1035a', 'Rue1035b', 'Rue1035c', 'Rue1035d', 'Rue1035e', 'Rue2028', 'Rue2030', 'Rue2032', 'Rue3004', 'Rue3013', 'Tin3002']
    manual_invalid_files = [
        'Jos0603a',
    ]
    invalid_files = invalid_files.extend(manual_invalid_files)

    # Example usage:
    df = jrp_data_eda(DATASET_PATH, valid_files=valid_files, invalid_files=invalid_files, ano_mode='skip')
    
    print(f"Total number of pieces loaded: {len(df)}") # Print total piece count
    print(df.describe())
    # Plot the distribution
    plot_slice_distribution(df)


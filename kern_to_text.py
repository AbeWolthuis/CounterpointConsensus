import os


def rename_files(directory, old_ext, new_ext):
    """
    Recursively renames files in a directory from old_ext to new_ext.

    Args:
        directory (str): The path to the target directory.
        old_ext (str): The old file extension (e.g., '.krn').
        new_ext (str): The new file extension (e.g., '.txt').
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    print(f"Scanning '{directory}' for files ending with '{old_ext}'...")
    count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(old_ext):
                old_filepath = os.path.join(root, filename)
                base_name = filename[:-len(old_ext)]
                new_filename = base_name + new_ext
                new_filepath = os.path.join(root, new_filename)

                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"Renamed: '{old_filepath}' -> '{new_filepath}'")
                    count += 1
                except OSError as e:
                    print(f"Error renaming '{old_filepath}': {e}")

    print(f"Finished renaming. {count} files were renamed from '{old_ext}' to '{new_ext}'.")

# --- Configuration for direct execution ---
# Set the desired direction: 'krn_to_txt' or 'txt_to_krn'
direction = 'krn_to_txt'  # <-- CHANGE THIS VALUE AS NEEDED

# Set the target directory
target_directory = "data/full/text_versions_more_than_10" # <-- CHANGE THIS PATH IF NEEDED
# --- End Configuration ---

# --- Main execution logic ---
if __name__ == "__main__": # Keep this guard for good practice
    print(f"Running script with direction: '{direction}' on directory: '{target_directory}'")

    if direction == 'krn_to_txt':
        rename_files(target_directory, '.krn', '.txt')
    elif direction == 'txt_to_krn':
        rename_files(target_directory, '.txt', '.krn')
    else:
        print(f"Error: Invalid direction '{direction}'. Choose 'krn_to_txt' or 'txt_to_krn'.")
# --- End Main execution logic ---

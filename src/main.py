import os
import cProfile
import pandas as pd
import traceback
import linecache

from pprint import pprint
from IPython.display import display


''' Our own modules '''
from kern_parser import parse_kern, validate_all_rules
from process_salami_slices import post_process_salami_slices

from counterpoint_rules import CounterpointRules, RuleViolation
from data_preparation import violations_to_df, sort_df_by_rule_id
from annotate_kern import annotate_all_kern
from data_preparation import find_jrp_files



def main():
    # --- Configurations ---
    CONTINUE_ON_ERROR = True  # Set to False to break on first error
    BREAK_EARLY = False
    BREAK_COUNT = 120
    SKIP_SUCCESSFUL_FILES = True  # Flag to control skipping successfully processed files
    VERBOSE = True

    # --- Load kern
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))  # This is src/
    PROJECT_ROOT = os.path.dirname(ROOT_PATH)               # Go up one level to CounterpointConsensus/
    DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "full", "more_than_10", "SELECTED")

    invalid_files = ['Bus3064', 'Bus3078', 'Com1002a', 'Com1002b', 'Com1002c', 'Com1002d', 'Com1002e', 'Duf2027a', 'Duf3015', 'Duf3080', 'Gas0204c', 'Gas0503', 'Gas0504', 'Jos0302e', 'Jos0303c', 'Jos0303e', 'Jos0304c', 'Jos0402d', 'Jos0402e', 'Jos0602e', 'Jos0603d', 'Jos0901e', 'Jos0902a', 'Jos0902b', 'Jos0902c', 'Jos0902d', 'Jos0902e', 'Jos0904a', 'Jos0904b', 'Jos0904d', 'Jos0904e', 'Jos1302', 'Jos1501', 'Jos1610', 'Jos1706', 'Jos1802', 'Jos2015', 'Jos2102', 'Jos2602', 'Jos3004', 'Jos9901a', 'Jos9901e', 'Jos9905', 'Jos9906', 'Jos9907a', 'Jos9907b', 'Jos9908', 'Jos9909', 'Jos9910', 'Jos9911', 'Jos9912', 'Jos9914', 'Jos9923', 'Mar1003c', 'Mar3040', 'Oke1003a', 'Oke1003b', 'Oke1003c', 'Oke1003d', 'Oke1003e', 'Oke1005a', 'Oke1005b', 'Oke1005c', 'Oke1005d', 'Oke1005e', 'Oke1010a', 'Oke1010b', 'Oke1010c', 'Oke1010d', 'Oke1010e', 'Oke1011d', 
                     'Oke3025', 'Ort2005', 'Rue1007a', 'Rue1007b', 'Rue1007c', 'Rue1007d', 'Rue1007e', 'Rue1029a', 'Rue1029b', 'Rue1029c', 'Rue1029d', 'Rue1029e', 'Rue1035a', 'Rue1035b', 'Rue1035c', 'Rue1035d', 'Rue1035e', 'Rue2028', 'Rue2030', 'Rue2032', 'Rue3004', 'Rue3013', 'Tin3002']
    manual_invalid_files = ['Jos0603a',]
    valid_files = None
    invalid_files.extend(manual_invalid_files)

    # filepaths = [os.path.join("..", "data", "test", "Jos1408-Miserimini_mei.krn")]
    # filepaths = [os.path.join("..", "data", "test", "Oke1014-Credo_Village.krn")]
    # filepaths = [os.path.join("..", "data", "test", "Rue1024a.krn")]
    # filepaths = [os.path.join("..", "data", "test", "Rue1024a.krn")]
    # filepaths = [os.path.join("..", "data", "test", "Jos1408-Miserimini_mei.krn"), os.path.join("..", "data", "test", "Rue1024a.krn"),os.path.join("..", "data", "test", "Oke1014-Credo_Village.krn")]

    # filepaths.extend([os.path.join("..", "data", "test", "Rue1024a.krn")]); filepaths = filepaths[::-1]  # Reverse the list to process in reverse order
    # filepaths = [os.path.join("..", "data", "test", "Rue1024a.krn"), 
    #             r'c:\Users\T460\Documents\Uni_spul\Jaar_7\Scriptie\CounterpointConsensus2\CounterpointConsensus\data\full\more_than_10\SELECTED\Bus\Bus2003-Anima_mea_liquefacta_est__Stirips_Jesse.krn',
    #             r'c:\Users\T460\Documents\Uni_spul\Jaar_7\Scriptie\CounterpointConsensus2\CounterpointConsensus\data\full\more_than_10\SELECTED\Com\Com2002a-Hodie_nobis_cycle-Hodie_nobis.krn'
    #             ]

    filepaths = find_jrp_files(DATASET_PATH, valid_files, invalid_files, anonymous_mode='skip')

    # File tracking setup
    successful_files_log = os.path.join("..", "output", "successful_files.txt")
    # Load previously successful files if the flag is enabled
    previously_successful = set()
    if SKIP_SUCCESSFUL_FILES and os.path.exists(successful_files_log):
        try:
            with open(successful_files_log, 'r', encoding='utf-8') as f:
                # Normalize paths to lowercase for case-insensitive comparison on Windows
                previously_successful = set(line.strip().lower() for line in f if line.strip())
            print(f"Loaded {len(previously_successful)} previously successful files from {successful_files_log}")
        except Exception as e:
            print(f"Warning: Could not load successful files log: {e}")
            previously_successful = set()
    # Sort set by file path
    previously_successful = sorted(previously_successful)

    # profiler = cProfile.Profile()
    # profiler.enable()

    # Collections for successful and failed processing
    all_violations: dict[str, list[RuleViolation]] = {}
    all_metadata: dict[str, dict] = {}
    full_violations_df = pd.DataFrame()

    # Error tracking
    failed_files = []
    successful_files = []
    skipped_files = []

    print(" Duplicate voice names not handled. Editorial flats not handled. Linking of salami slices to previous next occurrence wrong, when the previous next occurrence is a tied note that is in the same bar as the start of that note. E.g. Jos1408, bar 28.")
    print(f"Found {len(filepaths)} kern filepaths...")

    # --- Parsing and counterpoint analysis ---
    for file_count, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        filename_no_ext = filename.split('-')[0]  # Extract base name like "Jos1408"

        # Skip if already successfully processed
        if SKIP_SUCCESSFUL_FILES and filepath.lower() in previously_successful:
            skipped_files.append(filepath)
            continue

        try:
            if BREAK_EARLY and file_count >= BREAK_COUNT:
                print(f"Breaking early after {BREAK_COUNT} files.")
                break

            salami_slices, metadata = parse_kern(filepath)
            salami_slices, metadata = post_process_salami_slices(salami_slices, metadata, expand_metadata_flag=True)

            cp_rules = CounterpointRules()
            only_validate_rules = [
                # Rhytm
                #'brevis_at_begin_end', 'longa_only_at_endings', 
                # Dots and ties
                   'tie_into_strong_beat', 'tie_into_weak_beat',
                # Melody
                   'leap_too_large',  'leap_approach_left_opposite', 'interval_order_motion', 'successive_leap_opposite_direction', 'leap_up_accented_long_note',
                # Other aspects
                'eight_pair_stepwise',
                # Quarter note idioms
                'leap_in_quarters_balanced',
                
                # Chords
                    'non_root_1st_inv_maj', 

                # Normalization functions
                'norm_ties_contained_in_bar',
                'norm_count_tie_starts', 'norm_count_tie_ends', 'norm_tie_end_not_new_occurrence'
            ]

            violations = validate_all_rules(salami_slices, metadata, cp_rules, only_validate_rules)
            curr_df = violations_to_df(violations, metadata)
            if full_violations_df.empty:
                full_violations_df = curr_df
            else:
                full_violations_df = pd.concat([full_violations_df, curr_df], ignore_index=True)

            print(f"\t✓ JRP-ID: {metadata['jrpid']}")
            
            # Store the violations in a dict, with its JRP ID as the key.
            # In order to annotate the violations in each piece later, we save metadata with the violations for each piece.
            # In order to do so, remove the very large (and now unndeeded) key-sig info # TODO: also remove time-sigs
            curr_jrpid = metadata['jrpid']
            del metadata['key_signatures']

            all_metadata[curr_jrpid] = metadata
            all_violations[curr_jrpid] = violations
            successful_files.append(filepath)

        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            filename = os.path.basename(filepath)
            filename_no_ext = filename.split('-')[0] if '-' in filename else filename.split('.')[0]

            # Enhanced error reporting for specific error types
            if VERBOSE and ((isinstance(e, ValueError) and "too many values to unpack" in str(e)) or isinstance(e, IndexError)):
                # Common detailed error analysis
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        kern_lines = [line.rstrip('\n') for line in f.readlines()]
                    
                    # Look for relevant variables in traceback frames
                    problematic_kern_line_num = None
                    problematic_token = None
                    problematic_code_line = None
                    tb_obj = e.__traceback__
                    
                    # Find the innermost traceback frame and extract the problematic code line
                    while tb_obj.tb_next:
                        tb_obj = tb_obj.tb_next
                    
                    try:
                        filename_from_tb = tb_obj.tb_frame.f_code.co_filename
                        line_number_from_tb = tb_obj.tb_lineno
                        problematic_code_line = linecache.getline(filename_from_tb, line_number_from_tb).strip()
                    except:
                        problematic_code_line = None
                    
                    # Look for relevant variables in all traceback frames
                    tb_obj = e.__traceback__
                    while tb_obj:
                        local_vars = tb_obj.tb_frame.f_locals
                        
                        if 'original_line_idx' in local_vars:
                            line_val = local_vars['original_line_idx']
                            if isinstance(line_val, int) and 0 <= line_val < len(kern_lines):
                                problematic_kern_line_num = line_val + 1
                        if 'token' in local_vars or 'kern_token' in local_vars:
                            problematic_token = local_vars.get('token') or local_vars.get('kern_token')
                        if 'line' in local_vars and isinstance(local_vars['line'], str):
                            line_content = local_vars['line']
                            for idx, kern_line in enumerate(kern_lines):
                                if kern_line.strip() == line_content.strip():
                                    problematic_kern_line_num = idx + 1
                                    break
                        tb_obj = tb_obj.tb_next
                    

                    # Compact output with emojis
                    error_prefix = "📍 ValueError (unpack)" if isinstance(e, ValueError) else "📍 IndexError"
                    print(f"\t{error_prefix} in {filename_no_ext}", end="")
                    
                    if problematic_kern_line_num:
                        line_content = kern_lines[problematic_kern_line_num - 1]
                        print(f" | 🎯 Line {problematic_kern_line_num}: '{line_content}'", end="")
                        if problematic_token:
                            print(f" | 🎯 Token: '{problematic_token}'", end="")
                    
                    if problematic_code_line:
                        print(f" | 💥 Code: {problematic_code_line}")
                    else:
                        print()
                        
                except Exception as debug_e:
                    print(f"\t✗ {error_type} in {filename_no_ext}: {error_message}")
            else:
                print(f"\t✗ ERROR in {filename_no_ext}. {error_type}: {error_message}")
                
            
            # Track failed files
            failed_files.append(filepath)
            
            # Decide whether to continue or break
            if not CONTINUE_ON_ERROR:
                print(f"\nStopping processing due to error in file: {filepath}")
                print(f"Error: {error_type}: {error_message}")
                if not isinstance(e, IndexError):  # Don't duplicate IndexError details
                    print("Full traceback:")
                    traceback.print_exc()
                print('\n')
                raise e
            else:
                # Continue to next file
                continue
    
    # Print processing summary
    processed_count = len(successful_files) + len(failed_files)
    total_count = len(filepaths)
    
    print(f"\n" + "="*60)
    print(f"PROCESSING SUMMARY")
    print(f"="*60)
    
    print(f"Total files: {total_count}")
    print(f"Skipped (already processed): {len(skipped_files)}")
    print(f"Processed this run: {processed_count}")
    print(f"Successful: {len(successful_files)}")
    print(f"Failed: {len(failed_files)}")
    if processed_count > 0:
        print(f"Success rate: {len(successful_files)/processed_count*100:.1f}%")
    print()

    # Save successful files to log (only write if file doesn't exist already)
    if successful_files:
        try:
            # Only proceed if the log file doesn't already exist
            if not os.path.exists(successful_files_log):
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(successful_files_log), exist_ok=True)
                
                # Write the new log file
                with open(successful_files_log, 'w', encoding='utf-8') as f:
                    for filepath in successful_files:
                        f.write(f"{filepath}\n")
                
                print(f"Created new successful files log with {len(successful_files)} files: {successful_files_log}")
            else:
                print(f"Successful files log already exists, not overwriting: {successful_files_log}")
                print(f"Would have added {len(successful_files)} files to the log")
                
        except Exception as e:
            print(f"Warning: Could not save successful files log: {e}")

    if failed_files:
        print(f"\n" + "-"*40)
        print(f"FAILED FILES ({len(failed_files)}):")
        print(f"-"*40)
        
        # Limit the number of files displayed
        max_display = 10
        files_to_display = failed_files[:max_display]
        
        for failed_file in files_to_display:
            filename_base = os.path.basename(failed_file)
            # Extract just the filename without extension for cleaner display
            filename_no_ext = filename_base.split('-')[0]
            print(f"  • {filename_no_ext}, {failed_file}")
        
        # Show truncation message if there are more files
        if len(failed_files) > max_display:
            remaining_count = len(failed_files) - max_display
            print(f"  ... and {remaining_count} more files")
    print('\n')

    # Sort the final DF
    full_violations_df = sort_df_by_rule_id(full_violations_df)

    # Save DataFrame to CSV file
    if not BREAK_EARLY:
        csv_output_path = os.path.join("..", "output", "violations", "violations_output.csv")
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)  # Create directory if it doesn't exist
        full_violations_df.to_csv(csv_output_path, index=False)
        print(f"Violations DataFrame saved to: {csv_output_path}")
        print(f"Shape: {full_violations_df.shape}")

    # --- Annotate violations
    annotate_violations_flag = True
    if annotate_violations_flag:
        destination_dir = os.path.join("..", "output", "annotated")
        annotate_all_kern(destination_dir, all_metadata, all_violations, 
                          use_rule_ids=True, overwrite=True, maintain_jrp_structure=True,
                          verbose=False)
        # Report on how many files were annotated
        count = len(all_metadata)
        print(f"Annotated {count} file{'s' if count!=1 else ''} and saved to: {destination_dir}")


    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    
    print('\n')
    pd.set_option('display.max_columns', None)
    #display(full_violations_df); print()


    # --- Classify


    # Analyze classification


    return


if __name__ == "__main__":
    main()


from Note import Note, SalamiSlice
import os
from collections import defaultdict

from counterpoint_rules import RuleViolation


def annotate_all_kern(destination_dir: str, all_metadata: dict[str, dict], all_violations: dict[str, list[RuleViolation]],
                        overwrite: bool = True, maintain_jrp_structure: bool = True, 
                        save_as_txt: bool = False, include_invisible_comments: bool = True, verbose: bool = False) -> None:
    """Annotate all kern files in the given list of file paths.

    Args:
        destination_dir (str): Directory to save the annotated files.
        all_metadata (dict[str, dict]): Metadata for each piece.
        all_violations (dict[str, list[RuleViolation]]): Rule violations for each piece.
        use_rule_ids (bool, optional): If True, use rule IDs in annotations. Defaults to True.
        overwrite (bool, optional): If True, overwrite existing annotated files. Defaults to True.
        maintain_jrp_structure (bool, optional): If True, maintain JRP directory structure. Defaults to True.
        save_as_txt (bool, optional): If True, save as .txt instead of .krn. Defaults to False.
        include_invisible_comments (bool, optional): If True, include invisible comments with rule names. Defaults to True.
        verbose (bool, optional): If True, print additional information. Defaults to False.
    """
    for jrpid, metadata in all_metadata.items():
        src_path = metadata["src_path"]
        base_krn = os.path.basename(src_path)
        
        # Choose file extension based on flag
        if save_as_txt:
            annotated_filename = f"annotated__{base_krn.replace('.krn', '.txt')}"
        else:
            annotated_filename = f"annotated__{base_krn}"
        
        # If maintain_jrp_structure is True, create the directory for the current composer, if not yet present
        if maintain_jrp_structure:
            composer = jrpid[:3]  
            composer_dir = os.path.join(destination_dir, composer)
            os.makedirs(composer_dir, exist_ok=True)
            dst_path = os.path.join(composer_dir, annotated_filename)
        else:
            dst_path = os.path.join(destination_dir, annotated_filename)

        # Get the rule violations for this piece
        violations = all_violations[jrpid]

        # Annotate a copy of the .krn source with comments for each violation   
        annotate_kern(src_path, dst_path, violations, metadata, 
                      overwrite=overwrite, 
                      include_invisible_comments=include_invisible_comments, verbose=verbose)

        if verbose:
            print(f"Annotated {jrpid} and saved to {dst_path}")


def annotate_kern(src_path: str, dst_path: str, violations: dict, metadata: dict, 
                  overwrite: bool, include_invisible_comments: bool, verbose: bool) -> None:
    """
    Inserts comment lines above each line in the .krn file that has a violation.
    Each violation includes the line index ("original_line_num"), voice(s),
    and rule_name(s). 
    
    Format:
    *  *color:red  *   *                         (color directive line)
    !  !LO:TX:a:t=<rule_id1>, <rule_id2>  !   !  (visible text annotation line with rule IDs)
    !  !LO:TX:a:vis=0:color=none:t=(<rule_name1>, <note_names1>),   !   !  (invisible text with rule names & notes)
    <original line from source file>
    *  *color:black  *   *                       (reset color directive line)
    
    '*' for the color, and '!' for the comment, denotes no change for that voice. Each token is separated by a tab.
    """
    # Check if file exists and respect overwrite preference
    if os.path.exists(dst_path) and not overwrite:
        if verbose:
            print(f"Skipping {dst_path}: File exists and overwrite=False")
        return
    
    # Create reverse mapping from analysis voice order back to original file voice order
    voice_sort_map = metadata['voice_sort_map']
    reverse_voice_map = {analysis_voice_idx:original_file_voice_idx for analysis_voice_idx, original_file_voice_idx in voice_sort_map.items()}
    
    # Create a mapping of line numbers to RuleViolation objects for each voice.
    # This will look like: {line_num: {voice_index: [RuleViolation, RuleViolation, ...]}}
    violations_to_line_voice_map = defaultdict(lambda: defaultdict(list))

    # Populate violations_to_line_voice_map (filter out normalization functions)
    for rule_function_name, rule_violations in violations.items():
        # Skip empty rule violations
        if not rule_violations:
            continue

        # Skip normalization functions (rules where last part after ',' starts with 'N' followed by digit)
        rule_id = rule_violations[0].rule_id  # All violations for this rule should have the same rule_id
        if rule_id[0] == 'N' and rule_id[1:].isdigit():
            continue
            
        for v in rule_violations:
            line_num = v.original_line_num

            # voice_indices can be a single int or a tuple
            if isinstance(v.voice_indices, tuple):
                for voice_idx_from_violation in v.voice_indices:
                    original_voice_mapped_idx = reverse_voice_map[voice_idx_from_violation]
                    violations_to_line_voice_map[line_num][original_voice_mapped_idx].append(v) # Store the RuleViolation object
            elif isinstance(v.voice_indices, int):
                original_voice_mapped_idx = reverse_voice_map[v.voice_indices]
                violations_to_line_voice_map[line_num][original_voice_mapped_idx].append(v) # Store the RuleViolation object
            else:
                raise ValueError(f"Unexpected type for voice_indices: {type(v.voice_indices)}. Expected int or tuple.")    # Determine number of voices from original file structure
    num_original_voices = len(metadata['unsorted_voice_order'])
    
    # Calculate total number of columns in original file (kern + text)
    if metadata.get('has_text_columns', False):
        total_columns = len(metadata['note_column_indices']) + len(metadata['text_column_indices'])
    else:
        total_columns = num_original_voices

    # Go line-by-line, writing both comment lines and the original line
    with open(src_path, "r", encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin):
            # Check if there are violations for this line
            if line_idx in violations_to_line_voice_map:
                # Create annotation lines for original file structure (including text columns)
                visible_comment_tokens = ["!"] * total_columns
                invisible_comment_tokens = ["!"] * total_columns
                # Track which voices have violations, to color only those voices red
                voices_with_violations = set()
                
                
                for original_voice_idx, stored_violation_objects in violations_to_line_voice_map[line_idx].items():
                    rule_ids_for_this_voice = []
                    rule_details_for_this_voice = []  # Will store (rule_name, note_names) pairs
                    
                    for v_obj in stored_violation_objects: # v_obj is a RuleViolation object
                        rule_ids_for_this_voice.append(str(v_obj.rule_id))
                        
                        # Format note_names appropriately based on type
                        note_names = v_obj.note_names
                        if isinstance(note_names, tuple) or isinstance(note_names, list):
                            formatted_notes = " ".join(str(n) for n in note_names)
                        else:
                            formatted_notes = str(note_names)
                            
                        # Add rule name with its associated note names
                        rule_details_for_this_voice.append(f"({v_obj.rule_name}, {formatted_notes})")
                      # Create visible comment with rule IDs
                    # Map to correct column position in original file
                    rule_ids_str = ", ".join(rule_ids_for_this_voice)
                    if metadata.get('has_text_columns', False):
                        # For text columns, we need to map to the kern column position
                        kern_column_idx = metadata['note_column_indices'][original_voice_idx]
                        visible_comment_tokens[kern_column_idx] = f"!LO:TX:a:t={rule_ids_str}"
                    else:
                        visible_comment_tokens[original_voice_idx] = f"!LO:TX:a:t={rule_ids_str}"
                    
                    # Create invisible comment with rule names and note names
                    rule_details_str = "; ".join(rule_details_for_this_voice)
                    if metadata.get('has_text_columns', False):
                        # For text columns, we need to map to the kern column position
                        kern_column_idx = metadata['note_column_indices'][original_voice_idx]
                        invisible_comment_tokens[kern_column_idx] = f"!LO:TX:a:vis=0:color=none:t={rule_details_str}"
                    else:
                        invisible_comment_tokens[original_voice_idx] = f"!LO:TX:a:vis=0:color=none:t={rule_details_str}"
                    
                    voices_with_violations.add(original_voice_idx)                # Create the color directive line (set affected voices to red)
                color_tokens = ["*"] * total_columns
                for voice_idx in voices_with_violations:
                    # Map voice index to correct column position in original file
                    if metadata.get('has_text_columns', False):
                        # For text columns, we need to map to the kern column position
                        kern_column_idx = metadata['note_column_indices'][voice_idx]
                        color_tokens[kern_column_idx] = "*color:red"
                    else:
                        color_tokens[voice_idx] = "*color:red"
                
                color_line = "\t".join(color_tokens) + "\n"
                fout.write(color_line)

                # Write the visible comment line (rule IDs)
                visible_comment_line = "\t".join(visible_comment_tokens) + "\n"
                fout.write(visible_comment_line)
                
                # Write the invisible comment line only if enabled
                if include_invisible_comments:
                    invisible_comment_line = "\t".join(invisible_comment_tokens) + "\n"
                    fout.write(invisible_comment_line)

                # Write the original line
                fout.write(line)
                  # Create the reset color directive line (set affected voices back to black)
                reset_tokens = ["*"] * total_columns
                for voice_idx in voices_with_violations:
                    # Map voice index to correct column position in original file
                    if metadata.get('has_text_columns', False):
                        # For text columns, we need to map to the kern column position
                        kern_column_idx = metadata['note_column_indices'][voice_idx]
                        reset_tokens[kern_column_idx] = "*color:black"
                    else:
                        reset_tokens[voice_idx] = "*color:black"
                
                reset_line = "\t".join(reset_tokens) + "\n"
                fout.write(reset_line)
            else:
                # Just write the original line if there are no violations
                fout.write(line)

    return



if __name__ == "__main__":
    # Example usage
    pass

from Note import Note, SalamiSlice
import os
from collections import defaultdict

from counterpoint_rules import RuleViolation


def annotate_all_kern(kern_filepaths: list[str], destination_dir: str, all_metadata: dict[str, dict], all_violations: dict[str, list[RuleViolation]],
                        overwrite: bool = True, maintain_jrp_structure: bool = True, verbose: bool = False,) -> None:
    """Annotate all kern files in the given list of file paths.

    Args:
        kern_filepaths (list[str]): List of file paths to kern files.
        destination_dir (str): Directory to save the annotated files.
        all_metadata (dict[str, dict]): Metadata for each piece.
        all_violations (dict[str, list[RuleViolation]]): Rule violations for each piece.
        maintain_jrp_structure (bool, optional): If True, maintain JRP directory structure. Defaults to True.
        verbose (bool, optional): If True, print additional information. Defaults to False.
        overwrite (bool, optional): If True, overwrite existing annotated files. Defaults to False.
    """
    for jrpid, metadata in all_metadata.items():
        if verbose:
            print(f"Annotating {jrpid}...")
        
        # Annotate the violations in a copy of the kern file
        src_path = metadata["src_path"]

         # Decide the annotated filename
        base_krn = os.path.basename(src_path)
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
        annotate_kern(src_path, dst_path, violations, metadata, overwrite=overwrite, verbose=verbose)

        if verbose:
            print(f"Annotated {jrpid} and saved to {dst_path}")


def annotate_kern(src_path: str, dst_path: str, violations: dict, metadata: dict, overwrite: bool, verbose: bool) -> None:
    """
    Inserts comment lines above each line in the .krn file that has a violation.
    Each violation includes the line index ("original_line_number"), voice(s),
    and rule_name(s). 
    
    Format:
    *  *color:red  *   *                         (color directive line)
    !  !LO:TX:a:t=<rule_name1>, <rule_name2>  !   !  (text annotation line)
    <original line from source file>
    *  *color:black  *   *                       (reset color directive line)
    
    '*' denotes no change for that voice. Each token is separated by a tab.
    """
    # Check if file exists and respect overwrite preference
    if os.path.exists(dst_path) and not overwrite:
        if verbose:
            print(f"Skipping {dst_path}: File exists and overwrite=False")
        return
    
    # Create a mapping of line numbers to rule names for each voice
    violations_to_line_voice_map = defaultdict(lambda: defaultdict(list))
    voice_sort_map = metadata['voice_sort_map']
    reverse_voice_map = {v:k for k,v in voice_sort_map.items()}

    # Populate violations_to_line_voice_map
    for rule_name, rule_violations in violations.items():
        for v in rule_violations:
            line_num = v.original_line_number

            # voice_indices can be a single int or a tuple
            if isinstance(v.voice_indices, tuple):
                for voice in v.voice_indices:
                    original_voice = reverse_voice_map[voice]
                    violations_to_line_voice_map[line_num][original_voice].append(rule_name)
            else:
                original_voice = reverse_voice_map[v.voice_indices]
                violations_to_line_voice_map[line_num][original_voice].append(rule_name)

    # Go line-by-line, writing both comment lines and the original line
    with open(src_path, "r", encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            # Check if there are violations for this line
            if i in violations_to_line_voice_map:
                # Create the text annotation line with rule names
                comment_tokens = ["!"] * len(voice_sort_map)
                voices_with_violations = set()
                
                for voice_idx, rule_names in violations_to_line_voice_map[i].items():
                    # Use set to avoid duplicates
                    unique_rule_names = sorted(set(rule_names))
                    rule_names_str = ", ".join(unique_rule_names)
                    comment_tokens[voice_idx] = f"!LO:TX:a:t={rule_names_str}"
                    voices_with_violations.add(voice_idx)

                # Create the color directive line (set affected voices to red)
                color_tokens = ["*"] * len(voice_sort_map)
                for voice_idx in voices_with_violations:
                    color_tokens[voice_idx] = "*color:red"
                
                color_line = "\t".join(color_tokens) + "\n"
                fout.write(color_line)

                # Join the comment tokens with tabs and write after the color line
                comment_line = "\t".join(comment_tokens) + "\n"
                fout.write(comment_line)

                # Write the original line
                fout.write(line)
                
                # Create the reset color directive line (set affected voices back to black)
                reset_tokens = ["*"] * len(voice_sort_map)
                for voice_idx in voices_with_violations:
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
    
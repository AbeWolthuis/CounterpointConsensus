import os
import importlib
from IPython.display import display
from pprint import pprint
from collections import defaultdict
from typing import List, Dict, Tuple
import copy
import pandas as pd
import cProfile, pstats


import traceback


''' Our own modules '''
from constants import DURATION_MAP, TRIPLET_DURATION_MAP, INFINITY_BAR, PITCH_TO_MIDI
from counterpoint_rules import RuleViolation
from data_preparation import violations_to_df, sort_df_by_rule_id
from annotate_kern import annotate_all_kern
from data_preparation import find_jrp_files

# These modules are changed often, so we force reload them every time.
import Note
import counterpoint_rules
import process_salami_slices
# importlib.reload(Note)
# importlib.reload(counterpoint_rules)
# importlib.reload(process_salami_slices)

from Note import Note, SalamiSlice
from counterpoint_rules_base import CounterpointRulesBase
from process_salami_slices import post_process_salami_slices


DEBUG = False
DEBUG2 = False



metadata_template = {
    # JRP sepcified metadata
    'COM': '', # Composer 
    'COA_entries': {},  # Will store all COA entries with their numbers
    'CDT': '', # Copmoser's dates record
    'jrpid': '', # josquin research project ID
    'attribution-level@Jos': '',
    'SEGMENT': '', # filename

    'voices': [], # list with number of voices for each score-movement  # TODO: support voices changing during piece
    'voice_names': [], # list with voice names for each score-movement  # TODO: support voices changing during piece
    
    'key_signatures': [], # list of key signatures for each score-movement
    'time_signatures': [], # list of time signatures for each voice at each change point
    
    # Metadata we add ourselves (e.g. is not in the kern file explicitly)
    'voice_sort_map' : {}, # maps the order of the voices in the .krn file to its order as used here; e.g. if soprano is the 4th voice (3rd index), we map {0:3}, since the soprano should be the treated as the upper (index 0) voice.
    'section_starts': [], # list of the barnumbers of the first bars of each section 
    'section_ends': [],   # list of the barnumbers of the last bars of each section
    'total_bars': 0,      # total number of bars in the piece (length of piece)
}
metadata_linestarts_map = {
    '!!!COM': 'COM',
    '!!!CDT': 'CDT',
    '!!!jrp': 'jrpid',
    '!!!voi': 'voices',
    '!!attr': 'attribution-level@Jos',
    '!!!!SE': 'SEGMENT'
}
# Note, attribution-level@Jos only for Josquin pieces.
# Note, voice-names should be handled seperately  

accidentals = {
    '#': 1,  '##': 2,  '###': 3,
    '-': -1, '--': -2, '---': -3,
    'n': 0
}
ignored_tokens = {
    '\\' : 'down-stem',
    '/' : 'up-stem',
    ';' : 'fermata',
    'L' : 'start_beam',
    'LL' : 'start_double_beam',
    'J' : 'end_beam',
    'JJ' : 'end_double_beam',
    'y' : 'uncertain_character',
    'X' : 'editorial_interpretation',
    'N' : 'signum_congruentiae'

}
ignored_line_tokens = {}

# Used to parse the key signature(s)
key_signature_template = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0}


def parse_kern(kern_filepath) -> Tuple[List, Dict]:
    metadata = copy.deepcopy(metadata_template)  # Deep copy since template is created with lists as values
    metadata['src_path'] = kern_filepath
    
    with open(kern_filepath, 'r', encoding='utf-8') as f: 
        bar_count = 0 
        
        # Track text column information
        has_text_columns = False
        text_column_indices = []
        note_column_indices = []
        text_detection_done = False
    
        # Store tuples: (original_line_idx, line_string)
        note_section_with_indices: List[Tuple[int, str]] = []
        lines = f.readlines()

        for line_idx, line in enumerate(lines):
            # --- Detect text columns from the first data line with tokens per voice ---
            if not text_detection_done and line.startswith('**'):
                # Parse column headers to detect text columns
                column_headers = line.strip().split('\t')
                text_column_indices = [i for i, header in enumerate(column_headers) if header == '**text']
                note_column_indices = [i for i, header in enumerate(column_headers) if header == '**kern']
                
                if text_column_indices:
                    has_text_columns = True
                    
                    # Validate that we have some kern columns
                    if not note_column_indices:
                        raise ValueError(f"Found **text columns but no **kern columns: {column_headers}")
                    
                    # Validate that all columns are either **kern or **text
                    for i, header in enumerate(column_headers):
                        if header not in ['**kern', '**text']:
                            raise ValueError(f"Unexpected column header '{header}' at position {i}. Expected only **kern or **text.")
                    
                    if DEBUG: print(f"Detected text columns at indices: {text_column_indices}. Note columns at indices: {note_column_indices}")
                
                text_detection_done = True
                continue
              # --- Filter text tokens from ALL subsequent lines (if text columns detected) ---
            if has_text_columns and not line.startswith('!'):
                # For metadata lines (*), barlines (=), and note lines - filter out text tokens
                all_tokens = line.strip().split('\t')
                expected_tokens = len(note_column_indices) + len(text_column_indices)
                
                # Only apply filtering if we have the expected number of tokens
                if len(all_tokens) == expected_tokens:
                    # Extract only note tokens, ignoring text tokens
                    note_tokens = [all_tokens[i] for i in note_column_indices]
                    # Reconstruct line with only note tokens
                    line = '\t'.join(note_tokens) + '\n'
                # If token count doesn't match, let the line pass through for normal error handling
            
            # --- Metadata Parsing ---
            if line.startswith('!'):
                if line.startswith('!!!COA'):
                    # Handle attributed composers, to be used instead of COM when COM is not present.
                    coa_key = line.split(':')[0][3:]  # Extract "COA", "COA2", "COA3", etc.
                    coa_value = line.split(':')[1].strip()
                    metadata['COA_entries'][coa_key] = coa_value
                elif line[0:6] in metadata_linestarts_map.keys():
                    # Splt '!!!jrpid: xxx' to 'jrpid',' xxx'
                    a = line.split(':')[0][0:6]
                    metadata_key, metadata_value = metadata_linestarts_map[a], line.split(':')[1].strip()
                    metadata[metadata_key] = metadata_value
                elif line.startswith('!!section'):
                    # Record the end of the previous section (if not the very first section)
                    if metadata['section_starts']:
                        # The previous section ends at the bar before this section starts
                        metadata['section_ends'].append(bar_count - 1)
                    # Record the start of the new section
                    metadata['section_starts'].append(bar_count)
            elif line.startswith('*'):
                # Parse voice names
                if line.startswith('*I'):
                    # If diffferent voices are partway through the piece, raise an error
                    if bar_count > 0:
                        raise NotImplementedError("Different voices mid-piece not yet implemented.")
                    
                    if line[2] == 'v':   # v represents the number of voices
                        continue
                    elif line[2] == "'": # ' represents short voice names
                        continue
                    elif line[2] == '"': # " represents long voice names
                        # Split on tabs to get each voice's full name token
                        voice_tokens = line.strip().split('\t')
                        voice_names = []
                        
                        for token in voice_tokens:
                            if token.startswith('*I"'):
                                # Remove the '*I"' prefix and keep the rest as the voice name
                                voice_name = token[3:].replace(' ', '')  # Remove '*I"' (3 characters)
                                voice_names.append(voice_name)
                        
                        metadata['voice_names'] = voice_names
                        
                        # TODO: support voices changing during piececes count
                        # barline_tuple = (last_metadata_update_bar, INFINITY_BAR)e_names):
                        # metadata['voice_names'].append((barline_tuple, voice_names)))}) does not match voices count ({metadata['voices']})")
                    else:
                        raise NotImplementedError("Something in parsing voices (*I) not yet implemented.")
                # Parse key signatures for all voices
                elif line.startswith('*k'):
                    metadata_dict = parse_keysig(line, metadata, bar_count, line_idx)
                    if DEBUG: print(f"Found key signatures at bar {bar_count}: {metadata_dict['key_signatures'][-1][1]}")
                # Parse time signatures for all voices  
                elif '*M' in line:
                    # Check if any token starts with *M but not *MM (more robust - checks for *M anywhere in line)
                    tokens = line.strip().split('\t')
                    has_timesig = any(token.startswith('*M') and not token.startswith('*MM') for token in tokens)
                    
                    if has_timesig:
                        # Parse time signatures for all voices
                        metadata_dict = parse_timesig(line, metadata, bar_count)
                        # if DEBUG: print(f"Found time signatures at bar {bar_count}: {metadata_dict['time_signatures'][-1][1]}")
                elif line.startswith('*MM'):
                    # Metronome marking, present in some files, but we don't use this.
                    continue
                elif '*met' in line:
                    # TODO: check if we need to implement this later (mensural)
                    continue
                elif '*rscale' in line:
                    # Rhythm scaling directive for rendering; notes seem to be encoded in a way that matches the other voices
                    continue
                elif line.startswith(('**kern', '*staff', '*clef')):
                    # Ignored headers
                    continue
                elif line.startswith('*part'):
                    # Part numbering like "*part4	*part4	*part3	*part3"
                    continue
                elif line.startswith('*-'):
                    # Ignored barlines
                    continue
                elif line.startswith('*>'):
                    # Lines indicating structure, or the begin of the structure.
                    continue
                elif line.startswith('*tb'):
                    # Tablature information - can be ignored for counterpoint analysis
                    continue
                elif '*lig' in line or '*Xlig' in line:
                    # Ligature notation - can be ignored for counterpoint analysis
                    continue
                elif '*col' in line:
                    # Column information - can be ignored for counterpoint analysis
                    continue
                elif '*d:' in line or '*a:' in line or '*F:' in line:
                    # Key designation like "*d:" or "*a:" - can be ignored for counterpoint analysis
                    continue
                else:
                    raise NotImplementedError(f"Some metadata starting with * not yet implemented. Line {line_idx}: {line}")            # --- Barline Handling ---
            elif line.startswith('='):
                # Check if this is a visual barline without bar numbers FIRST (e.g., "=", "=-", "==")
                # Extract only digits from the first barline token after stripping all visual markers
                first_token = line.split()[0] if line.split() else line.strip().split('\t')[0]
                
                cleaned_token = strip_visual_markers(first_token)
                digits_only = ''.join(c for c in cleaned_token if c.isdigit())
                
                if not digits_only:
                    # This is a visual barline without a bar number - skip it entirely
                    if DEBUG: print(f"Skipping visual barline at line {line_idx}: '{line.strip()}'")
                    continue
                
                # If we reach here, it's a numbered barline (visible or invisible) - now do consistency checks
                if has_text_columns:
                    # Validate that all barline tokens are identical after stripping visual markers
                    note_tokens = line.strip().split('\t')
                    if len(note_tokens) > 1:
                        first_barline_clean = strip_visual_markers(note_tokens[0])
                        if not all(strip_visual_markers(token) == first_barline_clean for token in note_tokens):
                            raise ValueError(f"Inconsistent barline tokens at line {line_idx}: {note_tokens}")
                
                # Update bars according to the barnumber found
                try:
                    bar_num = int(digits_only)
                    if bar_num < bar_count:
                        raise ValueError(f"Bar number {bar_num} is lower than the previous bar number {bar_count}") 
                    bar_count = bar_num
                    
                    # Add barline to note section WITH its original index
                    note_section_with_indices.append((line_idx, line))    
                except Exception as e:
                    raise ValueError(f"Could not parse bar number from line: '{line.strip()}'. Error: {str(e)}")
            elif False:
                # try-except-else is such weird syntax
                continue
            # --- Note Line Handling ---
            else: 
                # Note lines are already filtered above if needed
                if has_text_columns:
                    # Validate that we have the expected number of note tokens
                    note_tokens = line.strip().split('\t')
                    if len(note_tokens) != len(note_column_indices):
                        raise ValueError(f"Note line {line_idx} has {len(note_tokens)} tokens but expected {len(note_column_indices)} note tokens after filtering: '{line.strip()}'")
                
                note_section_with_indices.append((line_idx, line))
    
    # Store text column information in metadata for reference
    metadata['has_text_columns'] = has_text_columns
    metadata['text_column_indices'] = text_column_indices
    metadata['note_column_indices'] = note_column_indices
    
    ''' Post-process the meta-data. '''
    
    metadata['voices'] = int(metadata['voices'])

    # Handle composer attribution logic with priority
    if not metadata['COM'] and metadata['COA_entries']:
        # Find the highest priority COA entry
        # Priority: COA > COA1 > COA2 > COA3 > etc.
        if 'COA' in metadata['COA_entries']:
            # COA without number has highest priority
            metadata['COM'] = metadata['COA_entries']['COA'] + " (COA)"
        else:
            # Find the lowest numbered COA entry (including COA1)
            coa_numbers = []
            for key in metadata['COA_entries'].keys():
                if key == 'COA1':
                    # COA1 gets priority 1
                    coa_numbers.append(1)
                elif key.startswith('COA') and len(key) > 3:  # COA2, COA3, etc.
                    try:
                        num = int(key[3:])  # Extract number after "COA"
                        coa_numbers.append(num)
                    except ValueError:
                        continue
        
            if coa_numbers:
                # Use the lowest numbered COA entry
                min_num = min(coa_numbers)
                if min_num == 1:
                    coa_key = 'COA1'
                else:
                    coa_key = f'COA{min_num}'
                metadata['COM'] = metadata['COA_entries'][coa_key] + f" ({coa_key})"

    ''' Parse the notes into salami slices. '''
    salami_slices = parse_note_section(note_section_with_indices, metadata)
    
    return salami_slices, metadata

def parse_note_section(note_section_with_indices: List[Tuple[int, str]], metadata: dict) -> List:
    """ Parse the notes into salami slices """
    salami_slices = []
    accidental_tracker = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0} 
    accidental_trackers = [dict(a=0, b=0, c=0, d=0, e=0, f=0, g=0) for _ in range(metadata['voices'])]
    current_bar = 1  # Start with bar 1
    
    # A per-voice tie tracker: one bool per voice
    current_tie_open_flags = [False] * int(metadata['voices'])

    # Add tuplet trackers: one per voice to track if we're inside a V...Z tuplet
    tuplet_open_flags = [False] * int(metadata['voices'])
    tuplet_note_count = [0] * int(metadata['voices'])  # Count notes within each tuplet


    for index_of_line_tuple_in_note_section, (original_line_idx, line) in enumerate(note_section_with_indices):
        new_salami_slice = SalamiSlice(
            num_voices=int(metadata['voices']),
            bar = current_bar,
            original_line_num=original_line_idx,
        )        

        

        tokens = line.split()
        # Check token count consistency (only for non-barlines)
        if len(tokens) != metadata['voices'] and not line.startswith('='):
            raise ValueError(f"Line has {len(tokens)} tokens but metadata specifies {metadata['voices']} voices: '{line.strip()}'")

        # --- Note Processing Loop ---
        # Generate the new note based on the token
        for token_idx, token in enumerate(tokens):
            # Notice, current_tie_open_flags is updated IN PLACE due to passing it as a reference.
            # We create a shallow copy of the state of last round to preserve it comparing the current state to the previous state
            flags_before_token_processing = current_tie_open_flags[:]
            tuplet_flags_before = tuplet_open_flags[:]

            # Create the new note objects by parsing the token and state of the flags
            new_note, current_bar,            accidental_trackers, current_tie_open_flags, tuplet_open_flags, tuplet_note_count = kern_token_to_note(
                token,current_bar, token_idx, accidental_trackers, current_tie_open_flags, tuplet_open_flags, tuplet_note_count, include_editorial_accidentals=True
            )
            
            # --- Set tuplet flags based on the tuplet state changes ---
            if new_note.note_type == 'note':
                # Check if tuplet state changed for this voice
                was_in_tuplet = tuplet_flags_before[token_idx]
                is_in_tuplet = tuplet_open_flags[token_idx]
                
                if is_in_tuplet and not was_in_tuplet:
                    # Just entered a tuplet - this is the start
                    new_note.is_triplet_start_through_V = True
                elif is_in_tuplet and was_in_tuplet:
                    # Continue in tuplet - this is middle
                    new_note.is_triplet_middle_between_VZ = True
                elif not is_in_tuplet and was_in_tuplet:
                    # Just exited tuplet - this is the end
                    new_note.is_triplet_end_through_Z = True
            
            # --- Set Tie Flags using flags_before_token and current_tie_open_flags ---
            prev_note = None
            if len(salami_slices) > 0:
                # Get the most recent slice and check if it has a note for this voice
                prev_slice = salami_slices[-1]
                if prev_slice.notes[token_idx] is not None:
                    prev_note = prev_slice.notes[token_idx]

            
            is_tied_now = current_tie_open_flags[token_idx]
            was_tied_before = flags_before_token_processing[token_idx]
        


            # First, detect if this is a tie-start: a note is the start of a tie IF either of the following is true: 
            #   1) it is tied, and the previous note is not.
            #   2) it is tied, and the previous is a tie-end (in which case the previous note is allowed to be tied)

            # Logic for determining tie properties:
            # 1. Tie start: currently tied AND (wasn't tied before OR previous note had different pitch)
            # 2. Tie end: not tied now AND was tied before. This note is also set as being tied.
            # 3. Tie continuation: tied now AND was tied before, and the previous note is tied and not a tie-end.

            if is_tied_now and not was_tied_before: # and (prev_note.is_tied) and (not prev_note.is_tie_end) and (prev_note.midi_pitch == new_note.midi_pitch):
                # This is the start of a new tie
                new_note.is_tie_start = True
                new_note.is_tied = True
                new_note.is_new_occurrence = True  # This is a new occurrence of a note
            elif is_tied_now and was_tied_before:
                # Check if this continues the same tie, or starts a new tie.
                if (prev_note.is_tied) and (not prev_note.is_tie_end):
                    # This is a continuation of a tie with the same pitch
                    new_note.is_tied = True
                    new_note.is_tie_continuation
                elif (prev_note.is_tied) and (prev_note.is_tie_end):
                    # This is the start of a new tie (different pitch)
                    new_note.is_tie_start = True
                    new_note.is_tied = True
                else:
                    raise ValueError(f"Unexpected tie state: is_tied_now={is_tied_now}, was_tied_before={was_tied_before}, prev_note={prev_note}, new_note={new_note}, in voice {token_idx} in bar {current_bar}.")
            # Detect tie-end
            elif not is_tied_now and was_tied_before:
                new_note.is_tie_end = True
                new_note.is_tied = True
            elif not is_tied_now and not was_tied_before:
                # This is not a tie, so we do not set any tie flags
                pass
            else:
                raise ValueError(f"Unexpected tie state: is_tied_now={is_tied_now}, was_tied_before={was_tied_before}, prev_note={prev_note}, new_note={new_note}")

            

            new_salami_slice.add_note(note=new_note, voice=token_idx)

            # TODO: how do ties work with sections right now? sections are not considered atm, does that matter?
            # !!! TODO: issue with period notes; I think we should skip over period notes and keep looking back for the original note.
            # However, we must indeed note that period notes should ALSO have their ties set. This is maybe not happening right now?
            # Period notes have their own note type, so we must include that in the detection of which notes to set the tie to etc


            # Constructing the ntoe is now finished, but the type of note we encountered can have implications for the metadata, and the rest of the parsing.
            
            if new_note.note_type == 'barline':
                # Reset accidental trackers
                accidental_trackers = [dict(a=0, b=0, c=0, d=0, e=0, f=0, g=0) for _ in range(metadata['voices'])]
            if new_note.note_type == 'final_barline':               
                # # Check for remaining non-comment/metadata lines (thus: notes) after this one. That should not be possible,
                remaining_lines_with_indices = note_section_with_indices[index_of_line_tuple_in_note_section + 1:]
                for remaining_orig_idx, remaining_line in remaining_lines_with_indices:
                    if (not remaining_line.strip()) or (not remaining_line.startswith(('!', '*'))):
                        raise ValueError(f"Found unexpected line with notes or barlines after the final barline.: '{remaining_line}'")
            elif new_note is None:
                raise ValueError(f"Token '{token}' in line '{line}' lead to new_note being None.")
            
        salami_slices.append(new_salami_slice)              

    # Set the final bar number for the last slice
    metadata['total_bars'] = current_bar

    return salami_slices

def kern_token_to_note(
        kern_token: str,
        current_bar: int,
        token_idx: int, # Index of the current voice/token in the line; i.e. the voice (0-indexed)
        accidental_trackers: list[dict],
        tie_open_flags: list[bool],
        tuplet_open_flags: list[bool],
        tuplet_note_count: list[int],
        include_editorial_accidentals = True,
    ) -> Tuple[Note, int, list[dict], list[bool], list[bool], list[int]]:
    
    new_note = Note()

    i = 0 
    while i < len(kern_token):
        try:
            c = kern_token[i]

            # Check for anti-metric figure (look ahead for %)
            # This handles both regular durations (like '2%9r') and triplet durations (like '3%2.B-')
            if (c in DURATION_MAP or c in TRIPLET_DURATION_MAP) and i+1 < len(kern_token) and kern_token[i+1] == '%':
                # Handle anti-metric figure like '2%9r', '4%3e', or '3%2.B-'
                # Find the end of the multiplier (next non-digit character after %)
                percent_pos = kern_token.find('%', i)
                multiplier_start = percent_pos + 1
                multiplier_end = multiplier_start
                while multiplier_end < len(kern_token) and kern_token[multiplier_end].isdigit():
                    multiplier_end += 1
                
                if multiplier_end == multiplier_start:
                    raise ValueError(f"No multiplier found after '%' in token '{kern_token}'")
                
                # Extract the components
                base_duration_char = c
                multiplier = kern_token[multiplier_start:multiplier_end]
                
                # Calculate the duration based on which map the character belongs to
                if base_duration_char in DURATION_MAP:
                    base_duration = DURATION_MAP[base_duration_char]
                else:
                    # Must be in TRIPLET_DURATION_MAP
                    base_duration = TRIPLET_DURATION_MAP[base_duration_char]
                
                new_note.duration = base_duration * int(multiplier)
                new_note.is_measured_differently = True
                
                # Check what comes after the multiplier
                if multiplier_end < len(kern_token):
                    next_char = kern_token[multiplier_end]
                    
                    # Check for dots after the multiplier
                    if next_char == '.':
                        dot_count = count_dots_at_position(kern_token, multiplier_end)
                        new_note.duration *= 1.5**dot_count
                        new_note.is_dotted = True  # Set dotted property
                        i = multiplier_end + dot_count - 1
                    elif next_char == 'r':
                        # Rest with anti-metric figure
                        new_note.note_type = 'rest'
                        i = multiplier_end  # Will be incremented by 1 at end of loop
                    elif next_char in PITCH_TO_MIDI:
                        # Note follows immediately after multiplier - let pitch parsing handle it
                        i = multiplier_end - 1  # Will be incremented by 1, then pitch parsing will handle the note
                    else:
                        # Other characters - let them be handled by subsequent iterations
                        i = multiplier_end - 1
                else:
                    # End of token after multiplier - we've parsed the entire anti-metric figure
                    i = multiplier_end - 1
                    
            elif c in DURATION_MAP:
                new_note.duration = DURATION_MAP[c]
            elif c in TRIPLET_DURATION_MAP:
                # Handle triplet durations without % (like '6e', '3r', '3.F')
                if i+1 < len(kern_token) and kern_token[i+1] in PITCH_TO_MIDI:
                    # Format like '6e' - triplet followed directly by pitch
                    new_note.duration = TRIPLET_DURATION_MAP[c]
                    new_note.is_measured_differently = True
                    # Don't increment i here - let pitch parsing handle the next character
                elif i+1 < len(kern_token) and kern_token[i+1] == 'r':
                    # Format like '3r' - triplet rest
                    new_note.duration = TRIPLET_DURATION_MAP[c]
                    new_note.is_measured_differently = True
                    new_note.note_type = 'rest'
                    i += 1  # Skip the 'r' character
                elif i+1 < len(kern_token) and kern_token[i+1] == '.':
                    # Detect a triplet without a % sign, like '3.F' - dotted triplet.
                    new_note.duration = TRIPLET_DURATION_MAP[c]
                    new_note.is_measured_differently = True
                    # Count and apply dots, then skip over them
                    dot_count = count_dots_at_position(kern_token, i+1)
                    new_note.duration *= 1.5**dot_count
                    new_note.is_dotted = True  # Set dotted property for triplets
                    i += dot_count
                else:
                    raise ValueError(f"Could not parse triplet duration '{c}' in token '{kern_token}'")
                    
            elif c in PITCH_TO_MIDI:
                ''' First, set the note without regarding accidentals. '''
                pitch_token, pitch_token_len = parse_pitch_token(kern_token, i)
                new_note.note_type = 'note'

                # Increment tuplet note count if we're in a tuplet
                if tuplet_open_flags[token_idx]:
                    tuplet_note_count[token_idx] += 1

                # Set MIDI pitch
                new_note.midi_pitch = PITCH_TO_MIDI[pitch_token]

                # Parse accidentals using helper function
                accidental_token, accidental_token_len = parse_accidental_token(kern_token, i + pitch_token_len)

                if accidental_token:
                    # Store which note the accidental applies to
                    accidental_trackers[token_idx][pitch_token[0].lower()] = accidentals[accidental_token]

                # Apply the accidental
                new_note.midi_pitch += accidental_trackers[token_idx][pitch_token[0].lower()]

                # Handle editorial accidentals
                editorial_len = 0
                if include_editorial_accidentals:
                    # Look for 'i' after the accidental, skipping over any ignored tokens
                    search_pos = i + pitch_token_len + accidental_token_len
                    while search_pos < len(kern_token):
                        if kern_token[search_pos] == 'i':
                            editorial_len = search_pos - (i + pitch_token_len + accidental_token_len) + 1
                            break
                        elif kern_token[search_pos] in ignored_tokens:
                            # Skip ignored tokens like 'X', 'y', etc.
                            search_pos += 1
                        else:
                            # Hit a non-ignored, non-'i' character - stop searching
                            break
                else:
                    raise NotImplementedError("Not using editorial accidentals is not yet implemented.")

                # Set the spelled note name
                new_note.octave = new_note.midi_pitch // 12
                base_letter = pitch_token[0].upper()
                new_note.spelled_name = f"{base_letter}{accidental_token}" + str(new_note.octave)

                # Skip over all the parsed components
                i += pitch_token_len + accidental_token_len + editorial_len - 1

                
            elif c in accidentals:
                raise NotImplementedError("Accidentals should be after notes only")
            elif c in ignored_tokens:
                pass # TODO: placeholder for characters yet to be implemented
            elif c == '.':
                # This character being present means: this could either be a period (keep the prevoius note), or a dotting of the note duration.
                if len(kern_token) == 1:
                    new_note.note_type = 'period' 
                    new_note.was_originally_period = True # This will be used to set the new occurrence flag to False in processing the slices
                else:
                    # Count and apply dots for regular notes
                    dot_count = count_dots_at_position(kern_token, i)

                    dot_count = 1

                    if new_note.duration == -1:
                        raise ValueError(f"Could not find duration for token '{kern_token}' and dot-count '{dot_count}'")
        
                    new_note.duration *= 1.5**dot_count
                    new_note.is_dotted = True  # Set dotted property
                    i += dot_count - 1
            
            elif c == 'V':
                # Tuplet start marker 
                new_note.is_tuplet_start = True
                tuplet_open_flags[token_idx] = True
                tuplet_note_count[token_idx] = 0  # Reset counter for new tuplet
            elif c == 'Z':
                # Tuplet end marker
                new_note.is_tuplet_end = True
                tuplet_open_flags[token_idx] = False
                tuplet_note_count[token_idx] = 0  # Reset counter
            elif c == 'r':
                new_note.note_type = 'rest'
            elif c == '=':
                # Check if it's a final barline (double equals)
                #print(f"Found barline in token '{kern_token}'")
                if kern_token.startswith('=='):
                    if DEBUG: print('Found final barline.')
                    new_note.note_type = 'final_barline'
                else:
                    new_note.note_type = 'barline'
                
                # Extract bar number if present
                digits_str = ''.join(d for d in kern_token[i:] if d.isdigit())
                if digits_str:
                    extracted_bar_number = int(digits_str)
                    # Bar number should be greater than the current bar number
                    if extracted_bar_number < current_bar:
                        raise ValueError(f"Bar number {extracted_bar_number} is lower than the previous bar number {current_bar}")
                    
                    # Update the current bar number
                    new_note.bar_number = extracted_bar_number
                    current_bar = extracted_bar_number                
                # Skip to the end of the token
                i += len(kern_token[i:]) - 1
            # --- Tie Flag Validation Logic ---
            elif c == '[':
                # Start of a tie
                if tie_open_flags[token_idx] == True:
                    raise ValueError(f"Found unexpected start of tie (char '{c}') in token '{kern_token}', at index '{token_idx}' in some line'")
                else:
                    tie_open_flags[token_idx] = True
                if DEBUG: print(f"START end of tie (char '{c}') in token '{kern_token}', at voice index '{token_idx}' in bar '{current_bar}'. Tie open flags: {tie_open_flags}.")
            elif c == ']':
                # End of a tie
                if tie_open_flags[token_idx] == False:
                    raise ValueError(f"Found unexpected end of tie (char '{c}') in token '{kern_token}' in bar {current_bar}.")
                else:
                    tie_open_flags[token_idx] = False
                if DEBUG: print(f"END of tie (char '{c}') in token '{kern_token}', at voice index '{token_idx}' in bar '{current_bar}'. Tie open flags: {tie_open_flags}.")
            elif c == '_':
                # Only relevant to longas and other special cases. Middle of a tie: do not change tie open flags. This character is effectively skipped most of the time, and not related to normal notes.
                if tie_open_flags[token_idx] == False:
                    raise ValueError(f"Found unexpected middle of tie (char '{c}') in token '{kern_token}' in bar {current_bar}.")
            # --- Longa, ignored tokens, errors ---
            elif c == 'l':
                # This note is to be rendered as a longa. Longas are represented as two brevis notes (over two bars) in the JRP.
                # Note: they could also be represented with duration '00' in the kern standard. The 'l' character is just to mark that it was originally a longa.
                new_note.is_longa = True
            else:
                raise ValueError(f"Could not parse character '{c}' in token '{kern_token}', at index '{i}, bar {current_bar}.'")
        except Exception as e:
            print(f"For token '{kern_token}', char '{c}', bar {current_bar},  encountered error:\n")
            raise e
        
        i += 1
    #endwhile

    # Error checking
    if new_note.duration is None:
        raise ValueError(f"Could not find duration for token '{kern_token}'")

    return new_note, current_bar, accidental_trackers, tie_open_flags, tuplet_open_flags, tuplet_note_count


"""Helper functions to parse all the different kinds of lines."""
# Helper functions
def strip_visual_markers(token):
    """Strip visual markers from barline tokens for proper comparison.
    
    Args:
        token (str): The barline token to clean
        
    Returns:
        str: The cleaned token with visual markers removed
    """
    # Remove visual markers: ., |, !, ", ', `, and trailing -
    cleaned = token.rstrip('-')  # Remove trailing minus (invisible barline)
    cleaned = cleaned.strip('.|!"\'`')  # Remove other visual markers
    return cleaned

def parse_timesig(line: str, metadata: dict, bar_count: int) -> dict:
    """ Parse a time signature token and update the metadata dict """
        
    time_sig_tokens = line.strip().split('\t')
    time_signatures = [] 
    
    # First pass: Extract all valid time signatures
    for token in time_sig_tokens: 
        time_sig = token
        if time_sig.startswith('*M'):
            time_sig = token[2:]  # Remove '*M' prefix to get e.g. "2/1"

            # Handle various formats of time signatures
            if '/' in time_sig:
                if '%' in time_sig:
                    # Assumed format like 'M*3/3%2'
                    try:
                        time_sig = time_sig.split('/')[1]
                        numerator, denominator = (int(part) for part in time_sig.split('%'))
                        time_signatures.append((numerator, denominator))
                    except Exception as e:
                        raise ValueError(f"Could not parse time-sig '{time_sig}' in token '{token}'. Original error: {str(e)}")
                else:
                    # Format like "2/1"
                    try:
                        numerator, denominator = (int(part) for part in time_sig.split('/'))
                        time_signatures.append((numerator, denominator))
                    except Exception as e:
                        raise ValueError(f"Could not parse time-sig '{time_sig}' in token '{token}'. Original error: {str(e)}")
            
            elif len(time_sig) == 1 and time_sig.isdigit():
                # Handle cases where only numerator is specified, like "3"
                time_signatures.append((int(time_sig), 1))
        elif time_sig.strip() == '*':
            # Placeholder for copying previous time signature
            time_signatures.append('*')
        else:
            raise ValueError(f"Could not parse token '{token}' in time signature '{time_sig}'.")
            

    # Replace '*' with the previous time signature
    for idx, time_sig in enumerate(time_signatures):
        if time_sig == '*':
            if len(metadata['time_signatures']) == 0:
                raise ValueError("Cannot copy previous time signature if there is no previous time signature.")
            else:
                time_signatures[idx] = metadata['time_signatures'][-1][1][idx]
            
    # Record the bar number of this change
    last_metadata_update_bar = bar_count
    barline_tuple = (last_metadata_update_bar, INFINITY_BAR)
    
    # If this is not the first time signature encountered, set the ending of the previous one
    if len(metadata['time_signatures']) >= 1:
        # If the new time sig is at bar 10, then the previous one will be set from (0, 9), not (0, 10). This should happen everywhere.
        metadata['time_signatures'][-1] = ((metadata['time_signatures'][-1][0][0], last_metadata_update_bar - 1), metadata['time_signatures'][-1][1])
 
    # Store time signatures with the bar. This should happen everywhere. number where they changed
    metadata['time_signatures'].append((barline_tuple, time_signatures))
    
    if DEBUG:
            # print(f"Parsed time signatures at bar {bar_count}: {time_signatures}")
            pass
    return metadata

def parse_keysig(line: str, metadata: dict, bar_count: int, line_idx: int) -> dict:
    """ Parse a key signature token and update the metadata dict """

    # Split the keysig tokens (one per voice)
    keysig_tokens = line.strip().split('\t')
    if DEBUG: print(f"Keysig tokens: {' '.join(keysig_tokens)} at line {line_idx}")
     
    if len(keysig_tokens) != int(metadata['voices']):
        raise ValueError(f"Line {line_idx} JRP-ID '{metadata['jrpid']}' has {len(keysig_tokens)} key signature tokens but metadata specifies {metadata['voices']} voices: '{line.strip()}'")    # Parse each voice's key signature
    key_signatures = []
    for token_idx, token in enumerate(keysig_tokens):
        if token == '*':
            # '*' token indicates unchanged key signature - use the previous key signature for this voice
            if metadata['key_signatures']:
                # Get the most recent key signature for this voice
                previous_key_signature = metadata['key_signatures'][-1][1][token_idx]
                key_signatures.append(previous_key_signature)
            else:
                # If no previous key signature, use empty key signature (no accidentals)
                key_signatures.append({k:v for k,v in key_signature_template.items()})
        elif token.startswith('*k['):
            # Parse the key signature
            keysig_token_content = token[3:]  # Remove *k[ from the start
            key_signature = {k:v for k,v in key_signature_template.items()}
            i = 0 
            while i < len(keysig_token_content):
                c = keysig_token_content[i]
                if c == ']':
                    break
                elif c in key_signature.keys():
                    # Check for accidental modifiers like '+' or '-'
                    if i + 1 < len(keysig_token_content):
                        next_char = keysig_token_content[i+1]
                        if next_char == '+':
                            key_signature[c] += 1
                            i += 1 # Move past the '+'
                        elif next_char == '-':
                            key_signature[c] -= 1
                            i += 1 # Move past the '-'
                    # If no modifier, it's just the note name, handled by the outer loop's increment
                elif c in ['+', '-']:
                    # This case should ideally be handled by the above, but as a safeguard:
                    raise ValueError(f"Could not parse key signature '{keysig_token_content}', reached accidental '{c}' unexpectedly at char index {i}.")
                else:
                    raise ValueError(f"Could not parse char '{c}' in key signature content '{keysig_token_content}' at char index {i}.")
                i += 1
            key_signatures.append(key_signature)
        else:
            raise ValueError(f"Invalid key signature token '{token}' in voice {token_idx + 1} at line {line_idx}. Expected '*' or token starting with '*k['.")

    # Record the bar number of this change
    last_metadata_update_bar = bar_count
    barline_tuple = (last_metadata_update_bar, INFINITY_BAR)
    
    # If this is not the first key signature encountered, set the ending of the previous one
    if metadata['key_signatures']:
        # If the new key sig is at bar 10, then the previous one will be set from (0, 9), not (0, 10). This should happen everywhere.
        metadata['key_signatures'][-1] = ((metadata['key_signatures'][-1][0][0], last_metadata_update_bar - 1), metadata['key_signatures'][-1][1])

    # Add keysig to metadata
    metadata['key_signatures'].append((barline_tuple, key_signatures))

    return metadata

def parse_pitch_token(kern_token: str, start_index: int) -> tuple[str, int]:
    """
    Parse a pitch token starting at the given index.
    Returns (pitch_token, token_length).
    """
    if start_index >= len(kern_token):
        raise ValueError(f"Start index {start_index} is beyond token length")
    
    c = kern_token[start_index]
    if c not in PITCH_TO_MIDI:
        raise ValueError(f"Character '{c}' is not a valid pitch")
    
    # Build the pitch token by repeating the same character
    pitch_token = c
    pitch_token_len = 1
    
    while (start_index + pitch_token_len < len(kern_token)) and kern_token[start_index + pitch_token_len] == c:
        pitch_token += c
        pitch_token_len += 1
    
    return pitch_token, pitch_token_len

def parse_accidental_token(kern_token: str, start_index: int) -> tuple[str, int]:
    """
    Parse an accidental token starting at the given index.
    Returns (accidental_token, token_length).
    """
    if start_index >= len(kern_token):
        return "", 0
    
    c = kern_token[start_index]
    
    # Check if this character should be ignored (not an accidental)
    if c in ignored_tokens:
        return "", 0
    
    if c.lower() not in accidentals:
        return "", 0
    
    # Build the accidental token by repeating the same character
    accidental_token = c
    accidental_token_len = 1
    
    while (start_index + accidental_token_len < len(kern_token)) and kern_token[start_index + accidental_token_len] == c:
        accidental_token += c
        accidental_token_len += 1
    
    return accidental_token, accidental_token_len

def count_dots_at_position(kern_token: str, start_index: int) -> int:
    """
    Count consecutive dots starting at the given index.
    Returns the number of dots found.
    """
    dot_count = 0
    while start_index + dot_count < len(kern_token) and kern_token[start_index + dot_count] == '.':
        dot_count += 1
    return dot_count


def validate_all_rules(salami_slices, metadata, cp_rules: CounterpointRulesBase,
                       only_validate_rules: list = None):
    violations = defaultdict(list[RuleViolation])

    for i, slice_cur in enumerate(salami_slices):
        current_kwargs = {
            #"slice1": slice_cur,
            #"slice2": salami_slices[i-1],
            "slice_index": i,
            "salami_slices": salami_slices,
            "metadata": metadata, # TODO: metadata can be an *arg, but doesn't really matter
            "only_validate_rules": only_validate_rules
        }
        # Update violations with the output
        slice_violations = cp_rules.validate_all_rules(**current_kwargs)
        for rule_name, rule_violations in slice_violations.items():
            violations[rule_name].extend(rule_violations)
    
    return dict(violations)


if __name__ == "__main__":
    pass

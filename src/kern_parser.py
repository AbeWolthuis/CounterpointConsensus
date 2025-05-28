import os
import importlib
from IPython.display import display
from pprint import pprint
from collections import defaultdict
from typing import List, Dict, Tuple
import copy
import pandas as pd
import cProfile, pstats


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
importlib.reload(Note)
importlib.reload(counterpoint_rules)
importlib.reload(process_salami_slices)

from Note import Note, SalamiSlice
from counterpoint_rules import CounterpointRules
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
    'L' : 'start-beam',
    'LL' : 'start-double-beam',
    'J' : 'end-beam',
    'JJ' : 'end-double-beam',
    'y' : 'uncertain_character'
}
ignored_line_tokens = {}

# Used to parse the key signature(s)
key_signature_template = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0}


def parse_kern(kern_filepath) -> Tuple[List, Dict]:
    metadata = {k:v for k,v in metadata_template.items()}
    metadata['src_path'] = kern_filepath
    
    with open(kern_filepath, 'r', encoding='utf-8') as f: 
        bar_count = 0 
    
        # Store tuples: (original_line_idx, line_string)
        note_section_with_indices: List[Tuple[int, str]] = []
        lines = f.readlines()

        for line_idx, line in enumerate(lines):
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
                elif line.startswith('*k'):
                    # Parse key signatures for all voices
                    metadata_dict = parse_keysig(line, metadata, bar_count, line_idx)
                    if DEBUG: print(f"Found key signatures at bar {bar_count}: {metadata_dict['key_signatures'][-1][1]}")
                        
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
                elif line.startswith('*-'):
                    # Ignored barlines
                    continue
                elif line.startswith('*>'):
                    # Lines indicating structure, or the begin of the structure.
                    continue
                else:
                    raise NotImplementedError(f"Some metadata starting with * not yet implemented. Line {line_idx}: {line}")
            # --- Barline Handling ---
            elif line.startswith('='):
                # Update bars according to the barnumber found
                try:
                    # Extract only digits from the line
                    digits_only = ''.join(c for c in line.split()[0] if c.isdigit())

                    if digits_only:
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
                note_section_with_indices.append((line_idx, line))
    
    ''' Post-process the meta-data. '''
    
    metadata['voices'] = int(metadata['voices'])

    # Handle composer attribution logic with priority
    if not metadata['COM'] and metadata['COA_entries']:
        # Find the highest priority COA entry
        # Priority: COA > COA2 > COA3 > etc.
        if 'COA' in metadata['COA_entries']:
            # COA without number has highest priority
            metadata['COM'] = metadata['COA_entries']['COA'] + " (COA)"
        else:
            # Find the lowest numbered COA entry
            coa_numbers = []
            for key in metadata['COA_entries'].keys():
                if key.startswith('COA') and len(key) > 3:  # COA2, COA3, etc.
                    try:
                        num = int(key[3:])  # Extract number after "COA"
                        coa_numbers.append(num)
                    except ValueError:
                        continue


    ''' Parse the notes into salami slices. '''
    salami_slices = parse_note_section(note_section_with_indices, metadata)
    print(" Duplicate voice names not handled. Linking of salami slices to previous next occurrence wrong, when the previous next occurrence is a tied note that is in the same bar as the start of that note. E.g. Jos1408, bar 28.")


    return salami_slices, metadata

def parse_note_section(note_section_with_indices: List[Tuple[int, str]], metadata: dict) -> List:
    """ Parse the notes into salami slices """
    salami_slices = []
    accidental_tracker = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0} 
    accidental_trackers = [dict(a=0, b=0, c=0, d=0, e=0, f=0, g=0) for _ in range(metadata['voices'])]
    current_bar = 1  # Start with bar 1
    
    # A per-voice tie tracker: one bool per voice
    current_tie_open_flags = [False] * int(metadata['voices'])


    for current_tuple_index, (original_line_idx, line) in enumerate(note_section_with_indices):
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

            # Create a new note object for each token
            new_note, accidental_trackers, current_tie_open_flags, current_bar = kern_token_to_note(
                token,accidental_trackers, current_tie_open_flags, current_bar, token_idx, include_editorial_accidentals=True
            )
            
            
            # --- Set Tie Flags using flags_before_token and current_tie_open_flags ---
            prev_note = None
            if len(salami_slices) > 0:
                # Get the most recent slice and check if it has a note for this voice
                prev_slice = salami_slices[-1]
                if prev_slice.notes[token_idx] is not None:
                    prev_note = prev_slice.notes[token_idx]
            
            is_tied_now = current_tie_open_flags[token_idx]
            was_tied_before = flags_before_token_processing[token_idx]
            a = 1

            # First, detect if this is a tie-start: a note is the start of a tie IF either of the following is true: 
            #   1) it is tied, and the previous note is not.
            #   2) it is tied, and the previous is a tie-end (in which case the previous note is allowed to be tied)

            # Logic for determining tie properties:
            # 1. Tie start: currently tied AND (wasn't tied before OR previous note had different pitch)
            # 2. Tie end: not tied now AND was tied before. This note is also set as being tied.
            # 3. Tie continuation: tied now AND was tied before AND same pitch as previous


            if is_tied_now and not was_tied_before: # and (prev_note.is_tied) and (not prev_note.is_tie_end) and (prev_note.midi_pitch == new_note.midi_pitch):
                # This is the start of a new tie
                new_note.is_tie_start = True
                new_note.is_tied = True
            elif is_tied_now and was_tied_before:
                # Check if this continues the same pitch or starts a new tie
                if (prev_note.is_tied) and (not prev_note.is_tie_end):
                    # This is a continuation of a tie with the same pitch
                    new_note.is_tied = True
                else:
                    # This is the start of a new tie (different pitch)
                    new_note.is_tie_start = True
                    new_note.is_tied = True
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
                remaining_lines_with_indices = note_section_with_indices[current_tuple_index + 1:]
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
        accidental_trackers: list[dict],
        tie_open_flags: list[bool],
        current_bar: int,
        token_idx: int,                         # Index of the current voice/token in the line; i.e. the voice (0-indexed)
        include_editorial_accidentals = True,
    ) -> Tuple[Note, dict, list, int]:
    
    new_note = Note()

    i = 0 
    while i < len(kern_token):
        try:
            c = kern_token[i]

            if c in DURATION_MAP:
                new_note.duration = DURATION_MAP[c]
            elif c in TRIPLET_DURATION_MAP:
                # Detect triplet format. E.g. '3%2.B-' or '6e'
                if i+2 < len(kern_token) and kern_token[i+1] == '%':
                    # splice 3%2
                    triplet_duration_token = kern_token[i:i+3].split('%') 
                    duration = TRIPLET_DURATION_MAP[triplet_duration_token[0]]
                    duration_modifier = triplet_duration_token[1] 
                    # Set the duration
                    new_note.duration = duration * int(duration_modifier)
                    
                    # Set note as triplet
                    new_note.is_triplet = True
                    i += 2
                elif i+1 < len(kern_token) and kern_token[i+1] in PITCH_TO_MIDI:
                    # splice 6e
                    new_note.duration = TRIPLET_DURATION_MAP[c]
                else:
                    raise ValueError(f"Could not parse triplet duration '{c}' in token '{kern_token}'")

            elif c in PITCH_TO_MIDI:
                ''' First, set the note without regarding accidentals. '''
                pitch_token = c
                new_note.note_type = 'note'

                # Check if this character repeats, and how many times (e.g. to get the pitch "CC" instead of just "C")
                # This loop should stop when the a character is not the same as the previous character, or at the line end.
                # We advance the looping variable i by pitch_token_len-1 (it gets increased by 1 always at the end of the loop)
                pitch_token_len = 1
                while (i+pitch_token_len < len(kern_token)) and kern_token[i+pitch_token_len] == c:
                    pitch_token += c
                    pitch_token_len += 1

                new_note.midi_pitch = PITCH_TO_MIDI[pitch_token]

                ''' Check for new accidentals after the pitch. '''
                accidental_token = ''
                accidental_token_len = 0

                if (i+pitch_token_len < len(kern_token)) and kern_token[i+pitch_token_len].lower() in accidentals:
                    # Check if the accidental is followed by another accidental
                    accidental_token = kern_token[i+pitch_token_len]
                    accidental_token_len = 1
                    # TODO: this indexing might be wrong; where is the note compared to the accidental?
                    while (i+pitch_token_len+accidental_token_len < len(kern_token)) and kern_token[i+pitch_token_len+accidental_token_len] == c:
                        accidental_token += c
                        accidental_token_len += 1

                    # Store which note the accidental applies to. E.g., the sharp on 'CC#' is stored under 'c':1 in the accidental_tracker.
                    accidental_trackers[token_idx][pitch_token[0].lower()] = accidentals[accidental_token]

                ''' Apply the accidental. If it is not set, it is 0 (by default). '''  
                new_note.midi_pitch += accidental_trackers[token_idx][pitch_token[0].lower()]

                ''' Handle editorial accidentals. '''
                # By default, we include editorial accidentals, and thus skip the editorial token "i".
                if include_editorial_accidentals:
                    if (i+pitch_token_len+accidental_token_len < len(kern_token)) and kern_token[i+pitch_token_len+accidental_token_len] == 'i':
                        i += 1
                else:
                    raise NotImplementedError("Not using editorial accidentals is not yet implemented.")

                # **Set the spelled note name**
                new_note.octave = new_note.midi_pitch // 12 # integer division to get the octave
                base_letter = pitch_token[0].upper()
                new_note.spelled_name = f"{base_letter}{accidental_token}" + str(new_note.octave)


                # If the pitch token len is only 1 (e.g. 1F), then we skip zero extra places over the increase i+=1 at the end of the loop.
                # If it is longer, then skip that amount, plus any accidentals (if present).
                i += pitch_token_len + accidental_token_len - 1

            elif c in accidentals:
                raise NotImplementedError("Accidentals should be after notes only")
            elif c in ignored_tokens:
                pass # TODO: placeholder for characters yet to be implemented
            elif c == '.':
                # Could either be a period (keep the prevoius note), or a dotting of the note duration.
                if len(kern_token) == 1:
                    new_note.note_type = 'period'
                else:
                    # Check how many dots there are, and increase the duration accordingly.
                    dot_count = 1
                    while i+dot_count < len(kern_token) and kern_token[i+dot_count] == '.':
                        dot_count += 1
                    # The duration should already be set. Thus, we can multiply it by 1.5^dot_count
                    if new_note.duration == -1:
                        raise ValueError(f"Could not find duration for token '{kern_token}' and dot-count '{dot_count}'")
                    new_note.duration *= 1.5**dot_count

                    i += dot_count - 1
            elif c == 'r':
                new_note.note_type = 'rest'
            elif c == '=':
                # Check if it's a final barline (double equals)
                #print(f"Found barline in token '{kern_token}'")
                if kern_token.startswith('=='):
                    print('Found final barline')
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
                    raise ValueError(f"Found unexpected end of tie (char '{c}') in token '{kern_token}' in bar {current_bar}.'")
                else:
                    tie_open_flags[token_idx] = False
                if DEBUG: print(f"END of tie (char '{c}') in token '{kern_token}', at voice index '{token_idx}' in bar '{current_bar}'. Tie open flags: {tie_open_flags}.")
            elif c == '_':
                # Only relevant to longas and other special cases. Middle of a tie: do not change tie open flags. This character is effectively skipped most of the time, and not related to normal notes.
                if tie_open_flags[token_idx] == False:
                    raise ValueError(f"Found unexpected middle of tie (char '{c}') in token '{kern_token}' in bar {current_bar}.'")
            # --- Longa, ignored tokens, errors ---
            elif c == 'l':
                # This note is to be rendered as a longa. Longas are represented as two brevis notes (over two bars) in the JRP.
                # Note: they could also be represented with duration '00' in the kern standard. The 'l' character is just to mark that it was originally a longa.
                new_note.is_longa = True
            else:
                raise ValueError(f"Could not parse character '{c}' in token '{kern_token}', at index '{i}, bar {current_bar}.'")
        except Exception as e:
            print(f"For token '{kern_token}', char '{c}', bar {current_bar} encountered error:\n\n")
            raise e
        
        i += 1
    #endwhile

    # Error checking
    if new_note.duration is None:
        raise ValueError(f"Could not find duration for token '{kern_token}'")

    # TODO: return the accidental tracker, because it is relevant for the entire measure
    return new_note, accidental_trackers, tie_open_flags, current_bar


"""Helper functions to parse all the different kinds of lines."""

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

    # Check that all keysig tokens are identical
    keysig_tokens = line.strip().split('\t')
    if DEBUG: print(f"Keysig tokens: {' '.join(keysig_tokens)} at line {line_idx}")

    if len(set(keysig_tokens)) != 1: 
        raise ValueError("Different key signatures in voices not yet implemented.")
    
    # Use the first token to parse the key sig (they are indentical)
    keysig_token = keysig_tokens[0][3:] # Remove *k[ from the start
    key_signature = {k:v for k,v in key_signature_template.items()}
    i = 0 
    while i < len(keysig_token):
        c = keysig_token[i]
        if c == ']':
            break
        elif c in key_signature.keys():
            if keysig_token[i+1] == '+':
                key_signature[c] += 1
            elif keysig_token[i+1] == '-':
                key_signature[c] -= 1
            i += 1
        elif c  in ['+', '-']:
            raise ValueError(f"Could not parse key signature '{keysig_token}', reached accidental '{c}' unpredictedly.")                     
        else:
            raise ValueError(f"Could not parse char '{c}' in key signature '{keysig_token}'")                     
        i += 1
        #endwhile

    # Record the bar number of this change
    last_metadata_update_bar = bar_count
    barline_tuple = (last_metadata_update_bar, INFINITY_BAR)
    
    # If this is not the first time signature encountered, set the ending of the previous one
    if metadata['key_signatures']:
        # If the new time sig is at bar 10, then the previous one will be set from (0, 9), not (0, 10). This should happen everywhere.
        metadata['key_signatures'][-1] = ((metadata['key_signatures'][-1][0][0], last_metadata_update_bar - 1), metadata['key_signatures'][-1][1])

    # Add keysig to metadata
    metadata['key_signatures'].append((barline_tuple, key_signature))

    return metadata


def validate_all_rules(salami_slices, metadata, cp_rules: CounterpointRules,
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
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))  # This is src/
    PROJECT_ROOT = os.path.dirname(ROOT_PATH)               # Go up one level to CounterpointConsensus/
    DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "full", "more_than_10", "SELECTED")

    valid_files = None
    invalid_files = ['Bus3064', 'Bus3078', 'Com1002a', 'Com1002b', 'Com1002c', 'Com1002d', 'Com1002e', 'Duf2027a', 'Duf3015', 'Duf3080', 'Gas0204c', 'Gas0503', 'Gas0504', 'Jos0302e', 'Jos0303c', 'Jos0303e', 'Jos0304c', 'Jos0402d', 'Jos0402e', 'Jos0602e', 'Jos0603d', 'Jos0901e', 'Jos0902a', 'Jos0902b', 'Jos0902c', 'Jos0902d', 'Jos0902e', 'Jos0904a', 'Jos0904b', 'Jos0904d', 'Jos0904e', 'Jos1302', 'Jos1501', 'Jos1610', 'Jos1706', 'Jos1802', 'Jos2015', 'Jos2102', 'Jos2602', 'Jos3004', 'Jos9901a', 'Jos9901e', 'Jos9905', 'Jos9906', 'Jos9907a', 'Jos9907b', 'Jos9908', 'Jos9909', 'Jos9910', 'Jos9911', 'Jos9912', 'Jos9914', 'Jos9923', 'Mar1003c', 'Mar3040', 'Oke1003a', 'Oke1003b', 'Oke1003c', 'Oke1003d', 'Oke1003e', 'Oke1005a', 'Oke1005b', 'Oke1005c', 'Oke1005d', 'Oke1005e', 'Oke1010a', 'Oke1010b', 'Oke1010c', 'Oke1010d', 'Oke1010e', 'Oke1011d', 
                     'Oke3025', 'Ort2005', 'Rue1007a', 'Rue1007b', 'Rue1007c', 'Rue1007d', 'Rue1007e', 'Rue1029a', 'Rue1029b', 'Rue1029c', 'Rue1029d', 'Rue1029e', 'Rue1035a', 'Rue1035b', 'Rue1035c', 'Rue1035d', 'Rue1035e', 'Rue2028', 'Rue2030', 'Rue2032', 'Rue3004', 'Rue3013', 'Tin3002']
    manual_invalid_files = ['Jos0603a',]
    invalid_files.extend(manual_invalid_files)

    # filepaths = [os.path.join("..", "data", "test", "Jos1408-Miserimini_mei.krn")]
    filepaths = [os.path.join("..", "data", "test", "Oke1014-Credo_Village.krn")]
    # filepaths = [os.path.join("..", "data", "test", "Rue1024a.krn")]
    # filepaths = [os.path.join("..", "data", "test", "extra_parFifth_rue1024a.krn")]
    filepaths = [os.path.join("..", "data", "test", "Jos1408-Miserimini_mei.krn"), os.path.join("..", "data", "test", "Rue1024a.krn"),os.path.join("..", "data", "test", "Oke1014-Credo_Village.krn")]

    filepaths = find_jrp_files(DATASET_PATH, valid_files, invalid_files, anonymous_mode='skip')


    # profiler = cProfile.Profile()
    # profiler.enable()

    all_violations: dict[str, list[RuleViolation]] = {}
    all_metadata: dict[str, dict] = {}
    full_violations_df = pd.DataFrame()

    for filepath in filepaths:
        try:
            salami_slices, metadata = parse_kern(filepath)
            salami_slices, metadata = post_process_salami_slices(salami_slices, metadata, expand_metadata_flag=True)

            # print('\nNotes with ties:\n')
            # for sslice in salami_slices:
            #     # if any(note.is_tied for note in sslice.notes):
            #     if sslice.bar in [51, 52]:
            #         print(sslice)
            #         pass
            #pprint(metadata)
            #print(salami_slices)

            cp_rules = CounterpointRules()
            only_validate_rules = [
                # Rhytm
                #'brevis_at_begin_end', 'longa_only_at_endings', 
                # Dots and ties
                    #'tie_into_strong_beat', 'tie_into_weak_beat',
                # Melody
                    #'leap_too_large',  'leap_approach_left_opposite', 'interval_order_motion', 'successive_leap_opposite_direction', 
                    # 'leap_up_accented_long_note',
                # Other aspects
                #   'eight_pair_stepwise',
                # Quarter note idioms
                'leap_in_quarters_balanced',
                
                # Chords
                    #'non_root_1st_inv_maj', 
                # Normalization functions
                'norm_ties_contained_in_bar, '#'norm_label_chord_name_m21', #'norm_count_tie_ends', 'norm_count_tie_starts',
            ]

            violations = validate_all_rules(salami_slices, metadata, cp_rules, only_validate_rules)
            curr_df = violations_to_df(violations, metadata)
            if full_violations_df.empty:
                full_violations_df = curr_df
            else:
                full_violations_df = pd.concat([full_violations_df, curr_df], ignore_index=True)

            print(f"\tJRP-ID: {metadata['jrpid']}")
            #pprint(violations)
            
            # Store the violations in a dict, with its JRP ID as the key.
            # In order to annotate the violations in each piece later, we save metadata with the violations for each piece.
            # In order to do so, remove the very large (and now unndeeded) key-sig info # TODO: also remove time-sigs
            curr_jrpid = metadata['jrpid']
            del metadata['key_signatures']

            all_metadata[curr_jrpid] = metadata
            all_violations[curr_jrpid] = violations
        except Exception as e:
            print('\n')
            print(f"Error processing file {filepath}. \n\n")
            raise e
                  
            

    # Sort the final DF
    full_violations_df = sort_df_by_rule_id(full_violations_df)

    annotate_violations_flag = True
    if annotate_violations_flag:
        destination_dir = os.path.join("..", "data", "annotated")
        annotate_all_kern(filepaths, destination_dir, all_metadata, all_violations, 
                          use_rule_ids=True, overwrite=True, maintain_jrp_structure=True, verbose=True)

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    
    print()
    pd.set_option('display.max_columns', None)
    display(full_violations_df.head()); print()



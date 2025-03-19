import os
import importlib
from IPython.display import display
from pprint import pprint
from collections import defaultdict
from typing import List, Dict, Tuple
import copy
import pandas as pd

import cProfile, pstats

from constants import DURATION_MAP, TRIPLET_DURATION_MAP, TIME_SIGNATURE_NORMALIZATION_MAP ,PITCH_TO_MIDI, MIDI_TO_PITCH 
from constants import FLOAT_TRUNCATION_DIGITS, BEAT_GRID_DIVISIONS
from counterpoint_rules import RuleViolation
from data_preparation import violations_to_df


# Our own modules
import Note
import counterpoint_rules
importlib.reload(Note)
importlib.reload(counterpoint_rules)

from Note import Note, SalamiSlice
from counterpoint_rules import CounterpointRules

    
DEBUG = True
DEBUG2 = False
INFINITY_BAR = 1e6



metadata_template = {
    # JRP sepcified metadata
    'COM': '', # composer 
    'CDT': '', # Copmoser's dates record
    'jrpid': '', # josquin research project ID
    'attribution-level@Jos': '',
    'SEGMENT': '', # filename

    'voices': [], # list with number of voices for each score-movement  # TODO: support voices changing during piece
    'voice_names': [], # list with voice names for each score-movement  # TODO: support voices changing during piece
    'key_signatures': [], # list of key signatures for each score-movement
    
    # Add time signatures to metadata template
    'time_signatures': [], # list of time signatures for each voice at each change point
    
    # Metadata we add ourselves
    'section_ends': [], # list of the barnumbers of the last bars of each section 
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
    '[' : 'start_slur', # TODO: implement
    ']' : 'end_slur' # TODO: implement
}
ignored_line_tokens = {}

# Used to parse the key signature(s)
key_signature_template = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0}



def parse_kern(kern_filepath) -> Tuple[List, Dict]:
    metadata = {k:v for k,v in metadata_template.items()}
    
    with open(kern_filepath, 'r') as f:
        metadata_flag = True
        bar_count = 0 
        last_metadata_update_bar = 0

        note_section = []
        lines = f.readlines()

        # Parse metadata, and gather the section of the file containing notes
        for line_idx, line in enumerate(lines):
            # if DEBUG: print(line[0:6]) 
            if line.startswith('!'):
                if line[0:6] in metadata_linestarts_map.keys():
                    # Splt '!!!jrpid: xxx' to 'jrpid',' xxx'
                    a = line.split(':')[0][0:6]
                    metadata_key, metadata_value = metadata_linestarts_map[a], line.split(':')[1].strip()
                    metadata[metadata_key] = metadata_value

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
                        voice_names = [voice_name.replace('*I"', '') for voice_name in line.split()]
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
                    
                    # Add barline to note section regardless of whether it has a number
                    # This ensures barlines are properly processed in parse_note_section.
                    note_section.append(line)
                    
                except Exception as e:
                    raise ValueError(f"Could not parse bar number from line: '{line.strip()}'. Error: {str(e)}")
            elif False:
                # try-except-else is such weird syntax
                continue
            else: 
                note_section.append(line)

    # Post-process the meta-data
    metadata['voices'] = int(metadata['voices'])

    # Parse the notes
    salami_slices = parse_note_section(note_section, metadata)
    if DEBUG: print("Ties not implemented yet. Sorting of time-sigs not handled.")


    return salami_slices, metadata


def parse_note_section(note_section: List[str], metadata) -> List:
    """ Parse the notes into salami slices """
    salami_slices = []
    accidental_tracker = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0} 
    current_bar = 1  # Start with bar 1
    
    # A per-voice tie tracker: one bool per voice
    tie_open_flags = [False] * int(metadata['voices'])


    for line_idx, line in enumerate(note_section):
        new_salami_slice = SalamiSlice(
            num_voices=int(metadata['voices']),
            bar = current_bar
        )        

        tokens = line.split()
        # Check if we're trying to add too many/few notes
        if len(tokens) != metadata['voices'] and not line.startswith('='):
            raise ValueError(f"Line has {len(tokens)} tokens but metadata specifies {metadata['voices']} voices: '{line.strip()}'")

        for token_idx, token in enumerate(tokens):
            new_note, accidental_tracker, tie_open_flags, current_bar = kern_token_to_note(
                token, accidental_tracker, tie_open_flags, current_bar=current_bar, token_idx=token_idx
                )
            # Rounding notes is done in calculate_beat_positions() (TODO: should we also snap durations?).
            new_salami_slice.add_note(note=new_note, voice=token_idx)
            

            if new_note.note_type == 'final_barline':
                # Check if there are remaining lines with notes after this one.
                remaining_lines = note_section[line_idx+1:]
                for remaining_idx, remaining_line in enumerate(remaining_lines):
                    if (not remaining_line.strip() or 
                        remaining_line.startswith('!') or 
                        remaining_line.startswith('*')):
                        raise ValueError(f"Found unexpected line '{remaining_line}' after final barline.")
            elif new_note is None:
                raise ValueError(f"Token '{token}' in line '{line}' lead to new_note being None.")

        salami_slices.append(new_salami_slice)              

    return salami_slices

def kern_token_to_note(
        kern_token: str,
        accidental_tracker: Dict,
        tie_open_flags: List[bool],
        token_idx: int,
        current_bar = -1,
        include_editorial_accidentals = True,
    ) -> Tuple[Note, Dict, int]:
    
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
                accidental_token = 0
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
                    accidental_tracker[pitch_token[0].lower()] = accidentals[accidental_token]

                ''' Apply the accidental. If it is not set, it is 0 (by default). '''  
                new_note.midi_pitch += accidental_tracker[pitch_token[0].lower()]

                ''' Handle editorial accidentals. '''
                # By default, we include editorial accidentals, and thus skip the editorial token "i".
                if include_editorial_accidentals:
                    if (i+pitch_token_len+accidental_token_len < len(kern_token)) and kern_token[i+pitch_token_len+accidental_token_len] == 'i':
                        i += 1
                else:
                    raise NotImplementedError("Not using editorial accidentals is not yet implemented.")

            
                i += pitch_token_len + accidental_token_len

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
                if i+1 < len(kern_token) and kern_token[i+1] == '=':
                    new_note.note_type = 'final_barline'
                else:
                    new_note.note_type = 'barline'
                
                # Extract bar number if present
                digits_str = ''.join(d for d in kern_token[i:] if d.isdigit())
                if digits_str:
                    extracted_bar_number = int(digits_str)
                    new_note.bar_number = extracted_bar_number
                    # Update the current bar number, to be returned
                    current_bar = extracted_bar_number

                
                # Skip to the end of the token
                i += len(kern_token[i:]) - 1
            elif c == '[':
                # Start of a tie
                if tie_open_flags[token_idx] == True:
                    raise ValueError(f"Found unexpected start of tie (char '{c}') in token '{kern_token}', at index '{token_idx}' in some line'")
                else:
                    tie_open_flags[token_idx] = True
            elif c == ']':
                # End of a tie
                if tie_open_flags[token_idx] == False:
                    raise ValueError(f"Found unexpected end of tie (char '{c}') in token '{kern_token}', at index '{token_idx}' in some line'")
                else:
                    tie_open_flags[token_idx] = False
            else:
                raise ValueError(f"Could not parse character '{c}' in token '{kern_token}', at index '{i}'")
        except Exception as e:
            print(f"For token '{kern_token}', char '{c}', index '{i}', encountered error:\n\n")
            raise e
        
        i += 1
    #endwhile

    # If tie_open_flags[voice_idx] is True, we mark this note as tied
    new_note.is_tied = tie_open_flags[token_idx]

    # Error checking
    if new_note.duration is None:
        raise ValueError(f"Could not find duration for token '{kern_token}'")

    # TODO: return the accidental tracker, because it is relevant for the entire measure
    return new_note, accidental_tracker, tie_open_flags, current_bar


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
            print(f"Parsed time signatures at bar {bar_count}: {time_signatures}")
    
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
    barline_tuple = (last_metadata_update_bar, -1)
    
    # If this is not the first time signature encountered, set the ending of the previous one
    if metadata['key_signatures']:
        # If the new time sig is at bar 10, then the previous one will be set from (0, 9), not (0, 10). This should happen everywhere.
        metadata['key_signatures'][-1] = ((metadata['key_signatures'][-1][0][0], last_metadata_update_bar - 1), metadata['key_signatures'][-1][1])

    # Add keysig to metadata
    metadata['key_signatures'].append((barline_tuple, key_signature))

    return metadata

"""Functions for post-processing the salami slices such that they can be analysed."""

def post_process_salami_slices(salami_slices: List[SalamiSlice], metadata) -> Tuple[List[SalamiSlice], Dict[str, any]]:
    """ Post-process the salami slices.  """
    # Note, most of this could be done in one pass, but for developmental easy we do it in multiple passes.
    salami_slices = set_period_notes(salami_slices)
    salami_slices, metadata = order_voices(salami_slices, metadata)
    salami_slices = remove_barline_slice(salami_slices, metadata)
    salami_slices = set_interval_property(salami_slices)
    salami_slices = calculate_offsets(salami_slices)  # Calculate raw offsets from bar start
    salami_slices = calculate_beat_positions(salami_slices, metadata)  # Convert offsets to beats
    
    return salami_slices, metadata

def remove_barline_slice(salami_slices: List[SalamiSlice], metadata: Dict[str, any]) -> Tuple[List[SalamiSlice], Dict[str, any]]:
    """ Remove all barline slices. """
    salami_slices = [salami_slice for salami_slice in salami_slices if salami_slice.notes[0].note_type != 'barline']
    return salami_slices

def set_interval_property(salami_slices: List[SalamiSlice]) -> List[SalamiSlice]:
    """ Set the intervals in the salami slices """
    for i, salami_slice in enumerate(salami_slices):
        salami_slice.absolute_intervals = salami_slice._calculate_intervals()
        salami_slice.reduced_intervals = salami_slice._calculate_reduced_intervals()
    return salami_slices

def set_period_notes(salami_slices: List[SalamiSlice]) -> List[SalamiSlice]:
    """ Set the notes in the salami slices that are periods to the notes in the previous slice """
    for i, salami_slice in enumerate(salami_slices):
        for voice, note in enumerate(salami_slice.notes):
            if note.note_type == 'period':
                if i == 0:
                    raise ValueError("Period note in first slice")
                # Copy note, set 
                salami_slice.notes[voice] = salami_slices[i-1].notes[voice]
                salami_slices[i-1].notes[voice].new_occurrence = False
    return salami_slices

def order_voices(salami_slices: List[SalamiSlice], metadata: Dict[str, any]) -> Tuple[List[SalamiSlice], Dict[str, any]]:
    """ Order the voices from low to high in the salami slices."""
    # Find the highest note in each voice
    highest_notes = [0] * metadata['voices']
    for salami_slice in salami_slices:
        for voice, note in enumerate(salami_slice.notes):
            if note.note_type == 'note':
                highest_notes[voice] = max(highest_notes[voice], note.midi_pitch)

    # Reorder the voices in the salami slices, and metadata. Highest voice at index 0.
    voice_order = sorted(range(metadata['voices']), key=lambda x: highest_notes[x], reverse=True)
    for salami_slice in salami_slices:
        # python magic ensures the notes are not overwritten during the for loop
        salami_slice.notes = [salami_slice.notes[voice] for voice in voice_order]

    # Sort all other metadata that is given per voice, according to the sorting of voice_order
    metadata['voice_names'] = [metadata['voice_names'][_voice].lower() for _voice in voice_order]

    # Collect a new list of voice-order sorted time signatures
    reordered_time_signatures = []
    for barline_tuple, timesigs in metadata['time_signatures']:
        reordered_time_signatures.append(
            (barline_tuple, [timesigs[v] for v in voice_order])
        )
    metadata['time_signatures'] = reordered_time_signatures    

    return salami_slices, metadata

def calculate_offsets(salami_slices: List[SalamiSlice]) -> List[SalamiSlice]:
    """ Calculate the offset of each slice from the beginning of its bar. """
    current_bar = 1
    current_offset = 0.0

    # Keep track of leftover durations for each voice. We'll reset them whenever we move to a new slice.
    leftover_durations = [0.0] * (salami_slices[0].num_voices)
    # Set durations of first slice
    for voice_idx, note in enumerate(salami_slices[0].notes):
        if note.note_type in ('note', 'rest'):
            leftover_durations[voice_idx] = note.duration

    for slice_idx, cur_slice in enumerate(salami_slices):
        # If we moved to a new bar, reset offset
        if cur_slice.bar != current_bar:
            if DEBUG2: print('New bar:',leftover_durations)


            # Sanity check: all voices must have leftover durations <= 0 + EPSILON
            epsilon = 1e-5
            if any(duration > epsilon for duration in leftover_durations):
                # print('\n', salami_slices[0:10], '\n')
                #raise ValueError(f"Bar {cur_slice.bar} has leftover durations: {leftover_durations}")
                pass
            current_bar = cur_slice.bar
            current_offset = 0.0
            # The new leftover durations are the durations of the first slice of the new bar
            leftover_durations = [note.duration for voice_idx, note in enumerate(cur_slice.notes)]
        
        # This slice starts at the current_offset
        cur_slice.offset = current_offset

        # If any voice reached the end of its duration, then that means this slice has a new note for it
        for voice_idx, note in enumerate(cur_slice.notes):
            if leftover_durations[voice_idx] <= 0:
                leftover_durations[voice_idx] = note.duration

        # Get the minimum duration (for which the note is not a period note).
        # After that amount of time, the next slice will take place.
        time_step = min(leftover_durations)
        if DEBUG2:
            # Find all indices with the minimum duration (to detect ties)
            min_indices = [i for i, d in enumerate(leftover_durations) if d == time_step]
            is_tie = len(min_indices) > 1
            
            # Always use the first index as required
            min_duration_voice = min_indices[0]
            min_duration_note = cur_slice.notes[min_duration_voice]
            
            # Add tie information to debug output
            tie_message = f"tie between voices {', '.join(map(str, min_indices))}, using voice {min_duration_voice}" if is_tie else f"in voice {min_duration_voice}"
            
            slice_notes = ', '.join([note.compact_summary for note in cur_slice.notes if note.note_type == 'note'])
            print(f"Slice {slice_idx} ({slice_notes}) at bar {cur_slice.bar} has offset {cur_slice.offset} and time step {time_step} {tie_message} based on {min_duration_note.compact_summary}")
            print('\t',leftover_durations)
            triplet_notes = [note.is_triplet for note in cur_slice.notes]
            if any(triplet_notes):
                print(f"Triplet notes: {triplet_notes}")

        # Go forwards in time by the minimum duration of the notes in this slice
        current_offset += time_step 
        leftover_durations = [duration - time_step for duration in leftover_durations if duration]

        a = 1
    return salami_slices

def _DEPRECATED_calculate_offsets(salami_slices: List[SalamiSlice]) -> List[SalamiSlice]:
    """ Calculate the offset of each slice from the beginning of its bar """
    current_offset = 0.0
    current_bar = 1  # Start with bar 1
    
    for i, cur_slice in enumerate(salami_slices):
        # If this is a new bar, reset the offset
        if cur_slice.bar != current_bar:
            current_offset = 0
            current_bar = cur_slice.bar
        
        # Set the offset for this slice
        cur_slice.offset = current_offset
        # Calculate the next offset based on the shortest duration in this slice
        min_duration = float('inf')
        has_valid_note = False
        
        for note in cur_slice.notes:
            if note and note.note_type in ('note', 'rest'):
                min_duration = min(min_duration, note.duration)
                has_valid_note = True
        
        # Only update the current offset if we found valid notes.
        # Doesn't update if the slice is a barline or final barline.
        if has_valid_note:
            current_offset += min_duration
    
    return salami_slices

def calculate_beat_positions(salami_slices: List[SalamiSlice], metadata) -> List[SalamiSlice]:
    """ 
    Calculate the beat position for each slice based on its offset and the time signature. 
    This assigns a beat property to each salami slice with its position in musical beats.
    """
    # For each salami slice, find the applicable time signature and convert offset to beats
    timesig_index = 0
    current_time_sig_tuple = metadata['time_signatures'][timesig_index]

    # We'll move through slices bar by bar
    for cur_slice in salami_slices:

        # If we've passed the old time signature's bar range, move on to the next time sig
        # TODO: time sigs can be different; even though the slice will always fall on the same beat
        if cur_slice.bar > current_time_sig_tuple[0][1]:
            timesig_index += 1
            current_time_sig_tuple = metadata['time_signatures'][timesig_index]

        # Take the first time signature arbitrarily 
        numerator, denominator = current_time_sig_tuple[1][0]

        # 1) Figure out measure length in "whole-note" time
        #    E.g. for 4/4, denominator=4 => measure_length=1.0 from DURATION_MAP
        if str(denominator) not in DURATION_MAP:
            raise ValueError(f"Unsupported denominator {denominator}")
        measure_length = DURATION_MAP[str(denominator)] * numerator

         # 2) Get how many subdivisions we want for the entire measure
        subdivisions = get_subdivisions_for_timesig(numerator, denominator, division_per_beat = BEAT_GRID_DIVISIONS)
    
        # 3) Snap the offset
        snapped_offset = snap_offset_to_grid(cur_slice.offset, measure_length, subdivisions)
        cur_slice.offset = snapped_offset

        # 4) Also snap the note durations in the slices. Note, 
        for note in cur_slice.notes:
            if note.duration:
                note.duration = snap_offset_to_grid(note.duration, measure_length, subdivisions)

        if not current_time_sig_tuple or not numerator or not denominator:
            raise ValueError(f"No time signature found for bar {cur_slice.bar}")

        # Beat calculation magic
        beat = 1 + cur_slice.offset / DURATION_MAP[str(denominator)] 
        # Round to get rid of floating point division error. NOTE: this rounding might cause unpredictable bugs?
        cur_slice.beat = cur_slice.truncate_float_as_float(beat, FLOAT_TRUNCATION_DIGITS)

            
    return salami_slices

'''Other helper functions'''

def snap_offset_to_grid(offset: float, measure_length: float, subdivisions: int) -> float:
    """
    Snap 'offset' to the nearest of 'subdivisions' equally spaced points from 0 to measure_length.
    """
    step = measure_length / subdivisions
    index = round(offset / step)
    return index * step

def get_subdivisions_for_timesig(numerator: int, denominator: int, division_per_beat = BEAT_GRID_DIVISIONS) -> int:
    # TODO: customize for certain time signatures
    return numerator * division_per_beat

def validate_all_rules(salami_slices, metadata, cp_rules: CounterpointRules):
    violations = defaultdict(list)

    for i, slice_cur in enumerate(salami_slices):
        current_kwargs = {
            "slice1": slice_cur,
            "slice2": salami_slices[i-1],
            "slice_index": i,
            "metadata": metadata # TODO: metadata can be an *arg, but doesn't really matter
        }
        # Update violations with the output
        slice_violations = cp_rules.validate_all_rules(**current_kwargs)
        for rule_name, rule_violations in slice_violations.items():
            violations[rule_name].extend(rule_violations)
    
    return dict(violations)



if __name__ == "__main__":
    # filepath = os.path.join("..", "data", "test", "Jos1408-Miserimini_mei.krn")
    # filepath = os.path.join("..", "data", "test", "Jos1408-test.krn")
    filepath = os.path.join("..", "data", "test", "Rue1024a.krn")
    # filepath = os.path.join("..", "data", "test", "extra_parFifth_rue1024a.krn")

    # profiler = cProfile.Profile()
    # profiler.enable()
    
    salami_slices, metadata = parse_kern(filepath)
    salami_slices, metadata = post_process_salami_slices(salami_slices, metadata)
    print(salami_slices[180:])
    # print(metadata)

    cp_rules = CounterpointRules()
    violations = validate_all_rules(salami_slices, metadata, cp_rules)

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    
    print()
    pprint(violations)
    df = violations_to_df(violations, metadata)

    print()
    #display(df.head()); print()



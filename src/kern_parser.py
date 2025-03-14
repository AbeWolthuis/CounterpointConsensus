import os
import importlib
from IPython.display import display
from pprint import pprint
from collections import defaultdict
from typing import List, Dict, Tuple
import copy
import pandas as pd

from constants import DURATION_MAP, TRIPLET_DURATION_MAP, TIME_SIGNATURE_NORMALIZATION_MAP ,PITCH_TO_MIDI, MIDI_TO_PITCH 
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

INFINITY_BAR = 1e6

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
    
    for line_idx, line in enumerate(note_section):
        new_salami_slice = SalamiSlice(num_voices=int(metadata['voices']))
        barline_updated_flag = False
        
        tokens = line.split()
        # Check if we're trying to add too many/few notes
        if len(tokens) != metadata['voices'] and not line.startswith('='):
            raise ValueError(f"Line has {len(tokens)} tokens but metadata specifies {metadata['voices']} voices: '{line.strip()}'")
        
        for token_idx, token in enumerate(tokens):
            new_note, accidental_tracker = kern_token_to_note(kern_token=token, accidental_tracker=accidental_tracker, line_idx=line_idx)
            new_salami_slice.add_note(note=new_note, voice=token_idx)
            
            # Set the current bar based on extracted bar number or increment
            if new_note.note_type == 'barline' and not barline_updated_flag:
                barline_updated_flag = True
                # If the note has an extracted bar number, use it
                if hasattr(new_note, 'bar_number') and new_note.bar_number is not None:
                    current_bar = new_note.bar_number
                else:
                    # Otherwise increment the counter
                    current_bar += 1
                # Reset the accidental tracker for the next measure
                accidental_tracker = {key:0 for key in accidental_tracker}
                
            new_salami_slice.bar = current_bar

            if new_note.note_type == 'final_barline':
                # TODO we can stop parsing here, as the rest of the metadata should be handled further in the 
                # Check if there are remaining lines with notes after this one.
                # This should not happen, but if it does, we should raise an error.
                remaining_lines = note_section[line_idx+1:]
                for remaining_idx, remaining_line in enumerate(remaining_lines):
                    # Skip empty lines, comments, or metadata
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
        accidental_tracker: dict,
        include_editorial_accidentals = True,
        line_idx = -1
    ) -> Note:
    
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
                    new_note.duration *= 1.5**dot_countf present

                    i += dot_count - 11 < len(kern_token):
            elif c == 'r':line
                new_note.note_type = 'rest'
            elif c == '=': 'final_barline'
                if kern_token[i+1] == '=':   else:
                    new_note.note_type = 'final_barline'
                else:to extract bar number
                    new_note.note_type = 'barline'
                # Skips until the after the barnumber. This should also always be the end of the line, since the barline token looks like '=123'.     if digits_str:
                i += len(kern_token[i:]) - 1                    bar_number = int(digits_str)
            else:              new_note.bar_number = bar_number  # Store bar number on the note
                raise ValueError(f"Could not parse character '{c}' in token '{kern_token}', at index '{i}'")   
        except Exception as e:                # Skip to the end of the token
            print(f"For token '{kern_token}', char '{c}', index '{i}', encountered error:\n\n") len(kern_token[i:]) - 1
            raise e
        {kern_token}', at index '{i}'")
        i += 1        except Exception as e:
    #endwhile:\n\n")

    # Error checking        
    if new_note.duration is None:        i += 1
        raise ValueError(f"Could not find duration for token '{kern_token}'")

    # TODO: return the accidental tracker, because it is relevant for the entire measure
    return new_note, accidental_tracker
raise ValueError(f"Could not find duration for token '{kern_token}'")

"""Helper functions to parse all the different kinds of lines."""cidental tracker, because it is relevant for the entire measure
return new_note, accidental_tracker
def parse_timesig(line: str, metadata: dict, bar_count: int) -> dict:
    """ Parse a time signature token and update the metadata dict """
        arse all the different kinds of lines."""
    time_sig_tokens = line.strip().split('\t')
    time_signatures = [] 
        """ Parse a time signature token and update the metadata dict """
    # First pass: Extract all valid time signatures
    for token in time_sig_tokens: p().split('\t')
        time_sig = token
        if time_sig.startswith('*M'):
            time_sig = token[2:]  # Remove '*M' prefix to get e.g. "2/1"t all valid time signatures

            # Handle various formats of time signatures
            if '/' in time_sig:
                if '%' in time_sig: '*M' prefix to get e.g. "2/1"
                    # Assumed format like 'M*3/3%2'
                    try:various formats of time signatures
                        time_sig = time_sig.split('/')[1]
                        numerator, denominator = (int(part) for part in time_sig.split('%'))n time_sig:
                        time_signatures.append((numerator, denominator))
                    except Exception as e:
                        raise ValueError(f"Could not parse time-sig '{time_sig}' in token '{token}'. Original error: {str(e)}")g.split('/')[1]
                else:
                    # Format like "2/1"            time_signatures.append((numerator, denominator))
                    try:
                        numerator, denominator = (int(part) for part in time_sig.split('/'))_sig}' in token '{token}'. Original error: {str(e)}")
                        time_signatures.append((numerator, denominator))
                    except Exception as e:1"
                        raise ValueError(f"Could not parse time-sig '{time_sig}' in token '{token}'. Original error: {str(e)}")
            minator = (int(part) for part in time_sig.split('/'))
            elif len(time_sig) == 1 and time_sig.isdigit():           time_signatures.append((numerator, denominator))
                # Handle cases where only numerator is specified, like "3"
                time_signatures.append((int(time_sig), 1))            raise ValueError(f"Could not parse time-sig '{time_sig}' in token '{token}'. Original error: {str(e)}")
        elif time_sig.strip() == '*':            
            # Placeholder for copying previous time signaturesdigit():
            time_signatures.append('*')is specified, like "3"
        else:ures.append((int(time_sig), 1))
            raise ValueError(f"Could not parse token '{token}' in time signature '{time_sig}'.")
            
signatures.append('*')
    # Replace '*' with the previous time signature
    for idx, time_sig in enumerate(time_signatures):raise ValueError(f"Could not parse token '{token}' in time signature '{time_sig}'.")
        if time_sig == '*':
            if len(metadata['time_signatures']) == 0:
                raise ValueError("Cannot copy previous time signature if there is no previous time signature.")
            else:for idx, time_sig in enumerate(time_signatures):
                time_signatures[idx] = metadata['time_signatures'][-1][1][idx]
            ]) == 0:
    # Record the bar number of this change
    last_metadata_update_bar = bar_count
    barline_tuple = (last_metadata_update_bar, INFINITY_BAR)               time_signatures[idx] = metadata['time_signatures'][-1][1][idx]
    
    # If this is not the first time signature encountered, set the ending of the previous one
    if len(metadata['time_signatures']) >= 1:last_metadata_update_bar = bar_count
        # If the new time sig is at bar 10, then the previous one will be set from (0, 9), not (0, 10). This should happen everywhere.uple = (last_metadata_update_bar, INFINITY_BAR)
        metadata['time_signatures'][-1] = ((metadata['time_signatures'][-1][0][0], last_metadata_update_bar - 1), metadata['time_signatures'][-1][1])
 # If this is not the first time signature encountered, set the ending of the previous one
    # Store time signatures with the bar. This should happen everywhere. number where they changed['time_signatures']) >= 1:
    metadata['time_signatures'].append((barline_tuple, time_signatures))        # If the new time sig is at bar 10, then the previous one will be set from (0, 9), not (0, 10). This should happen everywhere.
    last_metadata_update_bar - 1), metadata['time_signatures'][-1][1])
    if DEBUG:
            print(f"Parsed time signatures at bar {bar_count}: {time_signatures}")    # Store time signatures with the bar. This should happen everywhere. number where they changed
    es))
    return metadata

def parse_keysig(line: str, metadata: dict, bar_count: int, line_idx: int) -> dict:            print(f"Parsed time signatures at bar {bar_count}: {time_signatures}")
                        """ Parse a key signature token and update the metadata dict """

                        # Check that all keysig tokens are identical
                        keysig_tokens = line.strip().split('\t')
                        if DEBUG: print(f"Keysig tokens: {' '.join(keysig_tokens)} at line {line_idx}")"

                        if len(set(keysig_tokens)) != 1: k that all keysig tokens are identical
                            raise ValueError("Different key signatures in voices not yet implemented.").split('\t')
                        g tokens: {' '.join(keysig_tokens)} at line {line_idx}")
                        # Use the first token to parse the key sig (they are indentical)
                        keysig_token = keysig_tokens[0][3:] # Remove *k[ from the startysig_tokens)) != 1: 
                        key_signature = {k:v for k,v in key_signature_template.items()} signatures in voices not yet implemented.")
                        i = 0 
                        while i < len(keysig_token):e key sig (they are indentical)
                            c = keysig_token[i]Remove *k[ from the start
                            if c == ']':ey_signature_template.items()}
                                break
                            elif c in key_signature.keys():):
                                if keysig_token[i+1] == '+':
                                    key_signature[c] += 1== ']':
                                elif keysig_token[i+1] == '-':
                                    key_signature[c] -= 1 in key_signature.keys():
                                i += 1ysig_token[i+1] == '+':
                            elif c  in ['+', '-']:                                    key_signature[c] += 1
                                raise ValueError(f"Could not parse key signature '{keysig_token}', reached accidental '{c}' unpredictedly.")                     
                            else:
                                raise ValueError(f"Could not parse char '{c}' in key signature '{keysig_token}'")                     
                            i += 1    elif c  in ['+', '-']:
                            #endwhilental '{c}' unpredictedly.")                     

                        # Record the bar number of this change
                        last_metadata_update_bar = bar_count
                        barline_tuple = (last_metadata_update_bar, -1)                            #endwhile
                        
                        # If this is not the first time signature encountered, set the ending of the previous one
                        if metadata['key_signatures']:                        last_metadata_update_bar = bar_count
                            # If the new time sig is at bar 10, then the previous one will be set from (0, 9), not (0, 10). This should happen everywhere. (last_metadata_update_bar, -1)
                            metadata['key_signatures'][-1] = ((metadata['key_signatures'][-1][0][0], last_metadata_update_bar - 1), metadata['key_signatures'][-1][1])                        
e ending of the previous one
                        # Add keysig to metadata                        if metadata['key_signatures']:
                        metadata['key_signatures'].append((barline_tuple, key_signature))10). This should happen everywhere.
gnatures'][-1] = ((metadata['key_signatures'][-1][0][0], last_metadata_update_bar - 1), metadata['key_signatures'][-1][1])
                        return metadata

"""Functions for post-processing the salami slices such that they can be analysed."""tuple, key_signature))

def post_process_salami_slices(salami_slices: List[SalamiSlice], metadata) -> Tuple[List[SalamiSlice], Dict[str, any]]:
    """ Post-process the salami slices.  """
    # Note, most of this could be done in one pass, but for developmental easy we do it in multiple passes.
    salami_slices = set_period_notes(salami_slices)
    salami_slices, metadata = order_voices(salami_slices, metadata)ami_slices: List[SalamiSlice], metadata) -> Tuple[List[SalamiSlice], Dict[str, any]]:
    salami_slices = remove_barline_slice(salami_slices, metadata)    """ Post-process the salami slices.  """
    salami_slices = set_interval_property(salami_slices)
    salami_slices = calculate_offsets(salami_slices)  # Calculate raw offsets from bar startalami_slices)
    salami_slices = calculate_beat_positions(salami_slices, metadata)  # Convert offsets to beats
    ve_barline_slice(salami_slices, metadata)
    return salami_slices, metadata    salami_slices = set_interval_property(salami_slices)
m bar start
def remove_barline_slice(salami_slices: List[SalamiSlice], metadata: Dict[str, any]) -> Tuple[List[SalamiSlice], Dict[str, any]]:i_slices, metadata)  # Convert offsets to beats
    """ Remove all barline slices. """
    salami_slices = [salami_slice for salami_slice in salami_slices if salami_slice.notes[0].note_type != 'barline']
    return salami_slices
(salami_slices: List[SalamiSlice], metadata: Dict[str, any]) -> Tuple[List[SalamiSlice], Dict[str, any]]:
def set_interval_property(salami_slices: List[SalamiSlice]) -> List[SalamiSlice]:    """ Remove all barline slices. """
    """ Set the intervals in the salami slices """i_slice.notes[0].note_type != 'barline']
    for i, salami_slice in enumerate(salami_slices):
        salami_slice.absolute_intervals = salami_slice._calculate_intervals()
        salami_slice.reduced_intervals = salami_slice._calculate_reduced_intervals()]) -> List[SalamiSlice]:
    return salami_slicesices """
 enumerate(salami_slices):
def set_period_notes(salami_slices: List[SalamiSlice]) -> List[SalamiSlice]:intervals()
    """ Set the notes in the salami slices that are periods to the notes in the previous slice """rvals = salami_slice._calculate_reduced_intervals()
    for i, salami_slice in enumerate(salami_slices):
        for voice, note in enumerate(salami_slice.notes):
            if note.note_type == 'period':ami_slices: List[SalamiSlice]) -> List[SalamiSlice]:
                if i == 0:    """ Set the notes in the salami slices that are periods to the notes in the previous slice """
                    raise ValueError("Period note in first slice")
                # Copy note, set 
                salami_slice.notes[voice] = salami_slices[i-1].notes[voice]:
                salami_slices[i-1].notes[voice].new_occurrence = False
    return salami_slicesPeriod note in first slice")

def order_voices(salami_slices: List[SalamiSlice], metadata: Dict[str, any]) -> Tuple[List[SalamiSlice], Dict[str, any]]:] = salami_slices[i-1].notes[voice]
    """ Order the voices from low to high in the salami slices."""
    # Find the highest note in each voice    return salami_slices
    highest_notes = [0] * metadata['voices']
    for salami_slice in salami_slices:Slice], Dict[str, any]]:
        for voice, note in enumerate(salami_slice.notes):igh in the salami slices."""
            if note.note_type == 'note':
                highest_notes[voice] = max(highest_notes[voice], note.midi_pitch)
    for salami_slice in salami_slices:
    # Reorder the voices in the salami slices, and metadata. Highest voice at index 0.
    voice_order = sorted(range(metadata['voices']), key=lambda x: highest_notes[x], reverse=True)
    for salami_slice in salami_slices:                highest_notes[voice] = max(highest_notes[voice], note.midi_pitch)
        # python magic ensures the notes are not overwritten during the for loop
        salami_slice.notes = [salami_slice.notes[voice] for voice in voice_order]
    voice_order = sorted(range(metadata['voices']), key=lambda x: highest_notes[x], reverse=True)
    # Sort all other metadata that is given per voice, according to the sorting of voice_order    for salami_slice in salami_slices:
    metadata['voice_names'] = [metadata['voice_names'][_voice].lower() for _voice in voice_order]    # python magic ensures the notes are not overwritten during the for loop
    salami_slice.notes = [salami_slice.notes[voice] for voice in voice_order]
    for time_sig in metadata['time_signatures']:
        metadata['time_signatures'] = [(time_sig[0], [time_sig[1][voice] for voice in voice_order])] is given per voice, according to the sorting of voice_order
    metadata['voice_names'] = [metadata['voice_names'][_voice].lower() for _voice in voice_order]

    
    _signatures'] = [(time_sig[0], [time_sig[1][voice] for voice in voice_order])]

    return salami_slices, metadata

def calculate_offsets(salami_slices: List[SalamiSlice]) -> List[SalamiSlice]:
    """ Calculate the offset of each slice from the beginning of its bar """
    current_offset = 0data
    current_bar = 1  # Start with bar 1
    ulate_offsets(salami_slices: List[SalamiSlice]) -> List[SalamiSlice]:
    for i, slice in enumerate(salami_slices):ice from the beginning of its bar """
        # If this is a new bar, reset the offset
        if slice.bar != current_bar:ent_bar = 1  # Start with bar 1
            current_offset = 0
            current_bar = slice.bar
        
        # Set the offset for this slice
        slice.offset = current_offset
        
        # Calculate the next offset based on this slice's duration
        # Use the shortest duration as the increment (usually the first voice is sufficient)for this slice
        if slice.notes and any(note and note.note_type in ('note', 'rest') for note in slice.notes):    slice.offset = current_offset
            # Find the first valid note/rest duration
            for note in slice.notes:        # Calculate the next offset based on this slice's duration
                if note and note.note_type in ('note', 'rest'):
                    current_offset += note.durationif slice.notes and any(note and note.note_type in ('note', 'rest') for note in slice.notes):
                    break
    
    return salami_slices         if note and note.note_type in ('note', 'rest'):

def calculate_beat_positions(salami_slices: List[SalamiSlice], metadata) -> List[SalamiSlice]:reak
    """ 
    Calculate the beat position for each slice based on its offset and the time signature. 
    This assigns a beat property to each salami slice with its position in musical beats.
    """def calculate_beat_positions(salami_slices: List[SalamiSlice], metadata) -> List[SalamiSlice]:
    # For each salami slice, find the applicable time signature and convert offset to beats
    timesig_index = 0 on its offset and the time signature. 
    current_time_sig_tuple = metadata['time_signatures'][timesig_index]mi slice with its position in musical beats.
    for slice in salami_slices:
        beat_per_voice = [] # Note: this will be a list with one value, except in debug mode.set to beats
sig_index = 0
        # Find the time signature in effect for this bares'][timesig_index]
        if slice.bar > current_time_sig_tuple[0][1]:
            # Move to the next time signaturept in debug mode.
            timesig_index += 1
            current_time_sig_tuple = metadata['time_signatures'][timesig_index]
        
        current_time_sig = current_time_sig_tuple[1]
        if not current_time_sig_tuple or not current_time_sig:
            raise ValueError(f"No time signature found for bar {slice.bar}")
            
        for voice_idx, _ in enumerate(slice.notes):                                       
            ''' Convert raw offset to beat position.current_time_sig_tuple or not current_time_sig:
            In a time signature like 3/4, one quarter-note spans 0.25 * (4/3) = 0.333 of the bar.slice.bar}")
            A third of a 3/4 bar means it spans the first beat: 0.25 * (4/3) * 3 = 1.
            Since the note with offset 0 falls on beat 1, we need to add 1 to the beat position.lice.notes):                                       
            So instead of : beat_value = (slice.offset * (numerator / denominator) * denominator) + 1''' Convert raw offset to beat position.
            Then, we divide by (1 / 0.25*denominator), stored in a mapping, to normalize against note lengths.-note spans 0.25 * (4/3) = 0.333 of the bar.
            '''3/4 bar means it spans the first beat: 0.25 * (4/3) * 3 = 1.
            numerator, denominator = current_time_sig[voice_idx] note with offset 0 falls on beat 1, we need to add 1 to the beat position.
            beat = slice.offset * numerator / TIME_SIGNATURE_NORMALIZATION_MAP[denominator]   + 1                                      So instead of : beat_value = (slice.offset * (numerator / denominator) * denominator) + 1
            beat_per_voice.append(beat)ping, to normalize against note lengths.
            
            # If we're not in debug moe the beat oncent_time_sig[voice_idx]
            if not DEBUG:minator]   + 1                          
                break
            
        # Check if all voices' beats are the same (only in debug mode)
        if DEBUG and len(beat_per_voice) > 1:
            first_beat = beat_per_voice[0]
            if not all(abs(beat - first_beat) < 0.0001 for beat in beat_per_voice):
                raise ValueError(f"In bar {slice.bar}, offset {slice.offset:.2f}: Not all voices have the same beat. Beats: {beat_per_voice}")eck if all voices' beats are the same (only in debug mode)
(beat_per_voice) > 1:
        # Set the beat position to the first beat value for simplicity.            first_beat = beat_per_voice[0]
        # Round to get rid of floating point division error. NOTE: this rounding might cause unpredictable bugs?            if not all(abs(beat - first_beat) < 0.0001 for beat in beat_per_voice):
        slice.beat = round(beat_per_voice[0], 5)set {slice.offset:.2f}: Not all voices have the same beat. Beats: {beat_per_voice}")

                    # Set the beat position to the first beat value for simplicity.
    return salami_slicession error. NOTE: this rounding might cause unpredictable bugs?
(beat_per_voice[0], 5)

def validate_all_rules(salami_slices, metadata, cp_rules):
    violations = defaultdict(list)

    for i, slice_cur in enumerate(salami_slices):
        current_kwargs = {ata, cp_rules):
            "slice1": slice_cur,
            "slice2": salami_slices[i-1],
            "slice_index": i,
            "metadata": metadata # TODO: metadata can be an *arg, but doesn't really matter    current_kwargs = {
        }_cur,
        # Update violations with the output            "slice2": salami_slices[i-1],
        slice_violations = cp_rules.validate_all_rules(**current_kwargs)            "slice_index": i,
        for rule_name, rule_violations in slice_violations.items():            "metadata": metadata # TODO: metadata can be an *arg, but doesn't really matter
            violations[rule_name].extend(rule_violations)
    
    return dict(violations)
ce_violations.items():
le_violations)
    
def violations_to_df(violations: Dict[str, List[RuleViolation]], metadata) -> pd.DataFrame:
    violation_counts = feature_counts(violations)
    # Add the composer from metadata (defaulting to "Unknown" if not present)
    composer = metadata.get("COM", "Unknown")
    violation_counts["composer"] = composerdef violations_to_df(violations: Dict[str, List[RuleViolation]], metadata) -> pd.DataFrame:
_counts = feature_counts(violations)
    # Create df (with dict keys as columns), and set composer as last column    # Add the composer from metadata (defaulting to "Unknown" if not present)
    df = pd.DataFrame([violation_counts])
    cols = [col for col in df.columns if col != "composer"] + ["composer"]
    df = df[cols]
    # Create df (with dict keys as columns), and set composer as last column
    return dflation_counts])

def feature_counts(violations: Dict[str, List[RuleViolation]]) -> Dict[str, int]:
    counts = {rule: len(violation_list) for rule, violation_list in violations.items()}
    return counts

if __name__ == "__main__":Violation]]) -> Dict[str, int]:
    # filepath = os.path.join("..", "data", "test", "Jos1408-Miserimini_mei.krn")ems()}
    # filepath = os.path.join("..", "data", "test", "Jos1408-test.krn")
#     filepath = os.path.join("..", "data", "test", "Rue1024a.krn")
    filepath = os.path.join("..", "data", "test", "extra_parFifth_rue1024a.krn")if __name__ == "__main__":
    , "data", "test", "Jos1408-Miserimini_mei.krn")
    salami_slices, metadata = parse_kern(filepath))
    salami_slices, metadata = post_process_salami_slices(salami_slices, metadata)  filepath = os.path.join("..", "data", "test", "Rue1024a.krn")
    print(salami_slices)h = os.path.join("..", "data", "test", "extra_parFifth_rue1024a.krn")
    # print(metadata)
th)
    cp_rules = CounterpointRules()    salami_slices, metadata = post_process_salami_slices(salami_slices, metadata)
    violations = validate_all_rules(salami_slices, metadata, cp_rules)alami_slices)
    
    print()
    pprint(violations)    cp_rules = CounterpointRules()






    print()    display(df.head())    print()    df = violations_to_df(violations, metadata)    violations = validate_all_rules(salami_slices, metadata, cp_rules)
    
    print()
    pprint(violations)
    df = violations_to_df(violations, metadata)

    print()
    display(df.head())
    print()

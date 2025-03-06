import os
import importlib
from IPython.display import display
from pprint import pprint
from collections import defaultdict
from typing import List, Dict, Tuple
import copy
import pandas as pd

from constants import DURATION_MAP, PITCH_TO_MIDI, MIDI_TO_PITCH
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

    'voices': [], # TODO list with number of voices for each score-movement 
    'voice_names': [], # list of lists with voice names for each score-movement 
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

            if line.startswith('*'):
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
                        barline_tuple = (last_metadata_update_bar, -1)
                        metadata['voice_names'].append((barline_tuple, voice_names))
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
                        if DEBUG: print(f"Found time signatures at bar {bar_count}: {metadata_dict['time_signatures'][-1][1]}")
                elif line.startswith('*MM'):
                    # Metronome marking, present in some files, but we don't use this.
                    continue
                elif line.startswith('*met'):
                    # TODO: check if we need to implement this later (mensural)
                    continue
                elif line.startswith(('**kern', '*staff', '*clef')):
                    continue
                elif line.startswith('*>'):
                    # Lines indicating structure, or the begin of the structure.
                    continue
                else:
                    raise NotImplementedError(f"Some metadata starting with * not yet implemented. Line {line_idx}: {line}")
            
            elif line.startswith('='):
                # Update bars according to the barnumber found. Note, this might reset the barnumbers, depending on how they are encoded/
                try:
                    bar_num = int(''.join(filter(str.isdigit, line)))
                    if bar_num < bar_count:
                        raise ValueError(f"Bar number {bar_num} is lower than the previous bar number {bar_count}") 
                    bar_count = bar_num
                except AttributeError as ae:
                    raise ae

                            
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
    barline_count = 1

    for line_idx, line in enumerate(note_section):
        new_salami_slice = SalamiSlice(num_voices=int(metadata['voices']))
        barline_updated_flag = False
        
        for token_idx, token in enumerate(line.split()): # TODO check what the **kern line seperator is (\t?)
            new_note, accidental_tracker = kern_token_to_note(kern_token=token, accidental_tracker=accidental_tracker, line_idx=line_idx)
            if new_note is None:
                continue
            else:
                new_salami_slice.add_note(note=new_note, voice=token_idx) 

                # Set the current bar, and handle what needs to happen at a new barline. (Barlines are set to themselves.)
                if new_note.note_type == 'barline' and not barline_updated_flag:
                    barline_updated_flag = True
                    barline_count += 1
                    # Reset the accidental tracker for the next measure
                    accidental_tracker = {key:0 for key in accidental_tracker}
                new_salami_slice.bar = barline_count

                if new_note.note_type == 'final_barline':
                    continue
                    # TODO we can stop parsing here, as the rest of the metadata should be handled further in the 
                    # `parse_kern` function.
                    

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
                if kern_token[i+1] == '=':
                    new_note.note_type = 'final_barline'
                else:
                    new_note.note_type = 'barline'
                # Skips until the after the barnumber. This should also always be the end of the line, since the barline token looks like '=123'.
                i += len(kern_token[i:]) - 1
            else:
                raise ValueError(f"Could not parse character '{c}' in token '{kern_token}', at index '{i}'")
        except Exception as e:
            print(f"For token '{kern_token}', char '{c}', index '{i}', encountered error:\n\n")
            raise e
        
        i += 1
    #endwhile

    # Error checking
    if new_note.duration is None:
        raise ValueError(f"Could not find duration for token '{kern_token}'")

    # TODO: return the accidental tracker, because it is relevant for the entire measure
    return new_note, accidental_tracker


"""Helper functions to parse all the different kinds of lines."""

def parse_timesig(line: str, metadata: dict, bar_count: int) -> dict:
    """ Parse a time signature token and update the metadata dict """
        
    time_sig_tokens = line.strip().split('\t')
    time_signatures = []
    
    for token in time_sig_tokens: 
        if token.startswith('*M'):
            time_sig = token[2:]  # Remove '*M' prefix to get e.g. "2/1"
            if '/' in time_sig:
                numerator, denominator = map(int, time_sig.split('/'))
                time_signatures.append((numerator, denominator))
            elif '*' in time_sig:
                time_signatures.append('*')
            else:
                # Handle cases where only numerator is specified
                time_signatures.append((int(time_sig), 1))

    # Replace '*' with the previous time signature
    for i, time_sig in enumerate(time_signatures):
        if time_sig == '*':
            time_signatures[i] = time_signatures[i-1]
            
    # Record the bar number of this change
    last_metadata_update_bar = bar_count
    barline_tuple = (last_metadata_update_bar, -1)
    
    # If this is not the first time signature encountered, set the ending of the previous one
    if metadata['time_signatures']:
        # If the new time sig is at bar 10, then the previous one will be set from (0, 9), not (0, 10). This should happen everywhere.
        metadata['time_signatures'][-1][0] = (metadata['time_signatures'][-1][0][0], last_metadata_update_bar - 1)
 
    # Store time signatures with the bar number where they changed
    metadata['time_signatures'].append((barline_tuple, time_signatures))
    
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
                            metadata['key_signatures'][-1][0] = (metadata['key_signatures'][-1][0][0], last_metadata_update_bar - 1)

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
                salami_slice.notes[voice].new_occurrence = False
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

    # Make names lowercase
    metadata['voice_names'] = [metadata['voice_names'][voice].lower() for voice in voice_order]

    return salami_slices, metadata
    

def validate_all_rules(salami_slices, metadata, cp_rules):
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



def violations_to_df(violations: Dict[str, List[RuleViolation]], metadata) -> pd.DataFrame:
    violation_counts = feature_counts(violations)
    # Add the composer from metadata (defaulting to "Unknown" if not present)
    composer = metadata.get("COM", "Unknown")
    violation_counts["composer"] = composer

    # Create df (with dict keys as columns), and set composer as last column
    df = pd.DataFrame([violation_counts])
    cols = [col for col in df.columns if col != "composer"] + ["composer"]
    df = df[cols]

    return df

def feature_counts(violations: Dict[str, List[RuleViolation]]) -> Dict[str, int]:
    counts = {rule: len(violation_list) for rule, violation_list in violations.items()}
    return counts

if __name__ == "__main__":
    # filepath = os.path.join("..", "data", "test", "Jos1408-Miserimini_mei.krn")
    # filepath = os.path.join("..", "data", "test", "Jos1408-test.krn")
    filepath = os.path.join("..", "data", "test", "Rue1024a.krn")
    # filepath = os.path.join("..", "data", "test", "extra_parFifth_rue1024a.krn")
    
    salami_slices, metadata = parse_kern(filepath)
    salami_slices, metadata = post_process_salami_slices(salami_slices, metadata)
    # print(salami_slices)
    # print(metadata)

    cp_rules = CounterpointRules()
    violations = validate_all_rules(salami_slices, metadata, cp_rules)
    
    print()
    pprint(violations)
    df = violations_to_df(violations, metadata)

    print()
    display(df.head())
    print()

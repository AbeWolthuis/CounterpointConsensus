#!/usr/bin/env python3
# Auto-generated merged script

import inspect
from collections import defaultdict
from typing import Dict, Tuple, List, Union
import pandas as pd
from typing import Dict, List
import os
import importlib
from IPython.display import display
from pprint import pprint
from typing import List, Dict, Tuple
import copy
from typing import Dict, Tuple, Union

## ========== START of constants.py (global constants) ========== ##
DURATION_MAP = {
    '0': 2.0,  # Special case; longa
    '1': 1.0,  # Whole note
    '2': 0.5,  # Half note
    '4': 0.25,
    '8': 0.125,
    '16': 0.0625,
    '32': 0.03125, 
}
## =========== END of constants.py (global constants) =========== ##

## ========== START of constants.py (global constants) ========== ##
PITCH_TO_MIDI = {
    'CCC': 24, 'DDD-': 25, 'DDD': 26, 'EEE-': 27, 'EEE': 28, 'FFF': 29, 'GGG-': 30, 'GGG': 31, 'AAA-': 32, 'AAA': 33, 'BBB-': 34, 'BBB': 35,
    'CC': 36, 'DD-': 37, 'DD': 38, 'EE-': 39, 'EE': 40, 'FF': 41, 'GG-': 42, 'GG': 43, 'AA-': 44, 'AA': 45, 'BB-': 46, 'BB': 47,
    'C': 48, 'D-': 49, 'D': 50, 'E-': 51, 'E': 52, 'F': 53, 'G-': 54, 'G': 55, 'A-': 56, 'A': 57, 'B-': 58, 'B': 59,
    'c': 60, 'd-': 61, 'd': 62, 'e-': 63, 'e': 64, 'f': 65, 'g-': 66, 'g': 67, 'a-': 68, 'a': 69, 'b-': 70, 'b': 71,
    'cc': 72, 'dd-': 73, 'dd': 74, 'ee-': 75, 'ee': 76, 'ff': 77, 'gg-': 78, 'gg': 79, 'aa-': 80, 'aa': 81, 'bb-': 82, 'bb': 83,
    'ccc': 84, 'ddd-': 85, 'ddd': 86, 'eee-': 87, 'eee': 88, 'fff': 89, 'ggg-': 90, 'ggg': 91, 'aaa-': 92, 'aaa': 93, 'bbb-': 94, 'bbb': 95,
    'cccc': 96, 'dddd-': 97, 'dddd': 98, 'eeee-': 99, 'eeee': 100, 'ffff': 101, 'gggg-': 102, 'gggg': 103, 'aaaa-': 104, 'aaaa': 105, 'bbbb-': 106, 'bbbb': 107,
}

## =========== END of constants.py (global constants) =========== ##

## ========== START of constants.py (global constants) ========== ##
MIDI_TO_PITCH = {v: k for k, v in PITCH_TO_MIDI.items()}

## =========== END of constants.py (global constants) =========== ##

## ========== START of constants.py (global constants) ========== ##
_DIATONIC_PITCH_TO_MIDI = {
    'CCC': 24, 'DDD': 26, 'EEE': 28, 'FFF': 29, 'GGG': 31, 'AAA': 33, 'BBB': 35,
    'CC': 36, 'DD': 38, 'EE': 40, 'FF': 41, 'GG': 43, 'AA': 45, 'BB': 47,
    'C': 48, 'D': 50, 'E': 52, 'F': 53, 'G': 55, 'A': 57, 'B': 59,
    'c': 60, 'd': 62, 'e': 64, 'f': 65, 'g': 67, 'a': 69, 'b': 71,
    'cc': 72, 'dd': 74, 'ee': 76, 'ff': 77, 'gg': 79, 'aa': 81, 'bb': 83,
    'ccc': 84, 'ddd': 86, 'eee': 88, 'fff': 89, 'ggg': 91, 'aaa': 93, 'bbb': 95,
    'cccc': 96, 'dddd': 98, 'eeee': 100, 'ffff': 101, 'gggg': 103, 'aaaa': 105, 'bbbb': 107,
}## =========== END of constants.py (global constants) =========== ##

## ========== START of counterpoint_rules.py (global constants) ========== ##
DEBUG = True

## =========== END of counterpoint_rules.py (global constants) =========== ##

## ========== START of kern_parser.py (global constants) ========== ##
DEBUG = True

metadata_template = {
    # JRP sepcified metadata
    'COM': '',
    'CDT': '',
    'jrpid': '', # josquin research project ID
    'attribution-level@Jos': '',
    'SEGMENT': '', # filename

    'voices': [], # TODO list with number of voices for each score-movement 
    'voice_names': [], # list of lists with voice names for each score-movement 
    'key_signatures': [], # list of key signatures for each score-movement

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


## =========== END of kern_parser.py (global constants) =========== ##

## ========== START of counterpoint_rules.py (class definitions) ========== ##
class RuleViolation:
    def __init__(self, rule_name: str, slice_index: int, bar: int,
                 voice_indices: Union[Tuple, str] = None, voice_names: Union[Tuple, str] = None,
                 note_names: Union[Tuple, str] = None, beat: float = None) -> None:
        self.rule_name = rule_name
        self.slice_index = slice_index
        self.bar = bar

        self.voice_indices = voice_indices  # None if not specific to voice pairs
        self.voice_names = voice_names # 
        self.note_names = note_names
        self.beat = beat

    def __repr__(self):
        if self.voice_indices:
            voice_info = self.voice_indices
        elif self.voice_names:
            voice_info = self.voice_names
        else:
            voice_info = None
        
        base_str = f"({self.rule_name}, bar={self.bar}"
        if self.beat:
            base_str += f", beat={self.beat}"
        if self.note_names:
            base_str += f", note={self.note_names}"
        base_str += f", voice={voice_info}"

        base_str += ")"

        return base_str
        # return (f"({self.rule_name}, bar={self.bar}, voice={voice_info})")

class CounterpointRules:
    allowed_ranges_modern_names = {
        'soprano': (60, 81),
        'alto': (53, 74),
        'tenor': (48, 69),
        'bass': (41, 62)
    }
    allowed_ranges = {
        'cantus': allowed_ranges_modern_names['soprano'],
        'discantus': allowed_ranges_modern_names['soprano'],
        'soprano': allowed_ranges_modern_names['soprano'],

        'contratenor': allowed_ranges_modern_names['alto'],
        'contra': allowed_ranges_modern_names['alto'],
        'altus': allowed_ranges_modern_names['alto'],

        'tenor': allowed_ranges_modern_names['tenor'],

        'bass': allowed_ranges_modern_names['bass'],
        'bassus': allowed_ranges_modern_names['bass']
    }



    def validate_all_rules(self, *args, **kwargs) -> Dict[str, List[RuleViolation]]:
        violations = defaultdict(list)
        for name, func in inspect.getmembers(CounterpointRules, predicate=inspect.isfunction):
            # if DEBUG: print(name)
            if name != "validate_all_rules" and not name.startswith("__"):
                result = func(name, **kwargs)
                violations[name].extend(result)
        return dict(violations)

    
    @staticmethod
    def has_no_parallel_fiths(name, **kwargs) -> Dict[str, List[RuleViolation]]:
        slice_cur = kwargs["slice1"]
        slice_prev = kwargs["slice2"]
        slice_index = kwargs["slice_index"]
        metadata = kwargs["metadata"]

        violations = []
        for voice_pair, interval1 in slice_cur.reduced_intervals.items():
            if interval1 == 7:
                if voice_pair in slice_prev.reduced_intervals:
                    interval2 = slice_prev.reduced_intervals[voice_pair]
                    # Parallel fifth is:
                    # 1. First and second interval are perfect fifths (7 semitones)
                    # 2. The second slice notes are not tied over. Note, a repetition is also seen as a parallel perfect fifth.
                    if (interval2 == interval1) and (slice_cur.notes[voice_pair[0]].new_occurrence and slice_cur.notes[voice_pair[1]].new_occurrence):
                        violations.append(RuleViolation(rule_name=name, slice_index=slice_index, bar=slice_cur.bar,
                            voice_indices=(metadata['voice_names'][voice_pair[0]], metadata['voice_names'][voice_pair[1]]),
                            note_names=(slice_cur.notes[voice_pair[0]].note_name, slice_cur.notes[voice_pair[1]].note_name)))
        return violations
    

    @staticmethod
    def has_valid_range(name, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check that each note in the current slice is within its allowed MIDI range.
        
        allowed_ranges: dict mapping voice index (int) to (min_midi, max_midi)
        """
        slice_cur = kwargs["slice1"]
        slice_index = kwargs["slice_index"]
        metadata = kwargs["metadata"]

        violations = []
        for voice_number, note in enumerate(slice_cur.notes):
            voice_name = metadata["voice_names"][voice_number]
            if voice_name not in CounterpointRules.allowed_ranges:
                raise ValueError(f"Allowed range in function {inspect.currentframe().f_code.co_name} not defined for voice {voice_name}")
            else:
                if note and note.note_type == 'note' and note.midi_pitch != -1:
                    min_pitch, max_pitch = CounterpointRules.allowed_ranges[voice_name]
                    if note.midi_pitch < min_pitch or note.midi_pitch > max_pitch:
                        violations.append(RuleViolation(rule_name=name, slice_index=slice_index, bar=slice_cur.bar, voice_names=voice_name, note_names=note.note_name))
        return violations


    def __has_no_parallel_thirds__(name, **kwargs) -> Dict[str, List[RuleViolation]]:
        slice_cur = kwargs["slice1"]
        slice_prev = kwargs["slice2"]
        slice_index = kwargs["slice_index"]

        violations = []
        for voice_pair, interval1 in slice_cur.reduced_intervals.items():
            # Check for thirds (minor or major)
            if interval1 in (3, 4):
                if voice_pair in slice_prev.reduced_intervals:
                    interval2 = slice_prev.reduced_intervals[voice_pair]
                    # Verify the thirds are of the same quality,
                    # at least one note is a new occurrence,
                    # and the actual pitches (absolute intervals) have changed.
                    if (interval2 == interval1) and \
                    (slice_cur.notes[voice_pair[0]].new_occurrence or slice_cur.notes[voice_pair[1]].new_occurrence):
                        violations.append(
                            RuleViolation(rule_name=name,
                                          slice_index=slice_index,
                                          bar=slice_cur.bar,
                                          voice_indices=voice_pair)
                        )
        return violations
    
    def __has_no_fifths__(name, **kwargs) -> Dict[str, List[RuleViolation]]:
        slice1 = kwargs["slice1"]
        slice2 = kwargs["slice2"]
        slice_index = kwargs["slice_index"]

        violations = []
        for (voice_pair, interval1) in slice1.intervals.items():
            # if True:
            if interval1 == 7:
                violations.append(RuleViolation(name, slice_index, voice_pair))
                # print(violations[-1])
        return violations

## =========== END of counterpoint_rules.py (class definitions) =========== ##

## ========== START of Note.py (class definitions) ========== ##
class SalamiSlice:
    def __init__(self, num_voices=-1, bar=-1) -> None:
        self.offset = 0
        self.num_voices = num_voices
        self.notes = [None] * num_voices
        self.bar = bar

        self.absolute_intervals = None # Will be: Dict[Tuple[int, int], int]
        self.reduced_intervals = None # Will be: Dict[Tuple[int, int], int]

    def add_note(self, note, voice):
        self.notes[voice] = note

    
    def _calculate_intervals(self) -> Dict[Tuple[int, int], int]:
        """
        Compute intervals between all pairs of notes in the slice.
        Returns a dictionary with keys as tuples of voice indices
        and values as intervals in semitones.
        """
        intervals = {}
        for i, note1 in enumerate(self.notes):
            for j, note2 in enumerate(self.notes):
                if i < j and note1 and note2 and note1.note_type == 'note' and note2.note_type == 'note':
                    interval = abs(note1.midi_pitch - note2.midi_pitch)
                    intervals[(i, j)] = interval
                    # intervals[(j, i)] = interval
        return intervals
    
    def _calculate_reduced_intervals(self) -> Dict[Tuple[int, int], int]:
        """
        Compute intervals between all pairs of notes in the slice.
        Returns a dictionary with keys as tuples of voice indices
        and values as intervals in semitones.
        """
        intervals = {}
        for i, note1 in enumerate(self.notes):
            for j, note2 in enumerate(self.notes):
                if i < j and note1 and note2 and note1.note_type == 'note' and note2.note_type == 'note':
                    interval = abs(note1.midi_pitch - note2.midi_pitch) % 12
                    intervals[(i, j)] = interval
        return intervals
    
    def __repr__(self):
        print_str = "[" + ", ".join([str(note) for note in self.notes]) + "]\n"
        return print_str


    def check(self):
        if self.num_voices == -1:
            raise ValueError("Number of voices is not set")

class Note():
    _POSSIBLE_TYPES = ('note', 'rest', 'period', 'barline', 'final_barline')
    midi_to_pitch = MIDI_TO_PITCH
    pitch_to_midi = PITCH_TO_MIDI
    # __slots__ = ('midi_pitch', 'duration', 'note_type', 'new_occurrence')

    def __init__(self, 
                 midi_pitch: int = -1,
                 duration: float = -1,
                 note_type: str = None,
                 new_occurence: bool = True) -> None:
        
        self.midi_pitch = midi_pitch
        self.duration = duration
        self.note_type = note_type
        self.new_occurrence = new_occurence
        return
    
    def __repr__(self):
        if self.note_type == 'note':
            return f"({self.note_type}: {self.duration}, {self.midi_pitch})"
        else:
            return f"({self.note_type}: {self.duration})"
        
    @property
    def octave_reduced(self) -> int:
        """Compute the octave reduced note number based on MIDI pitch where A (MIDI 57) is 0."""
        if self.midi_pitch == -1:
            return -1
        return (self.midi_pitch - 9) % 12
    
    @property
    def note_name(self) -> Union[str, None]:
        """Get the note name of the note."""
        if self.midi_pitch == -1:
            return None
        return self.midi_to_pitch[self.midi_pitch]
    
    # Checks to make sure the note is set
    def check(self):
        if self.midi_pitch == -1 or self.duration == -1: 
            raise ValueError("Note pitch or duration is not set")
        elif self.note_type not in self._POSSIBLE_TYPES:
            raise ValueError("Note type is not set")
        
        return
## =========== END of Note.py (class definitions) =========== ##

## ========== START of constants.py (non-class code) ========== ##


# Duration mapping

## =========== END of constants.py (non-class code) =========== ##

## ========== START of counterpoint_rules.py (non-class code) ========== ##




## =========== END of counterpoint_rules.py (non-class code) =========== ##

## ========== START of data_preparation.py (non-class code) ========== ##




def violations_to_df(violations: Dict[str, List[RuleViolation]], metadata) -> pd.DataFrame:
    violation_counts = feature_counts(violations)
    # Create DataFrame with one row, where keys become column names
    df = pd.DataFrame([violation_counts])
    
    return df

def feature_counts(violations: Dict[str, List[RuleViolation]]) -> Dict[str, int]:
    # Count the number of violations for each rule
    counts = {rule: len(violation_list) for rule, violation_list in violations.items()}
    return counts

#
## =========== END of data_preparation.py (non-class code) =========== ##

## ========== START of kern_parser.py (non-class code) ========== ##



# Our own modules




    
def parse_kern(kern_filepath) -> Tuple[List, Dict]:
    metadata = {k:v for k,v in metadata_template.items()}
    
    with open(kern_filepath, 'r') as f:
        metadata_flag = True
        bar_count = 0 
        last_metadata_update_bar = 0

        note_section = []
        lines = f.readlines()

        # Parse metadata, and gather the section of the file containing notes
        for line in lines:
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
                        barline_tuple = (last_metadata_update_bar, bar_count)
                        metadata['voice_names'].append((barline_tuple, voice_names))
                    else:
                        raise NotImplementedError("Something in parsing voices (*I) not yet implemented.")
                elif line.startswith('*k'):
                    # Check that keysigs are equal
                    keysig_tokens = line.split('\t')
                    if len(set(keysig_tokens)) != 1: 
                        raise ValueError("Different key signatures in voices not yet implemented.")
                    
                    # Parse the key signature
                    keysig_token = keysig_tokens[3:] # Remove *k[ from the start
                    key_signature = {k:v for k,v in key_signature_template.items()}
                    i = 0 
                    while i < len(keysig_token):
                        c = keysig_token[i]
                        if c == ']':
                            break
                        elif c in key_signature_template.keys():
                            if keysig_token[i+1] == '+':
                                key_signature_template[c] += 1
                            elif keysig_token[i+1] == '-':
                                key_signature_template[c] -= 1
                            i += 1
                        elif c  in ['+', '-']:
                            raise ValueError(f"Could not parse key signature '{keysig_token}', reached accidental '{c}' unpredictedly.")                     
                        else:
                            raise ValueError(f"Could not parse key signature '{keysig_token}'")                     
                        i += 1
                        #endwhile

                    # Set keysig
                    metadata['key_signatures'].append(key_signature)
                elif line.startswith('*M'):
                    raise NotImplementedError("Time signatures not yet implemented.")
                elif line.startswith(('**kern', '*staff', '*clef')):
                    continue
                else:
                    raise NotImplementedError("Some metadata starting with * not yet implemented.")
            
            elif line.startswith('='):
                try:
                    a = int(line[1])
                    bar_count += 1
                except AttributeError as ae:
                    raise ae
            
            a = a
            ''' # old attempt
            if metadata_flag:
                if line[0:6] in metadata_linestarts_map.keys():
                    # Splt '!!!jrpid: xxx' to 'jrpid',' xxx'
                    a = line.split(':')[0][0:6]
                    metadata_key, metadata_value = metadata_linestarts_map[a], line.split(':')[1].strip()
                    metadata[metadata_key] = metadata_value
                elif line.startswith('*I"'):
                    # Add voice names, in given order
                    metadata['voice_names'] = [voice_name.replace('*I"', '') for voice_name in line.split()]
                elif line.startswith('='):
                    metadata_flag = False
                
                # End of file meta-data handling:
            else:
                if line.startswith('*-'):
                    # Final barline has been reached, end of file can still contain some metadata
                    # We don't add this line (TODO: should we for parsing consistency?)
                    metadata_flag = True
                else:
                    note_section.append(line)
'''

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


## =========== END of kern_parser.py (non-class code) =========== ##

## ========== START of Note.py (non-class code) ========== ##




## =========== END of Note.py (non-class code) =========== ##

## ========== START of main.py (non-class code) ========== ##





## =========== END of main.py (non-class code) =========== ##

## ========== START of main.py (main function) ========== ##
def main():
    # Settings stuff

    # Load kern

    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, '..', "..", "data", "test", "extra_parFifth_rue1024a.krn")
    # filepath = os.path.join("..", '..', "data", "test", "extra_parFifth_rue1024a.krn")
    
    # Parse
    salami_slices, metadata = parse_kern(filepath)
    salami_slices, metadata = post_process_salami_slices(salami_slices, metadata)

    # Analyze
    cp_rules = CounterpointRules()
    violations = cp_rules.validate_all_rules(salami_slices, metadata, cp_rules)
    
    print()
    pprint(violations)
    df = violations_to_df(violations, metadata)


    # Classify


    # Analyze classification


    return



## =========== END of main.py (main function) =========== ##

if __name__ == '__main__':
    main()

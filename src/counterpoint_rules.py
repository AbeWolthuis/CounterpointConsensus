import inspect
from collections import defaultdict
from typing import Dict, Tuple, List, Union

from Note import Note, SalamiSlice

from constants import DURATION_MAP, PITCH_TO_MIDI, MIDI_TO_PITCH

DEBUG = True

class RuleViolation:
    def __init__(self, rule_name: str, rul_idx: int, slice_index: int, bar: int,
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
        voice_info = None
        
        # Convert voice_indices from 0-indexed to 1-indexed for display
        if self.voice_indices:
            if isinstance(self.voice_indices, tuple) and all(isinstance(i, int) for i in self.voice_indices):
                # For tuples of integers, add 1 to make them 1-indexed
                voice_info = tuple(i + 1 for i in self.voice_indices)
            else:
                # For voice names, use as is
                voice_info = self.voice_indices
        elif self.voice_names:
            voice_info = self.voice_names
            
        # Shorten rule name to 7 characters
        short_rule = self.rule_name[:7] + '...'
            
        base_str = f"({short_rule}, bar={self.bar}"
        if self.beat:
            base_str += f", beat={self.beat}"
        if self.note_names:
            base_str += f", note={self.note_names}"
        if voice_info is not None:
            base_str += f", voice={voice_info}"

        base_str += ")"

        return base_str
    



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
        only_validate_rules = kwargs['only_validate_rules'] if 'only_validate_rules' in kwargs else None
        
        violations = defaultdict(list)
        for name, func in inspect.getmembers(CounterpointRules, predicate=inspect.isfunction):
            # if DEBUG: print(name)
            if name != "validate_all_rules" and not name.startswith("__"):
                # If only_validate_rules is provided, check if the rule is in the list
                if (not only_validate_rules) or (name in only_validate_rules):
                    result = func(self, name, **kwargs)
                    violations[name].extend(result)
        return dict(violations)


    """
    %%% Rules for any amount of voices %%%
    """
    @staticmethod
    def use_longa_only_at_endings(self, rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check that the last slice of each voice has a longa or brevis note.
        """
        rule_id = 1


        cur_slice = kwargs["slice1"]
        cur_slice_index = kwargs["slice_index"]
        metadata = kwargs["metadata"]

        violations = []
        for voice_number, note in enumerate(cur_slice.notes):
            if note and note.note_type == 'note' and note.midi_pitch != -1:
                    # Check if note is longa or brevis
                    if note.duration in [2.0]:
                        if note.new_occurrence:
                            # Check if note is the last note in the section, or piece.
                            if not ( (cur_slice.bar+1 in metadata['section_ends']) or (cur_slice.bar == metadata['total_bars']) ):

                                # Include note name for violation
                                violations.append(RuleViolation(rule_name=rulename, rule_id=rule_id, slice_index=cur_slice_index, bar=cur_slice.bar, voice_indices=voice_number, note_names=note.note_name))
                    
        return violations

    
    @staticmethod
    def no_parallel_fiths(self, name, **kwargs) -> Dict[str, List[RuleViolation]]:
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
                        # Include notes from both slices: first the previous slice, then the current slice
                        prev_notes = (slice_prev.notes[voice_pair[0]].note_name, slice_prev.notes[voice_pair[1]].note_name)
                        curr_notes = (slice_cur.notes[voice_pair[0]].note_name, slice_cur.notes[voice_pair[1]].note_name)
                        all_notes = prev_notes + curr_notes
                        
                        violations.append(RuleViolation(
                            rule_name=name, 
                            slice_index=slice_index, 
                            bar=slice_cur.bar,
                            voice_indices=voice_pair,
                            note_names=all_notes,
                            beat=slice_cur.beat  # Include beat position
                        ))
        return violations
    


    @staticmethod
    def has_valid_range__(self, name, **kwargs) -> Dict[str, List[RuleViolation]]:
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

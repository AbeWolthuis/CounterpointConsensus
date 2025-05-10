import inspect
from collections import defaultdict
from typing import Dict, Tuple, List, Union

from Note import Note, SalamiSlice

from constants import DURATION_MAP, PITCH_TO_MIDI, MIDI_TO_PITCH

DEBUG = True

class RuleViolation:
    def __init__(self, rule_name: str, rule_id: int, slice_index: int, bar: int,
                 voice_indices: Union[Tuple, str], voice_names: Union[Tuple, str],
                 note_names: Union[Tuple, str], beat: float, original_line_number: int) -> None:
        self.rule_name = rule_name
        self.rule_id = rule_id
        self.slice_index = slice_index
        self.bar = bar
        
        self.voice_indices = voice_indices  # None if not specific to voice pairs
        self.voice_names = voice_names      # 
        self.note_names = note_names
        self.beat = beat

        # Record the original line number of the slice in the source file
        self.original_line_number = original_line_number

    def __repr__(self):
        voice_info = None
        
        # Prepare the voice-info (0 indexed, or name of the voice)
        if self.voice_indices:
            voice_info = self.voice_indices
        elif self.voice_names:
            voice_info = self.voice_names
            
        # Shorten rule name to rule_display_len characters
        rule_display_len = 7
        short_rule = self.rule_name[:rule_display_len] + '...' + ', id=' + str(self.rule_id)
            
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

    old_to_modern_voice_name_mapping = { 
        'soprano': 'soprano',
        'cantus': 'soprano',
        'discantus': 'soprano',
        'superius': 'soprano',
        
        'alto': 'alto',
        'contratenor': 'alto',
        'contra': 'alto',
        'altus': 'alto',

        'tenor': 'tenor',

        'bass': 'bass',
        'bassus': 'bass'
    }


    def validate_all_rules(self, *args, **kwargs) -> Dict[str, List[RuleViolation]]:
        only_validate_rules = kwargs['only_validate_rules'] if 'only_validate_rules' in kwargs else None
        
        violations = defaultdict(list)
        for name, func in inspect.getmembers(CounterpointRules, predicate=inspect.isfunction):
            # if DEBUG: print(name)
            if name != "validate_all_rules" and not name.startswith("__"):
                # If only_validate_rules is provided, check if the rule is in the list
                if (not only_validate_rules) or (name in only_validate_rules):
                    result = func(name, **kwargs)
                    violations[name].extend(result)
        return dict(violations)


    """
    %%% Rules for any amount of voices %%%
    """

    """ %% Rhytm and meter %% """

    @staticmethod
    def longa_only_at_endings(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check that the last slice of each voice has a longa or brevis note.
        """
        rule_id = 1

        salami_slices = kwargs['salami_slices']
        curr_slice_index = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_index]
        
        metadata = kwargs["metadata"]

        violations = []
        for voice_number, note in enumerate(curr_slice.notes):
            if note and note.note_type == 'note' and note.midi_pitch != -1:
                    # Check if note is longa or brevis
                    if note.is_longa:
                        if note.is_new_occurrence:
                            # Check if note is the last note in the section, or piece.
                            # We add one to the current bar, to check if the bar after is the start of the new section.
                            if not ( (curr_slice.bar+1 in metadata['section_starts']) or (curr_slice.bar == metadata['total_bars']) ):
                                # Include note name for violation
                                violations.append(RuleViolation(
                                    rule_name=rulename,
                                    rule_id=rule_id,
                                    slice_index=curr_slice_index,
                                    original_line_number=curr_slice.original_line_number,
                                    beat=curr_slice.beat,
                                    bar=curr_slice.bar,
                                    voice_indices=voice_number,
                                    voice_names=metadata['voice_names'][voice_number],
                                    note_names=note.note_name,
                                ))
                    
        return violations

    """ %% Melody %% """

    """ % Intervals and leaps % """
    
    @staticmethod
    def leap_too_large(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check for large leaps bigger than minor sixth (9 semitones). An octave jump is allowed. 
        """
        rule_id = 23

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs['slice_index']
        curr_slice = salami_slices[curr_slice_idx]

        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            prev_slice = curr_slice.previous_new_occurrence_per_voice[voice_number]
            # First check if this is a note, and if it has a previous slice with a new occurence (e.g. is not one of the first slices in the piece)
            if (curr_note.note_type == 'note') and (prev_slice is not None):
                prev_note = prev_slice.notes[voice_number]
                if prev_note is not None:
                    # If we found a previous note, check if we cross a section.
                    if curr_slice.bar in metadata['section_starts'] and prev_slice.bar in metadata['section_ends']:
                        continue # If there between the curr and prev note there is a section, we skip the check.
                    else:
                        # If we didnt cross a section, check the leap.
                        interval = abs(curr_note.midi_pitch - prev_note.midi_pitch)
                        if interval in (9, 10, 11) or interval >= 13:
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_idx,
                                bar=curr_slice.bar,
                                voice_names=metadata['voice_names'][voice_number],
                                # voice_indices=voice_number,
                                note_names=(prev_note.compact_summary, curr_note.compact_summary)
                                # note_names=(prev_note.note_name, curr_note.note_name)
                            ))


        return violations

    @staticmethod
    def interval_order_motion(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        In leaps: in ascending motion, the larger intervals come first; in descending, the smaller first.  
        """
        rule_id = 25

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs['slice_index']
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]

        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            prev_1st_slice = curr_slice.previous_new_occurrence_per_voice[voice_number]
            if prev_1st_slice:
                prev_2nd_slice = prev_1st_slice.previous_new_occurrence_per_voice[voice_number]

            # First check if this is a note, and if it has two previous slices with a new occurence (e.g. is not one of the first slices in the piece)
            if (curr_note.note_type == 'note') and (prev_1st_slice is not None) and (prev_2nd_slice is not None):
                prev_1st_note = prev_1st_slice.notes[voice_number]
                prev_2nd_note = prev_2nd_slice.notes[voice_number]

                if prev_1st_note is not None and prev_2nd_note is not None:
                    # Check if we cross a section between either of the slices
                    if (curr_slice.bar in metadata['section_starts'] and prev_1st_slice.bar in metadata['section_ends']) or \
                       (prev_1st_slice.bar in metadata['section_starts'] and prev_2nd_slice.bar in metadata['section_ends']):
                        continue
                    else:
                        # If we didnt cross a section, check the leap.
                        interval1 = abs(curr_note.midi_pitch - prev_1st_note.midi_pitch)
                        interval2 = abs(prev_1st_note.midi_pitch - prev_2nd_note.midi_pitch)
                        
                        # Check if the interval is a leap.
                        if (interval1 > 2) and (interval2 > 2):
                            # If the motion is ascending, first interval in time (interval2) should be bigger or equal to interval1. Violation if this is not so.
                            if ( (curr_note.midi_pitch > prev_1st_note.midi_pitch > prev_2nd_note.midi_pitch) and not (interval2 >= interval1)) or \
                            ( (curr_note.midi_pitch < prev_1st_note.midi_pitch < prev_2nd_note.midi_pitch) and not (interval2 <= interval1)):
                                violations.append(RuleViolation(
                                    rule_name=rulename,
                                    rule_id=rule_id,
                                    slice_index=curr_slice_idx,
                                    original_line_number=curr_slice.original_line_number,
                                    beat=curr_slice.beat,
                                    bar=curr_slice.bar,
                                    voice_indices=voice_number,
                                    voice_names=metadata['voice_names'][voice_number],
                                    note_names=(prev_2nd_note.note_name, prev_1st_note.note_name, curr_note.note_name)
                                ))

        return violations

    @staticmethod
    def leap_approach_left_opposite(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check that a leap in one direction is approached by motion (step or leap) in the opposite direction.
        Also, it must be left in the direction of 
        """
        rule_id = 26

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]

        violations = []


        for voice_number, curr_note in enumerate(curr_slice.notes):
            prev_slice = curr_slice.previous_new_occurrence_per_voice[voice_number]
            next_slice = curr_slice.next_new_occurrence_per_voice[voice_number]
            if (curr_note.note_type == 'note') and (prev_slice is not None) and (next_slice is not None):
                prev_prev_slice = prev_slice.previous_new_occurrence_per_voice[voice_number]
                if prev_prev_slice is not None:
                    # First check if this is a note, and if 
                    prev_prev_note = prev_prev_slice.notes[voice_number]
                    prev_note = prev_slice.notes[voice_number]
                    next_note = next_slice.notes[voice_number]
                    if (prev_note is not None) and (next_note is not None) and (prev_prev_note is not None):
                        # Check if we cross a section between either of the slices
                        if (curr_slice.bar in metadata['section_starts'] and prev_slice.bar in metadata['section_ends']) or \
                        (curr_slice.bar in metadata['section_ends'] and next_slice.bar in metadata['section_starts']):
                            continue
                        else:
                            # If we didnt cross a section, check the leap.
                            interval_leap = abs(curr_note.midi_pitch - prev_note.midi_pitch)
                            interval1 = abs(prev_note.midi_pitch - prev_note.midi_pitch)
                            interval2 = abs(curr_note.midi_pitch - next_note.midi_pitch)
                            
                            # Check if the interval is a leap.
                            if interval_leap > 2:
                                # If the motion is ascending, first interval in time (interval1) should be descending and interval2 also descending.
                                # If the motion is descending, first interval in time (interval1) should be ascending.
                                ascending_leap_violation = False
                                descending_leap_violation = False

                                if (curr_note.midi_pitch > prev_note.midi_pitch): # Ascending leap:
                                    # Check if the first interval is descending and the second interval is descending
                                    if (prev_note.midi_pitch > prev_prev_note.midi_pitch) or (curr_note.midi_pitch < next_note.midi_pitch):
                                        ascending_leap_violation = True
                                if (curr_note.midi_pitch < prev_note.midi_pitch): # Descending leap:
                                    # Check if the first interval is ascending and the second interval is ascending
                                    if (prev_note.midi_pitch < prev_prev_note.midi_pitch) or (curr_note.midi_pitch > next_note.midi_pitch):
                                        descending_leap_violation = True
                                
                                if ascending_leap_violation or descending_leap_violation:
                                    violations.append(RuleViolation(
                                        rule_name=rulename,
                                        rule_id=rule_id,
                                        slice_index=curr_slice_idx,
                                        original_line_number=curr_slice.original_line_number,
                                        beat=curr_slice.beat,
                                        bar=curr_slice.bar,
                                        voice_indices=voice_number,
                                        voice_names=metadata['voice_names'][voice_number],
                                        note_names=(prev_prev_note.note_name, prev_note.note_name, next_note.note_name, curr_note.note_name)
                                    ))
                                
        return violations  



    @staticmethod
    def no_parallel_fiths(name, **kwargs) -> Dict[str, List[RuleViolation]]:
        
        rule_id = 1000

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs['slice_index']
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]

        

        violations = []

        for voice_pair, interval1 in curr_slice.reduced_intervals.items():
            if interval1 == 7:

                if voice_pair in slice_prev.reduced_intervals:
                    interval2 = slice_prev.reduced_intervals[voice_pair]
                    # Parallel fifth is:
                    # 1. First and second interval are perfect fifths (7 semitones)
                    # 2. The second slice notes are not tied over. Note, a repetition is also seen as a parallel perfect fifth.
                    if (interval2 == interval1) and (slice_cur.notes[voice_pair[0]].is_new_occurrence and slice_cur.notes[voice_pair[1]].is_new_occurrence):
                        # Include notes from both slices: first the previous slice, then the current slice
                        prev_notes = (slice_prev.notes[voice_pair[0]].note_name, slice_prev.notes[voice_pair[1]].note_name)
                        curr_notes = (slice_cur.notes[voice_pair[0]].note_name, slice_cur.notes[voice_pair[1]].note_name)
                        all_notes = prev_notes + curr_notes
                        
                        violations.append(RuleViolation(
                            rule_name=name, 
                            rule_id=rule_id,
                            slice_index=slice_index, 
                            bar=slice_cur.bar,
                            voice_indices=voice_pair,
                            note_names=all_notes,
                            beat=slice_cur.beat  # Include beat position
                        ))
        return violations
    


    @staticmethod
    def has_valid_range__(name, **kwargs) -> Dict[str, List[RuleViolation]]:
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


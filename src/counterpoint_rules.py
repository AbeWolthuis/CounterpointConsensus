import inspect
from collections import defaultdict
from typing import Dict, Tuple, List, Union

from Note import Note, SalamiSlice

from constants import DURATION_MAP, PITCH_TO_MIDI, MIDI_TO_PITCH, TIME_SIGNATURE_STRONG_BEAT_MAP

DEBUG = True

class RuleViolation:
    def __init__(self, rule_name: str, rule_id: int, slice_index: int, bar: int,
                 voice_indices: Union[Tuple, str], voice_names: Union[Tuple, str],
                 note_names: Union[Tuple, str], beat: float, original_line_num: int) -> None:
        self.rule_name = rule_name
        self.rule_id = rule_id
        self.slice_index = slice_index
        self.bar = bar
        
        self.voice_indices = voice_indices  # None if not specific to voice pairs
        self.voice_names = voice_names      # 
        self.note_names = note_names
        self.beat = beat

        # Record the original line number of the slice in the source file
        self.original_line_num = original_line_num

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
        #'contratenor': 'alto',
        'contra': 'alto',
        'altus': 'alto',
        'contratenoraltus': 'alto',

        'tenor': 'tenor',

        'bass': 'bass',
        'bassus': 'bass',
        'contratenorbassus': 'bass',
    }

    """ Helper functions """

    def validate_all_rules(self, *args, **kwargs) -> Dict[str, List[RuleViolation]]:
        only_validate_rules = kwargs['only_validate_rules'] if 'only_validate_rules' in kwargs else None
        
        violations = defaultdict(list)
        for name, func in inspect.getmembers(CounterpointRules, predicate=inspect.isfunction):
            # if DEBUG: print(name)
            if name != "validate_all_rules" and not name.startswith("_"):
                # If only_validate_rules is provided, check if the rule is in the list
                if (not only_validate_rules) or (name in only_validate_rules):
                    result = func(name, **kwargs)
                    violations[name].extend(result)
        return dict(violations)
    
    @staticmethod
    def _chronological_slices_cross_section(slices_in_order: List[SalamiSlice | None], metadata: Dict) -> bool:
        """
        Checks if a section boundary is crossed between any two consecutive slices.
        Slices should be provided in chronological order (earliest to latest).
        Returns True if a section is crossed, False otherwise.
        """
        if not slices_in_order or len(slices_in_order) < 2:
            return False

        section_starts = metadata.get('section_starts', set())
        section_ends = metadata.get('section_ends', set())

        for i in range(len(slices_in_order) - 1):
            prev_s = slices_in_order[i]
            curr_s = slices_in_order[i+1]

            # If either slice in a pair is None, we can't determine a crossing for that pair.
            if prev_s is None or curr_s is None:
                continue

            # A section is crossed if the current slice's bar is a section start
            # AND the previous slice's bar is a section end.
            if curr_s.bar in section_starts and prev_s.bar in section_ends:
                return True
        return False


    """
    %%% Rules for any amount of voices %%%
    """

    """ %% Rhytm and meter %% """


    """ % Note values % """
    @staticmethod
    def longa_only_at_endings(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check that the last slice of each voice has a longa or brevis note.
        """
        rule_id = '1'

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
                                    original_line_num=curr_slice.original_line_num,
                                    beat=curr_slice.beat,
                                    bar=curr_slice.bar,
                                    voice_indices=voice_number,
                                    voice_names=metadata['voice_names'][voice_number],
                                    note_names=note.spelled_name,
                                ))
                    
        return violations

    @staticmethod
    def brevis_at_begin_end(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """ The brevis is used at the end and occasionally at the beginning [of a piece or section]. """
        rule_id = '2'

        salami_slices = kwargs['salami_slices']
        curr_slice_index = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_index]
        
        metadata = kwargs["metadata"]

        violations = []
        for voice_number, note in enumerate(curr_slice.notes):
            if note and note.note_type == 'note' and note.midi_pitch != -1:
                # Check if note is a brevis (duration of 2.0, which is twice a whole note)
                if note.duration == DURATION_MAP['0']:  # Brevis duration
                    if note.is_new_occurrence:  # Only check new occurrences to avoid counting tied notes
                        # Check if note is in an allowed position:
                        # 1. First bar of the piece (bar 1)
                        # 2. Final bar of the piece
                        # 3. First bar of a new section
                        # 4. Final bar of a section
                        
                        allowed_position = False
                        
                        # Check if it's the first bar of the piece
                        if curr_slice.bar == 1:
                            allowed_position = True
                        
                        # Check if it's the final bar of the piece
                        elif curr_slice.bar == metadata['total_bars']:
                            allowed_position = True
                        
                        # Check if it's the first bar of a new section
                        elif curr_slice.bar in metadata['section_starts']:
                            allowed_position = True
                        
                        # Check if it's the final bar of a section
                        elif curr_slice.bar in metadata['section_ends']:
                            allowed_position = True
                        
                        # If not in an allowed position, it's a violation
                        if not allowed_position:
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_index,
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=voice_number,
                                voice_names=metadata['voice_names'][voice_number],
                                note_names=note.spelled_name,
                            ))
                    
        return violations

    """ % Dots and ties % """
    @staticmethod
    def tie_into_strong_beat(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check if the current note is a tie-end and falls on a strong beat.
        """
        rule_id = '14a'
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]

        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Gather all slices that contain the notes that need to be checked
            if curr_note.note_type == 'note' and curr_note.is_tie_end:
                # Check if the current slice is a strong beat
                time_sig = metadata['time_signatures'][curr_slice.bar][voice_number]
                numerator = str(time_sig[0])
                denomintator = str(time_sig[1])
                if numerator in TIME_SIGNATURE_STRONG_BEAT_MAP and denomintator in TIME_SIGNATURE_STRONG_BEAT_MAP[numerator]:
                    strong_beats = TIME_SIGNATURE_STRONG_BEAT_MAP[numerator][denomintator]
                    if curr_slice.beat in strong_beats:
                        violations.append(RuleViolation(
                            rule_name=rulename,
                            rule_id=rule_id,
                            slice_index=curr_slice_idx,
                            original_line_num=curr_slice.original_line_num,
                            beat=curr_slice.beat,
                            bar=curr_slice.bar,
                            voice_indices=voice_number,
                            voice_names=metadata['voice_names'][voice_number],
                            note_names=curr_note.note_name
                        ))
                else:
                    raise ValueError(f"Time signature {time_sig} not found in TIME_SIGNATURE_STRONG_BEAT_MAP")
            

        return violations

    @staticmethod
    def tie_into_weak_beat(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check if the current note is a tie-end and falls on a strong beat.
        """
        rule_id = '14b'
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]

        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Gather all slices that contain the notes that need to be checked
            if curr_note.note_type == 'note' and curr_note.is_tie_end:
                # Check if the current slice is a strong beat
                time_sig = metadata['time_signatures'][curr_slice.bar][voice_number]
                numerator = str(time_sig[0])
                denomintator = str(time_sig[1])
                if numerator in TIME_SIGNATURE_STRONG_BEAT_MAP and denomintator in TIME_SIGNATURE_STRONG_BEAT_MAP[numerator]:
                    strong_beats = TIME_SIGNATURE_STRONG_BEAT_MAP[numerator][denomintator]
                    if curr_slice.beat not in strong_beats:
                        violations.append(RuleViolation(
                            rule_name=rulename,
                            rule_id=rule_id,
                            slice_index=curr_slice_idx,
                            original_line_num=curr_slice.original_line_num,
                            beat=curr_slice.beat,
                            bar=curr_slice.bar,
                            voice_indices=voice_number,
                            voice_names=metadata['voice_names'][voice_number],
                            note_names=curr_note.note_name
                        ))
                else:
                    raise ValueError(f"Time signature {time_sig} not found in TIME_SIGNATURE_STRONG_BEAT_MAP")
            

        return violations


    """ %% Harmony %% """
    """ % Chords % """

    @staticmethod
    def non_root_1st_inv_maj(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check that only major and minor triads in root or 1st inversion are used when there are 3+ sounding voices.
        """
        rule_id = '18'

        salami_slices = kwargs['salami_slices']
        curr_slice_index = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_index]
        
        metadata = kwargs["metadata"]

        violations = []
        
        # Get chord analysis
        chord_analysis = curr_slice.chord_analysis
        if chord_analysis is None:
            # In this case, there should be all rests in the slice. Check for this, and else raise an error.
            if not all(note.note_type == 'rest' for note in curr_slice.notes):
                raise ValueError(f"Slice {curr_slice_index} does not have chord analysis, but contains notes: {curr_slice.notes}")
            else:
                return violations  # No chord analysis, no violations.
        else:
            # Count sounding notes (not rests)
            sounding_notes = [note for note in curr_slice.notes 
                            if note and note.note_type == 'note' and note.midi_pitch != -1]
            
            # Only apply this rule if there are 3 or more sounding voices
            if len(sounding_notes) < 3:
                return violations
            
            # Check if the chord is a major or minor triad
            maj_triad = chord_analysis['is_major_triad']
            min_triad = chord_analysis['is_minor_triad']
            inv = chord_analysis['inversion']
            
            # Allow major OR minor triads in root or 1st inversion
            if (maj_triad or min_triad) and inv in ['root', '1st']:
                # This is allowed, no violation
                pass
            else:
                # This is a violation - not a major/minor triad in root/1st inversion
                violations.append(RuleViolation(
                    rule_name=rulename,
                    rule_id=rule_id,
                    slice_index=curr_slice_index,
                    original_line_num=curr_slice.original_line_num,
                    beat=curr_slice.beat,
                    bar=curr_slice.bar,
                    voice_indices=0, # Top voice
                    voice_names=metadata['voice_names'][0],
                    note_names=chord_analysis['sounding_note_names_voice_order']
                ))           

        return violations


    """ %% Melody %%   """

    """ % Intervals and leaps % """
    @staticmethod
    def leap_too_large(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check for large leaps bigger than minor sixth (9 semitones). An octave jump is allowed. 
        """
        rule_id = '22'

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs['slice_index']
        curr_slice = salami_slices[curr_slice_idx]

        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            prev_slice = curr_slice.previous_note_per_voice[voice_number]
            # First check if this is a note, and if it has a previous slice with a new occurence (e.g. is not one of the first slices in the piece)
            if (curr_note.note_type == 'note') and (prev_slice is not None):

                # Gather all relevant notes
                prev_note = prev_slice.notes[voice_number]

                # Gather all relevant rests
                prev_rest_slice = curr_slice.previous_rest_per_voice[voice_number]

                if prev_note is not None:
                    # If we found a previous note, check if we cross a section.
                    section_crossed = curr_slice.bar in metadata['section_starts'] and prev_slice.bar in metadata['section_ends']
                    # Check if we cross a rest
                    rest_between_curr_prev = prev_rest_slice and prev_rest_slice.original_line_num > prev_slice.original_line_num

                    if section_crossed:
                        continue 
                    elif rest_between_curr_prev:
                        continue
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
                            ))


        return violations
    
    @staticmethod
    def leap_approach_left_opposite(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Check that a leap in one direction is approached by motion (step or leap) in the opposite direction.
        Also, it must be left in the opposite direction of the leap itself.
        """
        rule_id = '23'

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]

        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Gather all slices that contain the notes that need to be checked
            next_slice = curr_slice.next_note_per_voice[voice_number]
            prev_slice = curr_slice.previous_note_per_voice[voice_number]
            prev_prev_slice = None if prev_slice is None else prev_slice.previous_note_per_voice[voice_number]
            
            if (curr_note.note_type == 'note') and (prev_slice is not None) \
                and (prev_prev_slice is not None) and (next_slice is not None):

                # Gather all notes that need to be checked in the rule
                next_note = next_slice.notes[voice_number]
                prev_note = prev_slice.notes[voice_number]
                prev_prev_note = prev_prev_slice.notes[voice_number]

                # Gather all the slices of the rests that could potentially be in between any of the notes
                next_rest_slice = curr_slice.next_rest_per_voice[voice_number]
                prev_rest_slice = curr_slice.previous_rest_per_voice[voice_number]
                prev_prev_rest_slice = None if prev_rest_slice is None else prev_rest_slice.previous_rest_per_voice[voice_number]

                if (prev_note is not None) and (next_note is not None) and (prev_prev_note is not None):
                    section_crossed = \
                        (curr_slice.bar in metadata['section_starts'] and prev_slice.bar in metadata['section_ends']) or \
                        (prev_slice.bar in metadata['section_starts'] and prev_prev_slice.bar in metadata['section_ends']) or \
                        (next_slice.bar in metadata['section_starts'] and curr_slice.bar in metadata['section_ends'])
                    
                    # Check if there is a rest in between any of the relevant notes.
                    rest_between_curr_next = next_rest_slice and next_rest_slice.original_line_num < next_slice.original_line_num
                    rest_between_curr_prev = prev_rest_slice and prev_rest_slice.original_line_num > prev_slice.original_line_num
                    rest_between_curr_prev_prev = prev_prev_rest_slice and ((prev_rest_slice.original_line_num > prev_prev_slice.original_line_num)\
                                                                            or (prev_prev_rest_slice.original_line_num > prev_prev_slice.original_line_num) )
                    
                    if section_crossed:
                        continue
                    elif rest_between_curr_next or rest_between_curr_prev or rest_between_curr_prev_prev:
                        continue
                    else:
                        # If we didnt cross a section or rest, check the leap.
                        interval_leap = abs(curr_note.midi_pitch - prev_note.midi_pitch)  # Main leap (prev -> curr)
                        interval1 = abs(prev_note.midi_pitch - prev_prev_note.midi_pitch)  # Fixed: preceding interval
                        interval2 = abs(curr_note.midi_pitch - next_note.midi_pitch)       # Following interval
                        
                        # Check if the main interval is a leap AND both approach/departure intervals are leaps
                        if interval_leap > 2 and interval1 > 2 and interval2 > 2:
                            # If the motion is ascending, first interval in time (interval1) should be descending and interval2 also descending.
                            # If the motion is descending, first interval in time (interval1) should be ascending and interval2 also ascending.
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
                                    original_line_num=curr_slice.original_line_num,
                                    beat=curr_slice.beat,
                                    bar=curr_slice.bar,
                                    voice_indices=voice_number,
                                    voice_names=metadata['voice_names'][voice_number],
                                    # Fixed: chronological order (earliest to latest)
                                    note_names=(prev_prev_note.note_name, prev_note.note_name, curr_note.note_name, next_note.note_name)
                                ))
                            
        return violations  

    @staticmethod
    def interval_order_motion(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        In leaps: in ascending motion, the larger intervals come first; in descending, the smaller first.  
        """
        rule_id = '25'

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs['slice_index']
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]

        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Short-circuit evaluation prevents NoneType errors
            if (curr_note.note_type == 'note' and 
                (prev_1st_slice := curr_slice.previous_any_note_type_per_voice[voice_number]) is not None and
                (prev_2nd_slice := prev_1st_slice.previous_note_per_voice[voice_number]) is not None and
                (prev_1st_note := prev_1st_slice.notes[voice_number]) is not None and  # If the slice exists, the note should never be none. This check is only donefor the short-circuit walrus assignment syntax.
                (prev_2nd_note := prev_2nd_slice.notes[voice_number]) is not None and
                prev_1st_note.note_type == 'note' and 
                prev_2nd_note.note_type == 'note'):

                # Check if we cross a section between either of the slices
                if CounterpointRules._chronological_slices_cross_section([prev_2nd_slice, prev_1st_slice, curr_slice], metadata):
                    continue
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
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=voice_number,
                                voice_names=metadata['voice_names'][voice_number],
                                note_names=(prev_2nd_note.note_name, prev_1st_note.note_name, curr_note.note_name)
                            ))

        return violations
    
    @staticmethod
    def successive_leap_opposite_direction(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Successive leaps in opposite directions should not be overused.
        This rule identifies two consecutive leaps (intervals > 2 semitones) that move in opposite directions.
        
        The violation is reported at the current slice (end of the second leap).
        Note order in violation: (prev_2nd_note, prev_1st_note, curr_note) - chronological order.
        """
        rule_id = '26'

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]

        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Use walrus operator to safely collect the three notes needed for two consecutive intervals
            if (curr_note.note_type == 'note' and 
                (prev_1st_slice := curr_slice.previous_any_note_type_per_voice[voice_number]) is not None and
                (prev_2nd_slice := prev_1st_slice.previous_note_per_voice[voice_number]) is not None and
                (prev_1st_note := prev_1st_slice.notes[voice_number]) is not None and
                (prev_2nd_note := prev_2nd_slice.notes[voice_number]) is not None and
                prev_1st_note.note_type == 'note' and
                prev_2nd_note.note_type == 'note'):
                
                # Check if we cross a section between any of the slices
                if CounterpointRules._chronological_slices_cross_section([prev_2nd_slice, prev_1st_slice, curr_slice], metadata):
                    continue
                
                # Check for rests between the notes (similar to other leap rules)
                prev_rest_after_2nd = prev_2nd_slice.next_rest_per_voice[voice_number]
                prev_rest_after_1st = prev_1st_slice.next_rest_per_voice[voice_number]
                
                # Check if there's a rest between prev_2nd_note and prev_1st_note
                rest_between_2nd_1st = (prev_rest_after_2nd and 
                                    prev_rest_after_2nd.original_line_num < prev_1st_slice.original_line_num)
                
                # Check if there's a rest between prev_1st_note and curr_note
                rest_between_1st_curr = (prev_rest_after_1st and 
                                    prev_rest_after_1st.original_line_num < curr_slice.original_line_num)
                
                # Skip if there are rests interrupting the note sequence
                if rest_between_2nd_1st or rest_between_1st_curr:
                    continue
                
                # Calculate the two consecutive intervals
                interval1 = abs(prev_1st_note.midi_pitch - prev_2nd_note.midi_pitch)  # First leap (earlier in time)
                interval2 = abs(curr_note.midi_pitch - prev_1st_note.midi_pitch)      # Second leap (later in time)
                
                # Check if both intervals are leaps (> 2 semitones)
                if interval1 > 2 and interval2 > 2:
                    # Determine the direction of each leap
                    first_leap_ascending = prev_1st_note.midi_pitch > prev_2nd_note.midi_pitch
                    second_leap_ascending = curr_note.midi_pitch > prev_1st_note.midi_pitch
                    
                    # Check if the leaps are in opposite directions
                    if first_leap_ascending != second_leap_ascending:
                        violations.append(RuleViolation(
                            rule_name=rulename,
                            rule_id=rule_id,
                            slice_index=curr_slice_idx,
                            original_line_num=curr_slice.original_line_num,
                            beat=curr_slice.beat,
                            bar=curr_slice.bar,
                            voice_indices=voice_number,
                            voice_names=metadata['voice_names'][voice_number],
                            # Notes in chronological order: earliest -> latest
                            note_names=(prev_2nd_note.note_name, prev_1st_note.note_name, curr_note.note_name)
                        ))

        return violations

    @staticmethod
    def leap_up_accented_long_note(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Leaps up to accented long notes [strong beat, half note or greater] are not allowed.
        The violation is reported at the current slice (the note that the leap goes to).
        """
        rule_id = '27'

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]

        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is a note (not rest) and get previous note
            if (curr_note and curr_note.note_type == 'note' and 
                (prev_slice := curr_slice.previous_note_per_voice[voice_number]) is not None):
                
                prev_note = prev_slice.notes[voice_number]
                
                if prev_note and prev_note.note_type == 'note':
                    # Check if we cross a section or have a rest between notes
                    section_crossed = (curr_slice.bar in metadata['section_starts'] and 
                                    prev_slice.bar in metadata['section_ends'])
                    
                    prev_rest_slice = curr_slice.previous_rest_per_voice[voice_number]
                    rest_between = (prev_rest_slice and 
                                prev_rest_slice.original_line_num > prev_slice.original_line_num)
                    
                    if section_crossed or rest_between:
                        continue
                    
                    # Check if there's an upward leap (> 2 semitones)
                    interval = curr_note.midi_pitch - prev_note.midi_pitch
                    if interval > 2:  # Upward leap
                        # Check if current note is "accented long"
                        # 1. Duration >= 0.5 (half note or greater)
                        # 2. Falls on a strong beat
                        if curr_note.duration >= 0.5:
                            # Check if current slice falls on a strong beat
                            time_sig = metadata['time_signatures'][curr_slice.bar][voice_number]
                            numerator = str(time_sig[0])
                            denominator = str(time_sig[1])
                            
                            if (numerator in TIME_SIGNATURE_STRONG_BEAT_MAP and 
                                denominator in TIME_SIGNATURE_STRONG_BEAT_MAP[numerator]):
                                
                                strong_beats = TIME_SIGNATURE_STRONG_BEAT_MAP[numerator][denominator]
                                
                                if curr_slice.beat in strong_beats:
                                    # This is a violation: upward leap to accented long note
                                    violations.append(RuleViolation(
                                        rule_name=rulename,
                                        rule_id=rule_id,
                                        slice_index=curr_slice_idx,
                                        original_line_num=curr_slice.original_line_num,
                                        beat=curr_slice.beat,
                                        bar=curr_slice.bar,
                                        voice_indices=voice_number,
                                        voice_names=metadata['voice_names'][voice_number],
                                        # Notes in chronological order: prev_note -> curr_note
                                        note_names=(prev_note.note_name, curr_note.note_name)
                                    ))
                            else:
                                raise ValueError(f"Time signature {time_sig} not found in TIME_SIGNATURE_STRONG_BEAT_MAP")

        return violations
    
    
    """ % Other aspects % """
    @staticmethod
    def eight_pair_stepwise(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Eight notes occurr as stepwise-related pairs, and on the weak part of the beat. 
        Either after a quarter note or a dotted half note.
        """
        rule_id = '28'
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is an eighth note
            if (curr_note and curr_note.note_type == 'note' and 
                curr_note.duration == DURATION_MAP['8'] and  # Eighth note duration
                (prev_slice := curr_slice.previous_note_per_voice[voice_number]) is not None):
                
                prev_note = prev_slice.notes[voice_number]
                
                if (prev_note and prev_note.note_type == 'note' and 
                    prev_note.duration == DURATION_MAP['8']):  # Previous note is also an eighth note
                    
                    # Check if we cross a section or have a rest between notes
                    section_crossed = (curr_slice.bar in metadata['section_starts'] and 
                                     prev_slice.bar in metadata['section_ends'])
                    
                    prev_rest_slice = curr_slice.previous_rest_per_voice[voice_number]
                    rest_between = (prev_rest_slice and 
                                  prev_rest_slice.original_line_num > prev_slice.original_line_num)
                    
                    if section_crossed or rest_between:
                        continue
                    
                    # Check if the two eighth notes are stepwise-related (interval <= 2 semitones)
                    interval = abs(curr_note.midi_pitch - prev_note.midi_pitch)
                    if interval <= 2:  # Stepwise motion
                        
                        # Check if previous note (the first eighth) is on a weak beat
                        time_sig = metadata['time_signatures'][prev_slice.bar][voice_number]
                        numerator = str(time_sig[0])
                        denominator = str(time_sig[1])
                        
                        if (numerator in TIME_SIGNATURE_STRONG_BEAT_MAP and 
                            denominator in TIME_SIGNATURE_STRONG_BEAT_MAP[numerator]):
                            
                            strong_beats = TIME_SIGNATURE_STRONG_BEAT_MAP[numerator][denominator]
                            
                            # First eight note should be on weak beat (not in strong_beats)
                            if prev_slice.beat not in strong_beats:
                                
                                # Check what comes before the eighth note pair
                                if (prev_prev_slice := prev_slice.previous_note_per_voice[voice_number]) is not None:
                                    prev_prev_note = prev_prev_slice.notes[voice_number]
                                    
                                    if (prev_prev_note and prev_prev_note.note_type == 'note'):
                                        
                                        # Check for section crossing or rest before the pair
                                        section_crossed_before = (prev_slice.bar in metadata['section_starts'] and 
                                                                prev_prev_slice.bar in metadata['section_ends'])
                                        
                                        prev_prev_rest_slice = prev_slice.previous_rest_per_voice[voice_number]
                                        rest_before_pair = (prev_prev_rest_slice and 
                                                          prev_prev_rest_slice.original_line_num > prev_prev_slice.original_line_num)
                                        
                                        if not (section_crossed_before or rest_before_pair):
                                            ''' Calculate effective duration of preceding note, to detect also detect half note tied to quarter as a dotted quarter.  '''
                                            effective_duration = prev_prev_note.duration # Start with the duration of the immediately preceding note

                                            # If the preceding note is tied, follow the tie chain backwards
                                            if prev_prev_note.is_tied:
                                                # Walk backwards through the tie chain to find the beginning
                                                current_tie_slice = prev_prev_slice
                                                current_tie_note = prev_prev_note
                                                while (current_tie_note.is_tied and
                                                       (tie_prev_slice := current_tie_slice.previous_note_per_voice[voice_number]) is not None):
                                                    tie_prev_note = tie_prev_slice.notes[voice_number]
                                                    
                                                    if (tie_prev_note and tie_prev_note.note_type == 'note' and
                                                        tie_prev_note.is_new_occurrence):
                                                        # Found the beginning of the tie - add its duration
                                                        effective_duration += tie_prev_note.duration
                                                        break
                                                    elif (tie_prev_note and tie_prev_note.note_type == 'note'):
                                                        # This is a continuation of the tie - add its duration
                                                        effective_duration += tie_prev_note.duration
                                                        current_tie_slice = tie_prev_slice
                                                        current_tie_note = tie_prev_note
                                                    else:
                                                        # Encountered something unexpected in the tie chain
                                                        break

                                            # Check if effective duration is quarter note (0.25) or dotted half note (0.75)
                                            allowed_durations = [DURATION_MAP['4'], DURATION_MAP['2']*1.5]  # 0.25, 0.75

                                            if effective_duration not in allowed_durations:
                                                # This is a violation: eighth note pair not after quarter or dotted half
                                                violations.append(RuleViolation(
                                                    rule_name=rulename,
                                                    rule_id=rule_id,
                                                    slice_index=curr_slice_idx,
                                                    original_line_num=curr_slice.original_line_num,
                                                    beat=curr_slice.beat,
                                                    bar=curr_slice.bar,
                                                    voice_indices=voice_number,
                                                    voice_names=metadata['voice_names'][voice_number],
                                                    # Notes in chronological order: prev_prev -> prev -> curr
                                                    note_names=(prev_prev_note.note_name, prev_note.note_name, curr_note.note_name)
                                                ))
                        else:
                            raise ValueError(f"Time signature {time_sig} not found in TIME_SIGNATURE_STRONG_BEAT_MAP")

        return violations

    def ascending_leap_to_from_quarter(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Ascending leaps from accenteted quarter notes are very rare, as are ascending leaps [to a quarter] from dotted half notes.
        """
        rule_id = 35
        violations = []
        raise NotImplementedError(f"Rule {rulename} with ID {rule_id} is not implemented yet.")

        return violations



    """ %% Quarter note idioms %% """
    @staticmethod
    def leap_in_quarters_balanced(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """
        Leaps [in quarter notes] are balanced by [stepwise or leaping] motion in the opposite direction.
        """
        rule_id = '36'
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is a quarter note and get previous and next notes
            if (curr_note and curr_note.note_type == 'note' and 
                curr_note.duration == DURATION_MAP['4'] and  # Quarter note duration
                (prev_slice := curr_slice.previous_note_per_voice[voice_number]) is not None and
                (next_slice := curr_slice.next_note_per_voice[voice_number]) is not None):
                
                prev_note = prev_slice.notes[voice_number]
                next_note = next_slice.notes[voice_number]
                
                if (prev_note and prev_note.note_type == 'note' and
                    next_note and next_note.note_type == 'note'):
                    
                    # Check if we cross sections or have rests between notes
                    section_crossed_before = (curr_slice.bar in metadata['section_starts'] and 
                                            prev_slice.bar in metadata['section_ends'])
                    section_crossed_after = (next_slice.bar in metadata['section_starts'] and 
                                           curr_slice.bar in metadata['section_ends'])
                    
                    prev_rest_slice = curr_slice.previous_rest_per_voice[voice_number]
                    next_rest_slice = curr_slice.next_rest_per_voice[voice_number]
                    rest_before = (prev_rest_slice and 
                                 prev_rest_slice.original_line_num > prev_slice.original_line_num)
                    rest_after = (next_rest_slice and 
                                next_rest_slice.original_line_num < next_slice.original_line_num)
                    
                    if section_crossed_before or section_crossed_after or rest_before or rest_after:
                        continue
                    
                    # Check if there's a leap from previous note to current note (> 2 semitones)
                    leap_interval = abs(curr_note.midi_pitch - prev_note.midi_pitch)
                    if leap_interval > 2:  # This is a leap
                        
                        # Determine direction of the leap
                        leap_ascending = curr_note.midi_pitch > prev_note.midi_pitch
                        
                        # Check if there's motion after the leap and if it's in the opposite direction
                        following_interval = next_note.midi_pitch - curr_note.midi_pitch
                        
                        # Violation if there's no motion OR motion is in the same direction as the leap
                        violation_occurred = False
                        
                        if following_interval == 0:
                            # No motion after the leap - this doesn't balance the leap
                            violation_occurred = True
                        elif leap_ascending and following_interval > 0:
                            # Leap was ascending, following motion is also ascending - violation
                            violation_occurred = True
                        elif not leap_ascending and following_interval < 0:
                            # Leap was descending, following motion is also descending - violation
                            violation_occurred = True
                        
                        if violation_occurred:
                            # Record violation at the next note (the note that fails to balance)
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_idx + 1,  # Next slice index
                                original_line_num=next_slice.original_line_num,
                                beat=next_slice.beat,
                                bar=next_slice.bar,
                                voice_indices=voice_number,
                                voice_names=metadata['voice_names'][voice_number],
                                # Notes in chronological order: prev -> curr (leap) -> next (balancing motion)
                                note_names=(prev_note.note_name, curr_note.note_name, next_note.note_name)
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


    """ %%% Normalization functions %%% """
    @staticmethod
    def norm_count_tie_ends(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """ Detect tie ends in the current slice. """
        rule_id = '1002'

        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is a tie end
            if curr_note and curr_note.note_type == 'note' and curr_note.is_tie_end:
                violations.append(RuleViolation(
                    rule_name=rulename,
                    rule_id=rule_id,
                    slice_index=curr_slice_idx,
                    original_line_num=curr_slice.original_line_num,
                    beat=curr_slice.beat,
                    bar=curr_slice.bar,
                    voice_indices=voice_number,
                    voice_names=metadata['voice_names'][voice_number],
                    note_names=curr_note.note_name
                ))

        return violations
    
    @staticmethod
    def norm_count_tie_starts(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """ Detect tie ends in the current slice. """
        rule_id = '1001'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is a tie end
            if curr_note and curr_note.note_type == 'note' and curr_note.is_tie_start:
                violations.append(RuleViolation(
                    rule_name=rulename,
                    rule_id=rule_id,
                    slice_index=curr_slice_idx,
                    original_line_num=curr_slice.original_line_num,
                    beat=curr_slice.beat,
                    bar=curr_slice.bar,
                    voice_indices=voice_number,
                    voice_names=metadata['voice_names'][voice_number],
                    note_names=curr_note.note_name
                ))

        return violations


    @staticmethod
    def norm_label_chord_name_m21(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """ Labels m21 chord name """
        rule_id = '1003'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        # Get chord analysis
        chord_analysis = curr_slice.chord_analysis
        if chord_analysis is None:
            # In this case, there should be all rests in the slice. Check for this, and else raise an error.
            if not all(note.note_type == 'rest' for note in curr_slice.notes):
                raise ValueError(f"Slice {curr_slice_idx} does not have chord analysis, but contains notes: {curr_slice.notes}")
            else:
                return violations  # No chord analysis, no violations (all rests).
        else:
            # Count sounding notes (not rests)
            sounding_notes = [note for note in curr_slice.notes 
                            if note and note.note_type == 'note' and note.midi_pitch != -1]
            
            # Only create a "violation" (label) if there are sounding notes
            if len(sounding_notes) > 0:
                # Get the chord name from music21 analysis
                chord_name = chord_analysis.get('common_name_m21', 'unknown')
                
                # Create a "violation" that serves as a label for the chord name
                violations.append(RuleViolation(
                    rule_name=rulename,
                    rule_id=rule_id,
                    slice_index=curr_slice_idx,
                    original_line_num=curr_slice.original_line_num,
                    beat=curr_slice.beat,
                    bar=curr_slice.bar,
                    voice_indices=0,  # Top voice (following the pattern from non_root_1st_inv_maj)
                    voice_names=metadata['voice_names'][0],
                    note_names=(chord_name, chord_analysis['root_note_voice'])  # Store the chord name in note_names field
                ))

        return violations


    @staticmethod
    def norm_ties_contained_in_bar(rulename, **kwargs) -> Dict[str, List[RuleViolation]]:
        """ Count ties that fall within a bar and do not cross a barline. """
        rule_id = '1004'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is a tie start
            if curr_note and curr_note.note_type == 'note' and curr_note.is_tie_start:
                
                # Find the end of this tie by following the tie chain forward
                current_slice = curr_slice
                tie_crosses_barline = False
                tie_end_found = False
                
                # Walk forward through slices to find the tie end
                while current_slice is not None:
                    current_note = current_slice.notes[voice_number]
                    
                    # Check if we've moved to a different bar
                    if current_slice.bar != curr_slice.bar:
                        tie_crosses_barline = True
                        break
                    
                    # Check if this note is the end of the tie
                    if current_note and current_note.note_type == 'note' and current_note.is_tie_end:
                        tie_end_found = True
                        break
                    
                    # Move to the next slice for this voice
                    current_slice = current_slice.next_note_per_voice[voice_number]
                
                # Only count ties that are contained within the same bar
                if tie_end_found and not tie_crosses_barline:
                    violations.append(RuleViolation(
                        rule_name=rulename,
                        rule_id=rule_id,
                        slice_index=curr_slice_idx,
                        original_line_num=curr_slice.original_line_num,
                        beat=curr_slice.beat,
                        bar=curr_slice.bar,
                        voice_indices=voice_number,
                        voice_names=metadata['voice_names'][voice_number],
                        note_names=curr_note.note_name
                    ))

        return violations


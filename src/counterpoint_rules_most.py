
from counterpoint_rules_base import CounterpointRulesBase, RuleViolation
from Note import SalamiSlice

from music21 import pitch as m21_pitch
from music21 import interval as m21_interval

from constants import DURATION_MAP, TIME_SIGNATURE_STRONG_BEAT_MAP

class CounterpointRulesMost(CounterpointRulesBase):
    """
    %%% Rules for any amount of voices %%%
    """

    """ %% Rhytm and meter %% """


    """ % Note values % """
    @staticmethod
    def longa_only_at_endings(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
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
    def brevis_at_begin_end(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
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
    def tie_into_strong_beat(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """
        Check if the current note is a tie-end and falls on a strong beat.
        """
        rule_id = '14'
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
    def tie_into_weak_beat(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """
        Check if the current note is a tie-end and falls on a strong beat.
        """
        rule_id = '15'
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
    def non_root_1st_inv_maj(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
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
    def leap_too_large(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
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
            # First check if this is a note, and if it has a previous slice with a new occurrence (e.g. is not one of the first slices in the piece)
            if ((curr_note.note_type == 'note') and 
                (curr_note.is_new_occurrence) and 
                (prev_slice is not None)):

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
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=voice_number,
                                voice_names=metadata['voice_names'][voice_number],
                                # Notes in chronological order: prev_note -> curr_note
                                note_names=(prev_note.note_name, curr_note.note_name)
                            ))


        return violations
    
    @staticmethod
    def leap_approach_left_opposite(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
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
            
            if (curr_note.note_type == 'note') and (curr_note.is_new_occurrence) and (prev_slice is not None) \
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
    def interval_order_motion(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """
        In two consecutive leaps in the same direction: 
        - Ascending motion: larger interval comes first 
        - Descending motion: smaller interval comes first
        """
        rule_id = '25'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs['slice_index']
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            if not (curr_note.note_type == 'note' and curr_note.is_new_occurrence):
                continue

            # Get two previous note slices (both must be actual notes)
            prev_1st_slice = curr_slice.previous_note_per_voice[voice_number]
            if prev_1st_slice is None:
                continue
            
            prev_2nd_slice = prev_1st_slice.previous_note_per_voice[voice_number]
            if prev_2nd_slice is None:
                continue

            prev_1st_note = prev_1st_slice.notes[voice_number]
            prev_2nd_note = prev_2nd_slice.notes[voice_number]
            
            if (prev_1st_note.note_type != 'note' or prev_2nd_note.note_type != 'note'):
                continue

            # Check for section crossings or rests
            if CounterpointRulesBase._chronological_slices_cross_section(
                [prev_2nd_slice, prev_1st_slice, curr_slice], metadata):
                continue

            # Calculate intervals (rename for clarity)
            later_leap = abs(curr_note.midi_pitch - prev_1st_note.midi_pitch)     # More recent leap
            earlier_leap = abs(prev_1st_note.midi_pitch - prev_2nd_note.midi_pitch)  # Earlier leap

            # Both must be leaps
            if later_leap <= 2 or earlier_leap <= 2:
                continue

            # Check if motion is in same direction
            ascending = (prev_2nd_note.midi_pitch < prev_1st_note.midi_pitch < curr_note.midi_pitch)
            descending = (prev_2nd_note.midi_pitch > prev_1st_note.midi_pitch > curr_note.midi_pitch)
            
            if not (ascending or descending):
                continue  # Not same direction

            # Apply the rule:
            # Ascending: earlier leap should be >= later leap
            # Descending: earlier leap should be <= later leap  
            violation = False
            if ascending and earlier_leap < later_leap:
                violation = True
            elif descending and earlier_leap > later_leap:
                violation = True

            if violation:
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
    def successive_leap_opposite_direction(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
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
            if (curr_note.note_type == 'note' and curr_note.is_new_occurrence and
                (prev_1st_slice := curr_slice.previous_any_note_type_per_voice[voice_number]) is not None and
                (prev_2nd_slice := prev_1st_slice.previous_note_per_voice[voice_number]) is not None and
                (prev_1st_note := prev_1st_slice.notes[voice_number]) is not None and
                (prev_2nd_note := prev_2nd_slice.notes[voice_number]) is not None and
                prev_1st_note.note_type == 'note' and
                prev_2nd_note.note_type == 'note'):
                
                # Check if we cross a section between any of the slices
                if CounterpointRulesBase._chronological_slices_cross_section([prev_2nd_slice, prev_1st_slice, curr_slice], metadata):
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
    def leap_up_accented_long_note(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
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
                curr_note.is_new_occurrence and
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
    def eight_pair_stepwise(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
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

    @staticmethod
    def leading_tone_approach_step(rulename, **kwargs) -> list[RuleViolation]:
        """
        The leading tone is approached by step. Leading tone: Major third or sixth, resolves up to any note by half step either directly or on the next whole beat.
        Root note of the chord it resolves to is of different pitch class than the pitch class of the chord root it was the third/sixth of, and the note it resolves to is also 
        of the same pitch class as the root or the fifth of the resolve-to chord.
        """
        rule_id = '32'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []
        
        for voice_number, curr_note in enumerate(curr_slice.notes):
            if (curr_note and curr_note.note_type == 'note' and 
                curr_note.is_new_occurrence and
                (next_slice := curr_slice.next_note_per_voice[voice_number]) is not None):
                
                next_note = next_slice.notes[voice_number]
                
                if next_note and next_note.note_type == 'note' and next_note.is_new_occurrence:
                    # Check if we cross a section or have a rest between notes
                    section_crossed = (next_slice.bar in metadata['section_starts'] and 
                                     curr_slice.bar in metadata['section_ends'])
                    
                    next_rest_slice = curr_slice.next_rest_per_voice[voice_number]
                    rest_between = (next_rest_slice and 
                                  next_rest_slice.original_line_num < next_slice.original_line_num)
                    
                    if section_crossed or rest_between:
                        continue
                    
                    # Get chord analysis
                    curr_chord_analysis = curr_slice.chord_analysis
                    next_chord_analysis = next_slice.chord_analysis
                    if curr_chord_analysis is None or next_chord_analysis is None:
                        continue
                    
                    # Check if current note is a MAJOR third or sixth relative to the chord root
                    curr_reduced_interval = curr_note.reduced_interval_to_root
                    if curr_reduced_interval not in [4, 9]:  # Major third (4 semitones) or major sixth (9 semitones)
                        continue
                    
                    # Check if the note resolves upward by a half step (1 semitone)
                    interval = next_note.midi_pitch - curr_note.midi_pitch
                    if interval != 1:  # Must be exactly 1 semitone upward
                        continue

                    # Get the root note voices (these are voice indices, not note names)
                    curr_root_voice = curr_chord_analysis.get('root_note_voice')
                    next_root_voice = next_chord_analysis.get('root_note_voice')
                    
                    if curr_root_voice is None or next_root_voice is None:
                        continue
                    
                    # Get the actual Note objects using the root note voices
                    curr_root_note = curr_slice.notes[curr_root_voice]
                    next_root_note = next_slice.notes[next_root_voice]
                    
                    if (curr_root_note is None or next_root_note is None or
                        curr_root_note.note_type != 'note' or next_root_note.note_type != 'note'):
                        continue
                    
                    # Check if the root notes have different pitch classes using MIDI pitch modulo 12
                    curr_root_pc = curr_root_note.midi_pitch % 12
                    next_root_pc = next_root_note.midi_pitch % 12
                    
                    # Check if the root notes have different pitch classes
                    if abs(curr_root_pc - next_root_pc) % 12 == 0:
                        continue  # Same pitch class, so this is not a leading tone resolution
                    
                    # NEW: Check if the note resolves to the root or fifth of the destination chord
                    resolution_note_pc = next_note.midi_pitch % 12
                    destination_root_pc = next_root_pc
                    destination_fifth_pc = (next_root_pc + 7) % 12  # Perfect fifth is 7 semitones above root
                    
                    # The resolution note must be either the root or the fifth of the destination chord
                    if resolution_note_pc != destination_root_pc and resolution_note_pc != destination_fifth_pc:
                        continue  # Resolution note is not root or fifth, so this is not a proper leading tone resolution
                    
                    # If we've reached here, we have a potential leading tone that violates the rule
                    # The violation is that the leading tone (major third/sixth) was NOT approached by step
                    
                    # Check if the current note was approached by step from the previous note
                    if (prev_slice := curr_slice.previous_note_per_voice[voice_number]) is not None:
                        prev_note = prev_slice.notes[voice_number]
                        
                        if prev_note and prev_note.note_type == 'note':
                            # Check for section crossing or rest before current note
                            section_crossed_before = (curr_slice.bar in metadata['section_starts'] and 
                                                    prev_slice.bar in metadata['section_ends'])
                            
                            prev_rest_slice = curr_slice.previous_rest_per_voice[voice_number]
                            rest_before = (prev_rest_slice and 
                                         prev_rest_slice.original_line_num > prev_slice.original_line_num)
                            
                            if not (section_crossed_before or rest_before):
                                # Check if approached by step (1-2 semitones)
                                approach_interval = abs(curr_note.midi_pitch - prev_note.midi_pitch)
                                if approach_interval <= 2:  # Approached by step
                                    continue  # No violation - correctly approached by step
                            
                            # If we reach here, the leading tone was either:
                            # 1. Not approached by step (leap approach)
                            # 2. Approached after a section break or rest
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_idx,
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=voice_number,
                                voice_names=metadata['voice_names'][voice_number],
                                note_names=(prev_note.note_name if prev_note else "unknown", 
                                        curr_note.note_name, next_note.note_name)
                            ))
                else:
                    # No previous note available - leading tone at the beginning
                    violations.append(RuleViolation(
                        rule_name=rulename,
                        rule_id=rule_id,
                        slice_index=curr_slice_idx,
                        original_line_num=curr_slice.original_line_num,
                        beat=curr_slice.beat,
                        bar=curr_slice.bar,
                        voice_indices=voice_number,
                        voice_names=metadata['voice_names'][voice_number],
                        note_names=("start", curr_note.note_name, next_note.note_name)
                    ))

        return violations

    @staticmethod
    def ascending_leap_strong_quarter(rulename, **kwargs) -> list[RuleViolation]:
        """
        Ascending leaps from accented [meaning: on a strong beat] quarter notes are very rare, 
        as are ascending leaps [to a quarter] from dotted half notes.
        """ 
        rule_id = "34"
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        # TODO: read through this/test it

        for voice_number, curr_note in enumerate(curr_slice.notes):
            if (curr_note and curr_note.note_type == 'note' and 
                (next_slice := curr_slice.next_note_per_voice[voice_number]) is not None):
                
                next_note = next_slice.notes[voice_number]
                
                if next_note and next_note.note_type == 'note':
                    # Check if we cross a section or have a rest between notes
                    section_crossed = (next_slice.bar in metadata['section_starts'] and 
                                    curr_slice.bar in metadata['section_ends'])
                    
                    next_rest_slice = curr_slice.next_rest_per_voice[voice_number]
                    rest_between = (next_rest_slice and 
                                next_rest_slice.original_line_num < next_slice.original_line_num)
                    
                    if section_crossed or rest_between:
                        continue
                    
                    # Check if there's an ascending leap (> 2 semitones upward)
                    interval = next_note.midi_pitch - curr_note.midi_pitch
                    if interval > 2:  # Ascending leap
                        
                        # Flags to track which violation condition is met
                        leap_from_strong_quarter = False
                        leap_to_quarter_from_dotted_half = False
                        
                        # Scenario 1: Ascending leap FROM quarter note on strong beat
                        if curr_note.duration == DURATION_MAP['4']:  # Current note is quarter note
                            # Check if current note is on a strong beat (accented)
                            time_sig = metadata['time_signatures'][curr_slice.bar][voice_number]
                            numerator = str(time_sig[0])
                            denominator = str(time_sig[1])

                            # Check if the beat is a strong beat
                            if curr_slice.beat in TIME_SIGNATURE_STRONG_BEAT_MAP[numerator][denominator]:
                                    leap_from_strong_quarter = True

                        # Scenario 2: Ascending leap TO quarter note FROM dotted half note
                        elif next_note.duration == DURATION_MAP['4']:  # Next note is quarter note
                            # Check if current note is a dotted half note (inline detection)
                            dotted_half_duration = DURATION_MAP['2'] * 1.5  # 0.5 * 1.5 = 0.75
                            
                            # Method 1: Check if it's a half note that's dotted
                            is_dotted_half = False
                            if (curr_note.duration == DURATION_MAP['2'] and curr_note.is_dotted):
                                is_dotted_half = True
                            # Method 2: Check if duration matches dotted half (handles edge cases)
                            elif abs(curr_note.duration - dotted_half_duration) < 1e-6:
                                is_dotted_half = True
                            
                            if is_dotted_half:
                                leap_to_quarter_from_dotted_half = True
                        
                        # Add violation if either condition is met
                        if leap_from_strong_quarter or leap_to_quarter_from_dotted_half:
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_idx,
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=voice_number,
                                voice_names=metadata['voice_names'][voice_number],
                                note_names=(curr_note.note_name, next_note.note_name)
                            ))

        return violations

    """ %% Quarter note idioms %% """
    @staticmethod
    def ascending_leap_to_from_quarter(rulename, **kwargs) -> list[RuleViolation]:
        """
        Ascending leaps from accented quarter notes are very rare, as are ascending leaps [to a quarter] from dotted half notes.
        """
        rule_id = '35'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
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
                    
                    # Check if there's an ascending leap (> 2 semitones upward)
                    interval = curr_note.midi_pitch - prev_note.midi_pitch
                    if interval > 2:  # Ascending leap
                        
                        # Flags to track which violation condition is met
                        leap_from_accented_quarter = False
                        leap_to_quarter_from_dotted_half = False
                        
                        # Scenario 1: Ascending leap FROM accented quarter note
                        if prev_note.duration == DURATION_MAP['4']:  # Previous note is quarter note
                            # Check if previous note is on a strong beat (accented)
                            time_sig = metadata['time_signatures'][prev_slice.bar][voice_number]
                            numerator = str(time_sig[0])
                            denominator = str(time_sig[1])
                            
                            if (numerator in TIME_SIGNATURE_STRONG_BEAT_MAP and 
                                denominator in TIME_SIGNATURE_STRONG_BEAT_MAP[numerator]):
                                
                                strong_beats = TIME_SIGNATURE_STRONG_BEAT_MAP[numerator][denominator]
                                
                                if prev_slice.beat in strong_beats:
                                    leap_from_accented_quarter = True
                            else:
                                raise ValueError(f"Time signature {time_sig} not found in TIME_SIGNATURE_STRONG_BEAT_MAP")
                        
                        # Scenario 2: Ascending leap TO quarter note FROM dotted half note
                        elif curr_note.duration == DURATION_MAP['4']:  # Current note is quarter note
                            # Check if previous note is a dotted half note (inline detection)
                            dotted_half_duration = DURATION_MAP['2'] * 1.5  # 0.5 * 1.5 = 0.75
                            
                            # Method 1: Check if it's a half note that's dotted
                            is_dotted_half = False
                            if (prev_note.duration == DURATION_MAP['2'] and prev_note.is_dotted):
                                is_dotted_half = True
                            # Method 2: Check if duration matches dotted half (handles edge cases)
                            elif abs(prev_note.duration - dotted_half_duration) < 1e-6:
                                is_dotted_half = True
                            
                            if is_dotted_half:
                                leap_to_quarter_from_dotted_half = True
                        
                        # Add violation if either condition is met
                        if leap_from_accented_quarter or leap_to_quarter_from_dotted_half:
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_idx,
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=voice_number,
                                voice_names=metadata['voice_names'][voice_number],
                                note_names=(prev_note.note_name, curr_note.note_name)
                            ))

        return violations
    
    @staticmethod
    def leap_in_quarters_balanced(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
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
    

    




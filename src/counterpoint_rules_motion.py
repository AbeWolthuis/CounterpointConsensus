from counterpoint_rules_base import CounterpointRulesBase, RuleViolation
from Note import SalamiSlice

from music21 import pitch as m21_pitch
from music21 import interval as m21_interval

from constants import DURATION_MAP, TIME_SIGNATURE_STRONG_BEAT_MAP

class CounterpointRulesMotion(CounterpointRulesBase):
    """ %% Technical details %%   """

    """ % Motion relationships %   """
    @staticmethod
    def contrary_motion(rulename, **kwargs) -> list[RuleViolation]:
        """
        Count contrary motion.
        Contrary: there is movement between the voices move in the opposite direction. So, if two voices of the move up and down, movement is contrary (regardless of what other voices do).
        """
        rule_id = '41'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_i, note_i in enumerate(curr_slice.notes):
            if (note_i and note_i.note_type == 'note' and note_i.is_new_occurrence and
                (prev_slice_i := curr_slice.previous_note_per_voice[voice_i]) is not None and
                (prev_note_i := prev_slice_i.notes[voice_i]) and prev_note_i.note_type == 'note'):
                
                if CounterpointRulesBase._chronological_slices_cross_section([prev_slice_i, curr_slice], metadata):
                    continue

                motion_i = note_i.midi_pitch - prev_note_i.midi_pitch
                if motion_i == 0:
                    continue

                for voice_j, note_j in enumerate(curr_slice.notes):
                    # voice_j > voice_i to avoid checking the same pair twice
                    if (voice_j > voice_i and voice_j != voice_i and note_j and note_j.note_type == 'note' and note_j.is_new_occurrence and
                        (prev_slice_j := curr_slice.previous_note_per_voice[voice_j]) is not None and
                        (prev_note_j := prev_slice_j.notes[voice_j]) and prev_note_j.note_type == 'note'):
                        
                        if CounterpointRulesBase._chronological_slices_cross_section([prev_slice_j, curr_slice], metadata):
                            continue

                        motion_j = note_j.midi_pitch - prev_note_j.midi_pitch
                        if motion_j == 0:
                            continue

                        if (motion_i > 0 and motion_j < 0) or (motion_i < 0 and motion_j > 0):
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_idx,
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=(voice_i, voice_j),
                                voice_names=(metadata['voice_names'][voice_i], metadata['voice_names'][voice_j]),
                                note_names=(note_i.note_name, note_j.note_name)
                            ))

        return violations

    @staticmethod
    def oblique_motion(rulename, **kwargs) -> list[RuleViolation]:
        """
        Count oblique motion.
        Oblique motion: one note remains stationary (long/tied note, or note repeated), one of the other voices move.
        Only count once per moving voice occurrence.
        """
        rule_id = '42'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        # Check each voice to see if it's moving
        for voice_i, note_i in enumerate(curr_slice.notes):
            if (note_i and note_i.note_type == 'note' and note_i.is_new_occurrence and
                (prev_slice_i := curr_slice.previous_note_per_voice[voice_i]) is not None and
                (prev_note_i := prev_slice_i.notes[voice_i]) and prev_note_i.note_type == 'note'):
                
                if CounterpointRulesBase._chronological_slices_cross_section([prev_slice_i, curr_slice], metadata):
                    continue

                # Check if this voice is moving
                motion_i = note_i.midi_pitch - prev_note_i.midi_pitch
                if motion_i == 0:
                    continue  # This voice is not moving
                
                # This voice is moving - check if any other voice is stationary
                has_stationary_voice = False
                
                for voice_j, note_j in enumerate(curr_slice.notes):
                    if (voice_j != voice_i and note_j and note_j.note_type == 'note' and
                        (prev_slice_j := curr_slice.previous_note_per_voice[voice_j]) is not None and
                        (prev_note_j := prev_slice_j.notes[voice_j]) and prev_note_j.note_type == 'note'):
                        
                        if CounterpointRulesBase._chronological_slices_cross_section([prev_slice_j, curr_slice], metadata):
                            continue

                        # Check if voice_j is stationary
                        motion_j = note_j.midi_pitch - prev_note_j.midi_pitch
                        if motion_j == 0:
                            has_stationary_voice = True
                            break  # Found at least one stationary voice
                
                # If we found a stationary voice, record oblique motion for the moving voice
                if has_stationary_voice:
                    violations.append(RuleViolation(
                        rule_name=rulename,
                        rule_id=rule_id,
                        slice_index=curr_slice_idx,
                        original_line_num=curr_slice.original_line_num,
                        beat=curr_slice.beat,
                        bar=curr_slice.bar,
                        voice_indices=voice_i,  # Only the moving voice
                        voice_names=metadata['voice_names'][voice_i],
                        note_names=note_i.note_name
                    ))

        return violations

    @staticmethod
    def parallel_motion(rulename, **kwargs) -> list[RuleViolation]:
        """
        Count parallel motion.
        Parallel motion: two voices maintain the same interval class while moving in parallel.
        """
        rule_id = '43'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []


        for voice_i, note_i in enumerate(curr_slice.notes):
            if (note_i and note_i.note_type == 'note' and note_i.is_new_occurrence and
                (prev_slice_i := curr_slice.previous_note_per_voice[voice_i]) is not None and
                (prev_note_i := prev_slice_i.notes[voice_i]) and prev_note_i.note_type == 'note'):
                
                if CounterpointRulesBase._chronological_slices_cross_section([prev_slice_i, curr_slice], metadata):
                    continue

                motion_i = note_i.midi_pitch - prev_note_i.midi_pitch

                for voice_j, note_j in enumerate(curr_slice.notes):
                    if (voice_j > voice_i and note_j and note_j.note_type == 'note' and note_j.is_new_occurrence and
                        (prev_slice_j := curr_slice.previous_note_per_voice[voice_j]) is not None and
                        (prev_note_j := prev_slice_j.notes[voice_j]) and prev_note_j.note_type == 'note'):
                        
                        if CounterpointRulesBase._chronological_slices_cross_section([prev_slice_j, curr_slice], metadata):
                            continue

                        motion_j = note_j.midi_pitch - prev_note_j.midi_pitch
                        
                        # Efficiency check: both voices must move in same direction or both stationary
                        same_direction = (
                            (motion_i > 0 and motion_j > 0) or  # Both ascending
                            (motion_i < 0 and motion_j < 0) or  # Both descending
                            (motion_i == 0 and motion_j == 0)   # Both stationary
                        )
                        
                        if not same_direction:
                            continue  # Not parallel motion
                                                
                        # Get the most recent previous slice between the two voices
                        if prev_slice_i.original_line_num > prev_slice_j.original_line_num:
                            reference_prev_slice = prev_slice_i
                            prev_note_i_ref = prev_note_i
                            prev_note_j_ref = reference_prev_slice.notes[voice_j]
                        else:
                            reference_prev_slice = prev_slice_j
                            prev_note_i_ref = reference_prev_slice.notes[voice_i]
                            prev_note_j_ref = prev_note_j
                        
                        # Skip if we don't have both notes in the reference slice
                        if (not prev_note_i_ref or not prev_note_j_ref or 
                            prev_note_i_ref.note_type != 'note' or prev_note_j_ref.note_type != 'note'):
                            continue
                        
                        # Calculate interval classes using music21
                        current_interval_class = m21_interval.Interval(m21_pitch.Pitch(note_i.note_name), m21_pitch.Pitch(note_j.note_name))
                        previous_interval_class = m21_interval.Interval(m21_pitch.Pitch(prev_note_i_ref.note_name), m21_pitch.Pitch(prev_note_j_ref.note_name))
                        
                        if current_interval_class is None or previous_interval_class is None:
                            continue
                        
                        # Parallel motion: same interval class maintained
                        if current_interval_class == previous_interval_class:
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_idx,
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=(voice_i, voice_j),
                                voice_names=(metadata['voice_names'][voice_i], metadata['voice_names'][voice_j]),
                                note_names=(note_i.note_name, note_j.note_name)
                            ))

        return violations

    @staticmethod
    def similar_motion(rulename, **kwargs) -> list[RuleViolation]:
        """
        Count similar motion.
        Similar motion: voices move in the same direction, by any distance.
        """
        rule_id = '44'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_i, note_i in enumerate(curr_slice.notes):
            if (note_i and note_i.note_type == 'note' and note_i.is_new_occurrence and
                (prev_slice_i := curr_slice.previous_note_per_voice[voice_i]) is not None and
                (prev_note_i := prev_slice_i.notes[voice_i]) and prev_note_i.note_type == 'note'):
                
                if CounterpointRulesBase._chronological_slices_cross_section([prev_slice_i, curr_slice], metadata):
                    continue

                motion_i = note_i.midi_pitch - prev_note_i.midi_pitch
                if motion_i == 0:
                    continue

                for voice_j, note_j in enumerate(curr_slice.notes):
                    if (voice_j > voice_i and note_j and note_j.note_type == 'note' and note_j.is_new_occurrence and
                        (prev_slice_j := curr_slice.previous_note_per_voice[voice_j]) is not None and
                        (prev_note_j := prev_slice_j.notes[voice_j]) and prev_note_j.note_type == 'note'):
                        
                        if CounterpointRulesBase._chronological_slices_cross_section([prev_slice_j, curr_slice], metadata):
                            continue

                        motion_j = note_j.midi_pitch - prev_note_j.midi_pitch
                        if motion_j == 0:
                            continue

                        # Similar motion: same direction (any distance, but not the same because that is parallel_
                        if ((motion_i > 0 and motion_j > 0) or (motion_i < 0 and motion_j < 0)) and not (motion_i == motion_j):
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_idx,
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=(voice_i, voice_j),
                                voice_names=(metadata['voice_names'][voice_i], metadata['voice_names'][voice_j]),
                                note_names=(note_i.note_name, note_j.note_name)
                            ))

        return violations

    '''

    @staticmethod
    def ___parallel_fifth_octave(rulename, **kwargs) -> list[RuleViolation]:
        """
        Parallel fifths [or 5th+8th, see ex 52] or octaves are not found on successive [whole] beats, 
        or successive strong beats, even if there are consonant notes in between.
        """
        rule_id = '48'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        # Only check slices that fall on whole beats
        if not CounterpointRulesMotion._is_whole_beat(curr_slice):
            return violations

        for voice_i, note_i in enumerate(curr_slice.notes):
            if not (note_i and note_i.note_type == 'note'):
                continue

            for voice_j, note_j in enumerate(curr_slice.notes):
                # Don't compare voice i with itself, and check each voice pair only once
                if voice_j <= voice_i:
                    continue

                # Either note i or note j must be a new occurrence
                if not (note_j and note_j.note_type == 'note' and 
                       (note_i.is_new_occurrence or note_j.is_new_occurrence)):
                    continue

                # Find the most recent previous whole beat slice that contains both voices
                comparison_slice = CounterpointRulesMotion._find_most_recent_previous_whole_beat_slice_for_voices_(
                    salami_slices, curr_slice_idx, voice_i, voice_j)

                if comparison_slice is None:
                    continue

                prev_note_i = comparison_slice.notes[voice_i]
                prev_note_j = comparison_slice.notes[voice_j]

                # Both voices must have actual notes (not rests) in the comparison slice
                if not (prev_note_i and prev_note_i.note_type == 'note' and
                        prev_note_j and prev_note_j.note_type == 'note'):
                    continue

                # Check section crossing
                if CounterpointRulesBase._chronological_slices_cross_section([comparison_slice, curr_slice], metadata):
                    continue

                # Calculate current interval (octave reduced)
                curr_interval = abs(note_i.midi_pitch - note_j.midi_pitch) % 12
                curr_interval_type = 'fifth' if curr_interval == 7 else 'octave' if curr_interval == 0 else 'other'

                # Check if current interval is fifth (7) or octave/unison (0)
                if curr_interval_type in ['fifth', 'octave']:
                    # Calculate previous interval (octave reduced)
                    prev_interval = abs(prev_note_i.midi_pitch - prev_note_j.midi_pitch) % 12
                    prev_interval_type = 'fifth' if prev_interval == 7 else 'octave' if prev_interval == 0 else 'other'

                    # Check that the interval type is the same in both slices
                    if curr_interval_type == prev_interval_type:
                        violations.append(RuleViolation(
                            rule_name=rulename,
                            rule_id=rule_id,
                            slice_index=curr_slice_idx,
                            original_line_num=curr_slice.original_line_num,
                            beat=curr_slice.beat,
                            bar=curr_slice.bar,
                            voice_indices=(voice_i, voice_j),
                            voice_names=(metadata['voice_names'][voice_i], metadata['voice_names'][voice_j]),
                            note_names=(note_i.note_name, prev_note_i.note_name, note_j.note_name, prev_note_j.note_name)
                        ))

        return violations

    '''
    @staticmethod
    def _is_whole_beat(slice_obj: SalamiSlice) -> bool:
        """Check if slice falls on a whole beat."""
        # TODO: make explicit list of this, with also all triplet possibilities
        # Whole beat: beat is an integer (1.0, 2.0, etc.)
        return abs(slice_obj.beat - round(slice_obj.beat)) < 1e-6

    @staticmethod
    def _find_most_recent_previous_whole_beat_slice_for_voices_(salami_slices: list, curr_slice_idx: int, voice_i: int, voice_j: int) -> SalamiSlice|None:
        """
        Find the most recent previous whole beat slice by checking each voice separately
        and returning the slice with the largest original_line_num (most recent in time).
        """
        # Find previous whole beat slice for voice_i
        prev_slice_i = None
        slice_idx = curr_slice_idx - 1

        while slice_idx >= 0:
            candidate_slice = salami_slices[slice_idx]
            if (CounterpointRulesMotion._is_whole_beat(candidate_slice) and 
                candidate_slice.notes[voice_i] is not None):
                prev_slice_i = candidate_slice
                break
            slice_idx -= 1
        
        # Find previous whole beat slice for voice_j
        prev_slice_j = None
        slice_idx = curr_slice_idx - 1
        while slice_idx >= 0:
            candidate_slice = salami_slices[slice_idx]
            if (CounterpointRulesMotion._is_whole_beat(candidate_slice) and 
                candidate_slice.notes[voice_j] is not None):
                prev_slice_j = candidate_slice
                break
            slice_idx -= 1
        
        # Return the slice with the largest original_line_num (most recent)
        if prev_slice_i is None and prev_slice_j is None:
            return None
        elif prev_slice_i is None:
            return prev_slice_j
        elif prev_slice_j is None:
            return prev_slice_i
        else:
            most_recent_slice = prev_slice_i if prev_slice_i.original_line_num > prev_slice_j.original_line_num else prev_slice_j
            return most_recent_slice

    @staticmethod
    def parallel_fifth_octave(rulename, **kwargs) -> list[RuleViolation]:
        """
        Parallel fifths [or 5th+8th, see ex 52] or octaves are not found on successive [whole] beats, 
        or successive strong beats, even if there are consonant notes in between.
        """
        rule_id = '48'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []
        checked_pairs = set()  # Track already checked voice pairs

        # Only check slices that fall on whole beats
        if not CounterpointRulesMotion._is_whole_beat(curr_slice):
            return violations

        for voice_i, note_i in enumerate(curr_slice.notes):
            if not (note_i and note_i.note_type == 'note'):
                continue
            
            for voice_j, note_j in enumerate(curr_slice.notes):
                # Don't compare voice i with itself, and check each voice pair only once
                if voice_j < voice_i or voice_j == voice_i:
                    continue

                # Skip if we've already checked this pair
                pair_key = (voice_i, voice_j)
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Either note i or note j must be a new occurrence
                if not (note_j and note_j.note_type == 'note' and 
                    (note_i.is_new_occurrence and note_j.is_new_occurrence)):
                    continue

                # Find the most recent previous whole beat slice that contains both voices
                comparison_slice = CounterpointRulesMotion._find_most_recent_previous_whole_beat_slice_for_voices_(
                    salami_slices, curr_slice_idx, voice_i, voice_j)

                if comparison_slice is None:
                    continue

                prev_note_i = comparison_slice.notes[voice_i]
                prev_note_j = comparison_slice.notes[voice_j]

                # Both voices must have actual notes (not rests) in the comparison slice
                if not (prev_note_i and prev_note_i.note_type == 'note' and
                        prev_note_j and prev_note_j.note_type == 'note'):
                    continue

                # Check section crossing
                if CounterpointRulesBase._chronological_slices_cross_section([comparison_slice, curr_slice], metadata):
                    continue

                # Calculate current interval (octave reduced)
                curr_interval = abs(note_i.midi_pitch - note_j.midi_pitch) % 12
                curr_interval_type = 'fifth' if curr_interval == 7 else 'octave' if curr_interval == 0 else 'other'

                # Check if current interval is fifth (7) or octave/unison (0)
                if curr_interval_type in ['fifth', 'octave']:
                    # Calculate previous interval (octave reduced)
                    prev_interval = abs(prev_note_i.midi_pitch - prev_note_j.midi_pitch) % 12
                    prev_interval_type = 'fifth' if prev_interval == 7 else 'octave' if prev_interval == 0 else 'other'

                    # Check that the interval type is the same in both slices
                    if curr_interval_type == prev_interval_type:
                        violations.append(RuleViolation(
                            rule_name=rulename,
                            rule_id=rule_id,
                            slice_index=curr_slice_idx,
                            original_line_num=curr_slice.original_line_num,
                            beat=curr_slice.beat,
                            bar=curr_slice.bar,
                            voice_indices=(voice_i, voice_j),
                            voice_names=(metadata['voice_names'][voice_i], metadata['voice_names'][voice_j]),
                            note_names=(note_i.note_name, prev_note_i.note_name, note_j.note_name, prev_note_j.note_name)
                        ))

        return violations
from Note import Note, SalamiSlice
from constants import DURATION_MAP, TIME_SIGNATURE_STRONG_BEAT_MAP

from counterpoint_rules_base import CounterpointRulesBase, RuleViolation


class CounterpointRulesNormalization(CounterpointRulesBase):
    """Normalization functions for debugging and analysis."""

    """ %%% Normalization functions %%% """
    @staticmethod
    def _norm_count_tie_starts(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """ Detect tie ends in the current slice. """
        rule_id = '_N1'
        
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
    def _norm_count_tie_ends(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """ Detect tie ends in the current slice. """
        rule_id = '_N2'

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
    def _norm_label_chord_name_m21(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """ Labels m21 chord name """
        rule_id = '_N3'
        
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
    def _norm_ties_contained_in_bar(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """ Count ties that fall within a bar and do not cross a barline. """
        rule_id = '_N4'
        
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

    @staticmethod
    def _norm_tie_end_not_new_occurrence(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """ Count and label all new occurrence notes in the current slice. """
        rule_id = '_N5'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is a new occurrence
            if (curr_note and curr_note.note_type == 'note' and 
                curr_note.midi_pitch != -1 and curr_note.is_new_occurrence and curr_note.is_tie_end):
                
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
    def _norm_count_dotted_notes(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """ Count and label all dotted notes (new occurrences) in the current slice for debugging purposes.
        This helps verify that the is_dotted property is being set correctly during parsing."""
        rule_id = '_N6'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is a dotted note (new occurrence only to avoid counting tied notes multiple times)
            if (curr_note and curr_note.note_type in ['note', 'rest'] and 
                curr_note.is_dotted and curr_note.is_new_occurrence):
                
                # Create a descriptive note name that includes the duration and dotted status
                note_description = f"{curr_note.note_name if curr_note.note_type == 'note' else 'rest'}"
                duration_description = f"dur={curr_note.duration}"
                dotted_description = f"dotted={curr_note.is_dotted}"
                
                violations.append(RuleViolation(
                    rule_name=rulename,
                    rule_id=rule_id,
                    slice_index=curr_slice_idx,
                    original_line_num=curr_slice.original_line_num,
                    beat=curr_slice.beat,
                    bar=curr_slice.bar,
                    voice_indices=voice_number,
                    voice_names=metadata['voice_names'][voice_number],
                    note_names=curr_note.note_name + ', ' + duration_description
                ))

        return violations

    

    # ---- Actual normalization functions ---
    @staticmethod
    def norm_note_count(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """ Count all new occurrences of notes in the current slice. This normalizes rule 1 (note_count). """
        rule_id = 'N00'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is a new occurrence
            if curr_note and curr_note.note_type == 'note' and curr_note.is_new_occurrence:
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
    def norm_leap_count(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """Count all leaps (> 2 semitones) between consecutive notes. This normalizes rule 22 (leap_too_large) by counting total leap opportunities."""
        rule_id = 'N22'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            prev_slice = curr_slice.previous_note_per_voice[voice_number]
            # Check if this is a note, is a new occurrence, and has a previous slice
            if (curr_note.note_type == 'note' and 
                curr_note.is_new_occurrence and 
                prev_slice is not None):

                # Gather all relevant notes
                prev_note = prev_slice.notes[voice_number]

                # Gather all relevant rests
                prev_rest_slice = curr_slice.previous_rest_per_voice[voice_number]

                if prev_note is not None:
                    # Check if we cross a section or have a rest between notes (same logic as rule 22)
                    section_crossed = (curr_slice.bar in metadata['section_starts'] and 
                                    prev_slice.bar in metadata['section_ends'])
                    rest_between_curr_prev = (prev_rest_slice and 
                                            prev_rest_slice.original_line_num > prev_slice.original_line_num)

                    if section_crossed or rest_between_curr_prev:
                        continue
                    else:
                        # Count ALL leaps (> 2 semitones), regardless of size
                        interval = abs(curr_note.midi_pitch - prev_note.midi_pitch)
                        if interval > 2:  # This is a leap
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
    def norm_approached_followed_leap(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """
        Count all leaps that are approached and followed by a note (without section crossings or rests).
        This normalizes rule 23 (leap_approach_left_opposite) by counting total leap opportunities
        that could potentially violate the directional rule.
        """
        rule_id = 'N23'
        
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
                    
                    # Check if there is a rest in between any of the relevant notes
                    rest_between_curr_next = next_rest_slice and next_rest_slice.original_line_num < next_slice.original_line_num
                    rest_between_curr_prev = prev_rest_slice and prev_rest_slice.original_line_num > prev_slice.original_line_num
                    rest_between_curr_prev_prev = prev_prev_rest_slice and ((prev_rest_slice.original_line_num > prev_prev_slice.original_line_num)\
                                                                            or (prev_prev_rest_slice.original_line_num > prev_prev_slice.original_line_num) )
                    
                    if section_crossed:
                        continue
                    elif rest_between_curr_next or rest_between_curr_prev or rest_between_curr_prev_prev:
                        continue
                    else:
                        # Check if the main interval is a leap (> 2 semitones)
                        interval_leap = abs(curr_note.midi_pitch - prev_note.midi_pitch)
                        
                        # Count ALL leaps that are approached and followed by notes
                        # (regardless of direction - that's what rule 23 checks)
                        if interval_leap > 2:
                            violations.append(RuleViolation(
                                rule_name=rulename,
                                rule_id=rule_id,
                                slice_index=curr_slice_idx,
                                original_line_num=curr_slice.original_line_num,
                                beat=curr_slice.beat,
                                bar=curr_slice.bar,
                                voice_indices=voice_number,
                                voice_names=metadata['voice_names'][voice_number],
                                # Chronological order (earliest to latest)
                                note_names=(prev_prev_note.note_name, prev_note.note_name, curr_note.note_name, next_note.note_name)
                            ))
                            
        return violations

    @staticmethod
    def norm_successive_leap_count(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """
        Count all occurrences of two leaps in a row.
        This normalizes the interval_order_motion (25) rule by counting total possibilities.
        """
        rule_id = 'N25'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            if not (curr_note.note_type == 'note' and curr_note.is_new_occurrence):
                continue

            # Get two previous note slices (both must be actual notes) - same as rule 25
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

            # Check for section crossings or rests - same as rule 25
            if CounterpointRulesBase._chronological_slices_cross_section(
                [prev_2nd_slice, prev_1st_slice, curr_slice], metadata):
                continue

            # Calculate intervals between consecutive notes
            later_leap = abs(curr_note.midi_pitch - prev_1st_note.midi_pitch)     # More recent leap
            earlier_leap = abs(prev_1st_note.midi_pitch - prev_2nd_note.midi_pitch)  # Earlier leap
            
            # Count ALL cases where we have two consecutive leaps (> 2 semitones each)
            if later_leap > 2 and earlier_leap > 2:
                # This is a case of two leaps in a row - count it regardless of direction or interval order
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
    def norm_approached_strong_beat_count(rulename, **kwargs) -> dict[str, list[RuleViolation]]:
        """
        Count all new occurrences of notes on strong beats that are approached directly by a previous note.
        This means there should not be a section break or rest before the current note.
        This normalizes rule 27 (leap_up_accented_long_note).
        """
        rule_id = 'N27'
        
        salami_slices = kwargs['salami_slices']
        metadata = kwargs['metadata']
        curr_slice_idx = kwargs["slice_index"]
        curr_slice: SalamiSlice = salami_slices[curr_slice_idx]
        violations = []

        for voice_number, curr_note in enumerate(curr_slice.notes):
            # Check if current note is a new occurrence and get previous note
            if (curr_note and curr_note.note_type == 'note' and 
                curr_note.is_new_occurrence and
                (prev_slice := curr_slice.previous_note_per_voice[voice_number]) is not None):
            
                prev_note = prev_slice.notes[voice_number]
                
                if prev_note and prev_note.note_type == 'note':
                    # Check if we cross a section or have a rest between notes (same logic as rule 27)
                    section_crossed = (curr_slice.bar in metadata['section_starts'] and 
                                    prev_slice.bar in metadata['section_ends'])
                    
                    prev_rest_slice = curr_slice.previous_rest_per_voice[voice_number]
                    rest_between = (prev_rest_slice and 
                                prev_rest_slice.original_line_num > prev_slice.original_line_num)
                    
                    if section_crossed or rest_between:
                        continue  # Skip this note - not directly approached
                    
                    # Check if current slice falls on a strong beat
                    time_sig = metadata['time_signatures'][curr_slice.bar][voice_number]
                    numerator = str(time_sig[0])
                    denominator = str(time_sig[1])
                    
                    if (numerator in TIME_SIGNATURE_STRONG_BEAT_MAP and 
                        denominator in TIME_SIGNATURE_STRONG_BEAT_MAP[numerator]):
                        
                        strong_beats = TIME_SIGNATURE_STRONG_BEAT_MAP[numerator][denominator]
                        
                        if curr_slice.beat in strong_beats:
                            # This is a strong beat with a new note occurrence that's directly approached
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
    def norm_voice_pair_count(rulename, **kwargs) -> list[RuleViolation]:
        """
        Count total voice pairs in the piece. This is constant per piece.
        For n voices: voice_pairs = n * (n-1) / 2
        Only creates one "violation" per piece (at first slice).
        """
        rule_id = 'N41'
        
        curr_slice_idx = kwargs["slice_index"]
        
        # Only calculate once per piece (at first slice)
        if curr_slice_idx != 0:
            return []
        
        metadata = kwargs['metadata']
        voice_names = metadata['voice_names']
        n_voices = len(voice_names)
        
        # Calculate voice pairs: C(n,2) = n*(n-1)/2
        voice_pair_count = (n_voices * (n_voices - 1)) // 2
        
        # Create one violation per voice pair to get the correct total count
        violations = []
        for i in range(voice_pair_count):
            violations.append(RuleViolation(
                rule_name=rulename,
                rule_id=rule_id,
                slice_index=curr_slice_idx,
                original_line_num=1,  # First line
                beat=0.0,
                bar=1,
                voice_indices=0,  # Dummy voice
                voice_names=voice_names[0],  # First voice name
                note_names=f"voice_pairs_{n_voices}_voices"
            ))
        
        return violations
    









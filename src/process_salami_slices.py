from Note import Note, SalamiSlice
from counterpoint_rules import CounterpointRulesBase

from constants import DURATION_MAP, INFINITY_BAR, REDUCED_INTERVAL_CONSONANCE_MAP
from constants import FLOAT_TRUNCATION_DIGITS, BEAT_GRID_DIVISIONS

DEBUG2 = False

"""Functions for post-processing the salami slices such that they can be analysed."""



def post_process_salami_slices(salami_slices: list[SalamiSlice], 
                               metadata, expand_metadata_flag=True) -> tuple[list[SalamiSlice], dict[str, any]]:
    """ Post-process the salami slices.  """
    # Note, most of this could be done in one pass, but for developmental ease we do it in multiple passes.
    salami_slices, metadata = order_voices(salami_slices, metadata) 
    salami_slices = set_voice_in_notes(salami_slices) # Set the voice in the notes (can bemerged into order_voices function)
    salami_slices = remove_barline_slice(salami_slices)

    # Handle new occurrences and ties
    salami_slices = set_period_notes(salami_slices)
    salami_slices = set_tied_notes_new_occurrences(salami_slices)
    
    # Handle rhythmic properties
    salami_slices = calculate_offsets(salami_slices)  # Calculate raw offsets from bar start
    salami_slices = calculate_beat_positions(salami_slices, metadata)  # Convert offsets to beats

    # Calculate harmonic properties
    salami_slices = set_interval_property(salami_slices)
    salami_slices = set_chordal_properties(salami_slices) 


    salami_slices = link_salami_slices(salami_slices)

    # Expand metadata to be per bar
    if expand_metadata_flag:
        metadata = expand_metadata(salami_slices, metadata)  # Expand metadata to be per bar
    
    return salami_slices, metadata

def remove_barline_slice(salami_slices: list[SalamiSlice]) -> list[SalamiSlice]:
    """ Remove all barline slices. """
    salami_slices = [salami_slice for salami_slice in salami_slices if salami_slice.notes[0].note_type not in ['barline', 'final_barline']]
    return salami_slices

def set_voice_in_notes(salami_slices: list[SalamiSlice]) -> tuple[list[SalamiSlice], dict[str, any]]:
    """ Set the voice in the notes. This function should be called after the notes (voices) are sorted. """
    for salami_slice in salami_slices:
        for voice_idx, note in enumerate(salami_slice.notes):
            # Set the voice in the note
            note.voice = voice_idx
    return salami_slices


def set_period_notes(salami_slices: list[SalamiSlice]) -> list[SalamiSlice]:
    """ Set the notes in the salami slices that are periods to the notes in the previous slice """
    for i, salami_slice in enumerate(salami_slices):
        for voice, note in enumerate(salami_slice.notes):
            if note.note_type == 'period':
                if i == 0:
                    raise ValueError("Period note in first slice")
                
                # Get the previous note as a reference
                prev_note = salami_slices[i-1].notes[voice]

                if prev_note.note_type == 'rest':
                    a = 1
                # Copy all of the properties to the period note
                new_note = Note(
                    # Pitch properties
                    midi_pitch=prev_note.midi_pitch,
                    octave=prev_note.octave,
                    spelled_name=prev_note.spelled_name,

                    # Rhythm properties
                    bar=salami_slice.bar,  # Use current slice's bar, not previous note's bar
                    duration=prev_note.duration, # TODO: should be set to the duration of the slice, not the previous note
                    is_dotted=prev_note.is_dotted, 

                    # Tie properties
                    # These should explicitly not be copied, since the period can denote a tie-continuation or a tie-end.

                    # Note quality 
                    note_type=prev_note.note_type,
                    was_originally_period=True,  # THIS note was originally a period
                    is_new_occurrence=False,  # Explicitly not a new occurrence

                    is_measured_differently=prev_note.is_measured_differently,
                    is_longa=prev_note.is_longa,

                    # Harmonic properties
                    # Are set later, for all notes. Naturally, they can vary as the note underneath the period note changes (by definition).

                    # Voice information
                    voice=prev_note.voice,
                )

                # Replace the period note with our new note
                salami_slice.notes[voice] = new_note

    return salami_slices

def set_tied_notes_new_occurrences(salami_slices: list[SalamiSlice]) -> list[SalamiSlice]:
    """ Set the new occurrences of tied notes to False """
    for i, salami_slice in enumerate(salami_slices):
        for voice, note in enumerate(salami_slice.notes):
            if note.note_type == 'rest':
                # A rest is never tied, so we continue (the logic below does not handle this fact)
                if note.was_originally_period:
                    # If the rest was originally a period, we set it to not a new occurrence
                    salami_slice.notes[voice].is_new_occurrence = False
                continue
            
            # Period notes are never new occurrences, regardless of tie status
            if note.was_originally_period:
                salami_slice.notes[voice].is_new_occurrence = False

            # Store original values for debugging
            original_is_new_occurrence = note.is_new_occurrence
            original_tie_start = note.is_tie_start
            original_tie_end = note.is_tie_end
            original_tie_continuation = note.is_tie_continuation
            original_was_originally_period = note.was_originally_period
            
            # Apply tie logic - tie ends that are given as a new note (not a period) are labelled as NOT a new occurrence.
            if note.is_tie_end:
                # Tie end are not new occurrences
                salami_slice.notes[voice].is_new_occurrence = False
            elif note.is_tie_continuation:
                # Tie continuation notes are not new occurrences
                salami_slice.notes[voice].is_new_occurrence = False
            elif note.is_tie_start:
                # If the note is a tie start, set the new occurrence to True
                salami_slice.notes[voice].is_new_occurrence = True
            else:
                # If none of the tie conditions apply, keep the current value
                # (False for period notes, True for regular notes)
                pass

            # Debugging: Check for unexpected changes in tie logic. 
            new_is_new_occurrence = salami_slice.notes[voice].is_new_occurrence
            new_tie_start = salami_slice.notes[voice].is_tie_start
            new_tie_end = salami_slice.notes[voice].is_tie_end
            new_tie_continuation = salami_slice.notes[voice].is_tie_continuation
            new_was_originally_period = salami_slice.notes[voice].was_originally_period
            
            # Check if tie properties changed (they shouldn't change in this function)
            # Note: tie_end new occurrence status is expected to change, so dont check that one.
            tie_properties_changed = (
                original_tie_start != new_tie_start or
                original_tie_end != new_tie_end or
                original_tie_continuation != new_tie_continuation or
                original_was_originally_period != new_was_originally_period
            )

            if tie_properties_changed:
                raise ValueError(
                    f"DEBUGGING ERROR: Tie properties changed unexpectedly for note in slice {i}, voice {voice}.\n"
                    f"Note: {note}\n"
                    f"Original tie flags: start={original_tie_start}, end={original_tie_end}, continuation={original_tie_continuation}, was_period={original_was_originally_period}\n"
                    f"New tie flags: start={new_tie_start}, end={new_tie_end}, continuation={new_tie_continuation}, was_period={new_was_originally_period}\n"
                    f"This function should only modify is_new_occurrence, not tie flags."
                )
            
    return salami_slices


def order_voices(salami_slices: list[SalamiSlice], metadata: dict[str, any]) -> tuple[list[SalamiSlice], dict[str, any]]:
    """ Order the voices from high to low using pitch analysis. """
    # Try the pitch-based approach first
    try:
        return order_voices_by_pitch_analysis(salami_slices, metadata)
    except Exception as e:
        print(f"Warning: Pitch-based voice ordering failed ({e}), falling back to name-based ordering")
        return order_voices_by_name_legacy(salami_slices, metadata)
    
def order_voices_by_pitch_analysis(salami_slices: list[SalamiSlice], metadata: dict[str, any]) -> tuple[list[SalamiSlice], dict[str, any]]:
    """ Order the voices from high to low based on pitch analysis of actual note content. """
    
    num_voices = len(metadata['voice_names'])
    
    # Collect pitch statistics for each voice
    voice_pitch_stats = {}
    for voice_idx in range(num_voices):
        pitches = []
        for slice_obj in salami_slices:
            note = slice_obj.notes[voice_idx]
            if note and note.note_type == 'note' and note.midi_pitch != -1:
                pitches.append(note.midi_pitch)
        
        if pitches:
            voice_pitch_stats[voice_idx] = {
                'min_pitch': min(pitches),
                'max_pitch': max(pitches),
                'mean_pitch': sum(pitches) / len(pitches),
                'note_count': len(pitches)
            }
        else:
            # Voice has no notes - assign neutral values that won't interfere
            voice_pitch_stats[voice_idx] = {
                'min_pitch': 60,  # Middle C as neutral
                'max_pitch': 60,
                'mean_pitch': 60,
                'note_count': 0
            }
    
    # Determine voice ordering using the first and last voices as reference points
    first_voice_idx = 0
    last_voice_idx = num_voices - 1
    
    first_voice_stats = voice_pitch_stats[first_voice_idx]
    last_voice_stats = voice_pitch_stats[last_voice_idx]
    
    # Determine which is higher by comparing pitch ranges
    # Can raise ValueError if voices cannot be differentiated. The calling function will catch this and fall back to legacy ordering
    first_voice_is_higher = _is_voice_higher(first_voice_stats, last_voice_stats)
    
    
    # Determine voice order based on original ordering pattern
    if first_voice_is_higher:
        # Original order is highest to lowest (first voice is highest)
        # Keep the original order: [0, 1, 2, 3, ...]
        voice_order = list(range(num_voices))
    else:
        # Original order is lowest to highest (last voice is highest)
        # Reverse the order to put highest first: [..., 3, 2, 1, 0]
        voice_order = list(range(num_voices - 1, -1, -1))
    
    # Apply the reordering
    metadata['voice_sort_map'] = {
        new_idx: old_idx for new_idx, old_idx in enumerate(voice_order)
    }
    
    # Store original voice order for reference
    metadata['unsorted_voice_order'] = list(metadata['voice_names'])
    
    # Reorder the voices in the salami slices (highest voice at index 0)
    for salami_slice in salami_slices:
        salami_slice.notes = [salami_slice.notes[voice] for voice in voice_order]
    
    # Sort all other voice-dependent metadata
    metadata['voice_names'] = [metadata['voice_names'][voice].lower() for voice in voice_order]
    
    # Reorder time signatures
    reordered_time_signatures = []
    for barline_tuple, timesigs in metadata['time_signatures']:
        reordered_time_signatures.append(
            (barline_tuple, [timesigs[v] for v in voice_order])
        )
    metadata['time_signatures'] = reordered_time_signatures
    
    # Store pitch analysis results for debugging
    metadata['pitch_analysis'] = {
        'original_voice_stats': voice_pitch_stats,
        'first_voice_is_higher': first_voice_is_higher,
        'final_voice_order': voice_order
    }
    
    return salami_slices, metadata


def order_voices_by_name_legacy(salami_slices: list[SalamiSlice], metadata: dict[str, any]) -> tuple[list[SalamiSlice], dict[str, any]]:
    """ Legacy voice ordering by name (kept as fallback). """

    # First, map the voice_names to their order
    modern_names = [CounterpointRulesBase.old_to_modern_voice_name_mapping[voice_name.strip().lower()] for voice_name in metadata['voice_names']]

    # Sort according to the order of the voice names
    order_of_voice_names = {
        'soprano': 0,
        'alto': 1,
        'tenor': 2,
        'bass': 3,
    }
    # Record the old voice names (new object, since the metadata dict will be modified)
    metadata['unsorted_voice_order'] = list(metadata['voice_names'])

    # TODO: duplicate voice names now get an abitrary order?
    # Sort the voices according to the order of the voice names
    voice_order = sorted(
        range(len(modern_names)),
        key=lambda x: order_of_voice_names[modern_names[x]]
    )
    
    # Create a dictionary mapping new_position -> old_position.
    # E.g. if soprano is at index 3 in the old .krn, but we want it in new index 0,
    # then sorted_voice_order[0] = 3.
    metadata['voice_sort_map'] = {
        new_idx: old_idx for new_idx, old_idx in enumerate(voice_order)
    }

    # Reorder the voices in the salami slices, and metadata. Highest voice at index 0.
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



def calculate_offsets(salami_slices: list[SalamiSlice]) -> list[SalamiSlice]:
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
            triplet_notes = [note.is_measured_differently for note in cur_slice.notes]
            if any(triplet_notes):
                print(f"Triplet notes: {triplet_notes}")

        # Go forwards in time by the minimum duration of the notes in this slice
        current_offset += time_step 
        leftover_durations = [duration - time_step for duration in leftover_durations if duration]

        a = 1
    return salami_slices


def calculate_beat_positions(salami_slices: list[SalamiSlice], metadata) -> list[SalamiSlice]:
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
        #    E.g. for 3/4, denominator=4 => measure_length=0.75 (because of numerator * DURATION_MAP[4])
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

def set_interval_property(salami_slices: list[SalamiSlice]) -> list[SalamiSlice]:
    """ Set the intervals in the salami slices """
    for i, salami_slice in enumerate(salami_slices):
        salami_slice.absolute_intervals = salami_slice._calculate_intervals()
        salami_slice.reduced_intervals = salami_slice._calculate_reduced_intervals()
    return salami_slices

def set_chordal_properties(salami_slices: list[SalamiSlice]) -> list[SalamiSlice]:
    """ Set choral properties, using m21. This includes e.g. chord name, and consonance of notes."""
    
    # Calculates for each slice the chord analysis, the lowest note, and determines consonance/dissonance for each note.
    for salami_slice in salami_slices:
        # --- Calculate and store chord analysis for the slice ---
        current_chord_analysis_data = salami_slice._calculate_chord_analysis()
        salami_slice.chord_analysis = current_chord_analysis_data

        # --- Find the index of the voice with the lowest MIDI pitch ---
        # Create a list of (original_index, note_object) for valid, sounding notes
        indexed_sounding_notes = [
            (i, note_obj)
            for i, note_obj in enumerate(salami_slice.notes)
            if note_obj and note_obj.note_type == 'note' and note_obj.midi_pitch != -1
        ]

        if not indexed_sounding_notes:
            # No sounding notes in the slice
            salami_slice.lowest_voice = None
        else:
            # Use min() with a key that extracts the midi_pitch from the note_object part of the tuple.
            # min() will return the tuple (original_index, note_object) that has the smallest pitch.
            min_item_tuple = min(indexed_sounding_notes, key=lambda item: item[1].midi_pitch)
            salami_slice.lowest_voice = min_item_tuple[0]

        # --- Calculate the interval of each note to the root of the chord. Set consonance/dissonance accordingly. --- 
        if salami_slice.chord_analysis:
            root_note_voice = salami_slice.chord_analysis.get('root_note_voice', None)
        else:
            root_note_voice = None
        
        # If there is no root note voice, the slice has no notes.
        if root_note_voice is None:
            continue

        for voice_idx, note in enumerate(salami_slice.notes):
            if note.note_type == 'note':
                if current_chord_analysis_data['quality'] == 'single_note':
                    note.absolute_interval_to_root = 0
                    note.reduced_interval_to_root = 0
                    note.is_consonance = REDUCED_INTERVAL_CONSONANCE_MAP[0]
                else:
                    note.absolute_interval_to_root = salami_slice.absolute_intervals[(root_note_voice, voice_idx)]
                    note.reduced_interval_to_root = salami_slice.reduced_intervals[(root_note_voice, voice_idx)]
                    note.is_consonance = REDUCED_INTERVAL_CONSONANCE_MAP[note.reduced_interval_to_root]
            
    return salami_slices




def link_salami_slices(salami_slices: list[SalamiSlice]) -> list[SalamiSlice]:
    """ Link each slice to the next and previous slices containing notes or new occurrences for each voice. """
    num_slices = len(salami_slices)

    # --- Link forwards ---
    for i, cur_slice in enumerate(salami_slices):
        for voice_idx, _ in enumerate(cur_slice.notes):
            # Find the next slice containing *any* note, the next slice containing a *new occurrence* note,
            # and the next slice containing a *rest* for this voice
            next_slice_with_any_new_occ = None    # Will store the slice with whatever the next note (e.g. rest or note) is
            next_slice_with_note_new_occ = None   # Will store the slice with the following *new occurrence* note
            next_slice_with_rest_new_occ = None   # Will store the slice with the following *rest*

            # Loop forwards from the slice after the current slice
            for j in range(i + 1, num_slices):
                # Get the slice and the specific note we are checking
                next_slice_candidate = salami_slices[j]
                note_in_next_slice = next_slice_candidate.notes[voice_idx]

                # Check if it's a musical note (not rest, barline, etc.)
                is_musical_note = (note_in_next_slice.note_type == 'note')
                is_rest = (note_in_next_slice.note_type == 'rest')

                # Store any next new occurrence (rest or note).
                if not next_slice_with_any_new_occ and note_in_next_slice.is_new_occurrence:
                    next_slice_with_any_new_occ = next_slice_candidate

                # If we haven't found the following *new occurrence* note's slice yet,
                # and this slice contains a new occurrence note, store this slice.
                if not next_slice_with_note_new_occ and is_musical_note and note_in_next_slice.is_new_occurrence:
                    next_slice_with_note_new_occ = next_slice_candidate
                    
                # If we haven't found the following *rest* yet, and this slice contains a rest, 
                # and slice is the first slice of the rest (new occurrence), store this slice.
                if not next_slice_with_rest_new_occ and is_rest and note_in_next_slice.is_new_occurrence:
                    next_slice_with_rest_new_occ = next_slice_candidate

                # Optimization: If we've found all target slices, we can stop searching forwards.
                if next_slice_with_any_new_occ and next_slice_with_note_new_occ and next_slice_with_rest_new_occ:
                    break

            # Assign the found slices to the current slice's attributes
            cur_slice.next_note_per_voice[voice_idx] = next_slice_with_note_new_occ
            cur_slice.next_any_note_per_voice[voice_idx] = next_slice_with_any_new_occ
            cur_slice.next_rest_per_voice[voice_idx] = next_slice_with_rest_new_occ

    # --- Link backwards ---
    for idx, cur_slice in enumerate(salami_slices):
        for voice_idx, _ in enumerate(cur_slice.notes):
            # Find the previous slice containing *any* new occurrence note, the previous slice containing a *new occurrence* note,
            # and the previous slice containing a *rest* for this voice
            prev_slice_with_any_new_occ = None       # Will store the slice with the preceding *rest* OR *note*
            prev_slice_with_note_new_occ = None      # Will store the slice with the preceding *new occurrence* note
            prev_slice_with_rest_new_occ = None      # Will store the slice with the preceding *rest*

            # Loop backwards from the slice before the current slice
            for j in range(idx - 1, -1, -1):
                # Get the slice and the specific note we are checking
                prev_slice_candidate = salami_slices[j]
                note_in_prev_slice = prev_slice_candidate.notes[voice_idx]

                # Check if it's a musical note (not rest, barline, etc.)
                is_musical_note = (note_in_prev_slice.note_type == 'note')
                is_rest = (note_in_prev_slice.note_type == 'rest')

                # Store any previous new occurrence (rest or note).
                if not prev_slice_with_any_new_occ and note_in_prev_slice.is_new_occurrence:
                    prev_slice_with_any_new_occ = prev_slice_candidate

                # If we haven't found the previous *new occurrence* note's slice yet,
                # and this slice contains a new occurrence note, store this slice.
                if not prev_slice_with_note_new_occ and is_musical_note and note_in_prev_slice.is_new_occurrence:
                    prev_slice_with_note_new_occ = prev_slice_candidate
                    
                # If we haven't found the previous *rest* yet,
                # and this slice contains a rest, store this slice.
                if not prev_slice_with_rest_new_occ and is_rest and note_in_prev_slice.is_new_occurrence:
                    prev_slice_with_rest_new_occ = prev_slice_candidate

                # Optimization: If we've found all target slices, we can stop searching backwards.
                if prev_slice_with_any_new_occ and prev_slice_with_note_new_occ and prev_slice_with_rest_new_occ:
                    break

            # Assign the found slices to the current slice's attributes
            cur_slice.previous_any_note_type_per_voice[voice_idx] = prev_slice_with_any_new_occ
            cur_slice.previous_note_per_voice[voice_idx] = prev_slice_with_note_new_occ
            cur_slice.previous_rest_per_voice[voice_idx] = prev_slice_with_rest_new_occ

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

def expand_metadata(salami_slices, metadata: dict):
    # Go from a list of changes in key-signatures / voice-counts / etc to a 
    # list where each index represents the bar number and the value is value at that bar.
    # Final data structure: [None, (2,1), (2,1), (3,1) ], which denotes no time-sig in bar 0, then 2/1 time-sig for bar 1&2, and a change into 3/1 at bar 3.
    
    if metadata['jrpid'] in ['Bus100e']:
        a = 1  # Debugging: this piece breaks on metadata expansion.

    # Metadata categories to be expanded
    convert_from_barline_tuple_to_expanded = ['time_signatures', 'key_signatures']
    for category in convert_from_barline_tuple_to_expanded:
        
        expanded_values = [None] * (metadata['total_bars']+1) # bar numbers start at 1
        for barline_tuple, values in metadata[category]:
            # Get the start and end bar numbers from the tuple
            start_bar, end_bar = barline_tuple
            if end_bar == INFINITY_BAR:
                end_bar = metadata['total_bars']-1
            # Fill the expanded values for each bar in the range
            for bar in range(int(start_bar), int(end_bar) + 1):
                expanded_values[bar+1] = values
        
        metadata[category] = expanded_values

    return metadata

def _is_voice_higher(voice1_stats: dict, voice2_stats: dict) -> bool:
    """
    Determine if voice1 is higher than voice2 based on pitch statistics.
    Uses multiple criteria with weighted scoring.
    Raises ValueError if voices cannot be differentiated (tied score).
    """
    score = 0
    
    # Criterion 1: Mean pitch (most important)
    if voice1_stats['mean_pitch'] > voice2_stats['mean_pitch']:
        score += 3
    elif voice1_stats['mean_pitch'] < voice2_stats['mean_pitch']:
        score -= 3
    
    # Criterion 2: Maximum pitch
    if voice1_stats['max_pitch'] > voice2_stats['max_pitch']:
        score += 2
    elif voice1_stats['max_pitch'] < voice2_stats['max_pitch']:
        score -= 2
    
    # Criterion 3: Minimum pitch
    if voice1_stats['min_pitch'] > voice2_stats['min_pitch']:
        score += 1
    elif voice1_stats['min_pitch'] < voice2_stats['min_pitch']:
        score -= 1
    
    # Handle tie case explicitly
    if score == 0:
        raise ValueError(
            f"Cannot determine voice hierarchy: tied score between voices.\n"
            f"Voice 1 stats: {voice1_stats}\n"
            f"Voice 2 stats: {voice2_stats}\n"
            f"Both voices have identical or perfectly balanced pitch characteristics."
        )
    
    return score > 0







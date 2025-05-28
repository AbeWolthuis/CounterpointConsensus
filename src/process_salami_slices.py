from Note import Note, SalamiSlice
from counterpoint_rules import CounterpointRules

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

    # Handle new occurences and ties
    salami_slices = set_period_notes(salami_slices)
    salami_slices = set_tied_notes_new_occurences(salami_slices)
    
    # Handle rhythmic properties
    salami_slices = calculate_offsets(salami_slices)  # Calculate raw offsets from bar start
    salami_slices = calculate_beat_positions(salami_slices, metadata)  # Convert offsets to beats

    # Calculate harmonic properties
    salami_slices = set_interval_property(salami_slices)
    salami_slices = set_chordal_properties(salami_slices) 



    salami_slices = link_salami_slices(salami_slices)


    # Expand metadata to be per bar
    if expand_metadata_flag: metadata = expand_metadata(metadata)  # Expand metadata to be per bar
    
    return salami_slices, metadata

def remove_barline_slice(salami_slices: list[SalamiSlice]) -> tuple[list[SalamiSlice], dict[str, any]]:
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
                    # bar # TODO: bar can change from the previous note
                    duration=prev_note.duration, # TODO: should be set to the duration of the slice, not the previous note

                    # Note quality
                    note_type=prev_note.note_type,
                    is_new_occurence=False,  # Explicitly not a new occurrence
                    is_measured_differently=prev_note.is_measured_differently,
                    is_longa=prev_note.is_longa,

                    # Harmonic properties
                    # Are set later, for all notes

                    # Voice information
                    voice=prev_note.voice,
                )

                # Replace the period note with our new note
                salami_slice.notes[voice] = new_note

    return salami_slices

def set_tied_notes_new_occurences(salami_slices: list[SalamiSlice]) -> list[SalamiSlice]:
    """ Set the new occurrences of tied notes to False """
    for i, salami_slice in enumerate(salami_slices):
        for voice, note in enumerate(salami_slice.notes):
            if note.note_type == 'rest':
                # A rest is never tied, so we continue (the logic below does not handle this fact)
                continue
            # If the note is a tie end, set the new occurrence to False
            if note.is_tie_end:
                salami_slice.notes[voice].is_new_occurrence = False
            # If the note is a tie, but not a tie start, set the new occurrence to False
            elif note.is_tied and not note.is_tie_start:
                salami_slice.notes[voice].is_new_occurrence = False
            elif note.is_tie_start or not note.is_tied:
                # If the note is a tie start, set the new occurrence to True
                salami_slice.notes[voice].is_new_occurrence = True
            else:
                raise ValueError(f"Note {note} in slice {i} is not a tie start, tie end or tied note")
            
    return salami_slices



def order_voices(salami_slices: list[SalamiSlice], metadata: dict[str, any]) -> tuple[list[SalamiSlice], dict[str, any]]:
    """ Order the voices from low to high in the salami slices."""

    # First, map the voice_names to their order
    modern_names = [CounterpointRules.old_to_modern_voice_name_mapping[voice_name.strip().lower()] for voice_name in metadata['voice_names']]

    # Sort according to the order of the voice names
    order_of_voice_names = {
        'soprano': 0,
        'alto': 1,
        'tenor': 2,
        'bass': 3,
    }
    # Record the old voice names (new object, since the metadata dict will be modified)
    metadata['unsorted_voice_order'] = list(metadata['voice_names'])

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
    a = 1

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
            # Find the previous slice containing *any* note, the previous slice containing a *new occurrence* note,
            # and the previous slice containing a *rest* for this voice
            prev_slice_with_any_new_occ = None       # Will store the slice with the *immediately* preceding new occurrence
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

def expand_metadata(metadata: dict):
    # Go from a list of changes in key-signatures / voice-counts / etc to a 
    # list where each index represents the bar number and the value is value at that bar.
    # Final data structure: [None, (2,1), (2,1), (3,1) ], which denotes no time-sig in bar 0, then 2/1 time-sig for bar 1&2, and a change into 3/1 at bar 3.
    
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


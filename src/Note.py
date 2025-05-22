import music21 
import math
from typing import Dict, Tuple, Union

from constants import DURATION_MAP, PITCH_TO_MIDI, MIDI_TO_PITCH
from constants import FLOAT_TRUNCATION_DIGITS, BEAT_GRID_DIVISIONS

class FloatTruncator:
    """A mixin class providing float truncation functionality."""
    
    def _truncate_float_as_float(self, x: float, digits: int) -> float:
        s = f"{x:.{digits+4}f}"  # first get a longer decimal to be safe
        point_index = s.find(".")
        if point_index == -1:
            # No decimal point, nothing to truncate
            return float(s)
        # Then slice up to the desired number of digits past the decimal
        truncated_str = s[: point_index + 1 + digits]
        return float(truncated_str)

    def truncate_float_as_float(self, x: float, digits: int) -> float:
        factor = 10 ** digits
        return math.floor(x * factor) / factor
    
    def truncate_float_as_string(self, x: float, digits: int) -> str:
        """
        Truncate a float to a specific number of digits after the decimal point,
        and return it as a string.
        """
        s = f"{x:.{digits+4}f}"  # first get a longer decimal to be safe
        point_index = s.find(".")
        if point_index == -1:
            # No decimal point, nothing to truncate
            return s
        # Then slice up to the desired number of digits past the decimal
        truncated_str = s[: point_index + 1 + digits]
        
        # Remove trailing zeros after decimal point
        if "." in truncated_str:
            truncated_str = truncated_str.rstrip("0").rstrip(".")
            
        return truncated_str
    


class SalamiSlice(FloatTruncator):
    def __init__(self, offset=-1.0, beat=-1.0, num_voices=-1, bar=-1, original_line_num=-1, notes = None,
                 lowest_voice=None, chord_analysis=None) -> None:
        self.offset: float = offset # Offset from start of bar 
        self.beat: float = beat  # Position in bar in musical beats
        self.num_voices: int = num_voices
        self.bar: int = bar
        self.notes: list[Note|None] = [None] * num_voices

        # Harmony
        self.lowest_voice: int = lowest_voice
        self.absolute_intervals = None # Will be: Dict[Tuple[int, int], int]
        self.reduced_intervals = None # Will be: Dict[Tuple[int, int], int]
        self.chord_analysis = chord_analysis

        # Record original line number in the source file
        self.original_line_num: int = original_line_num
        
        # Link each voice to a previous/next slice
        self.next_any_note_per_voice: list[SalamiSlice|None] = [None] * num_voices
        self.next_note_per_voice: list[SalamiSlice|None] = [None] * num_voices
        self.next_rest_per_voice: list[SalamiSlice|None] = [None] * num_voices
                
        self.previous_any_note_per_voice: list[SalamiSlice|None] = [None] * num_voices
        self.previous_note_per_voice: list[SalamiSlice|None] = [None] * num_voices
        self.previous_rest_per_voice: list[SalamiSlice|None] = [None] * num_voices

    def __repr__(self):
        # Show both offset and beat in the representation
        offset_str = f"offset={self.truncated_offset_as_str}, beat={self.truncated_beat_as_str}, bar={self.bar}, "
        padding_needed = max(0, 30 - len(offset_str))  # Calculate how many spaces are needed
        padded_offset_str = offset_str + " " * padding_needed  # Add the required spacespadded_offset_str = f"{self.truncated_offset_as_str:<30}"
        
        notes_str = "[" + ", ".join([str(note) for note in self.notes]) + "]"
        
        return f"{padded_offset_str}{notes_str}\n"

    __slots__ = (
        'offset',
        'beat',
        'num_voices',
        'bar',
        'notes',
        'lowest_voice',
        'absolute_intervals',
        'reduced_intervals',
        'original_line_num',
        'next_any_note_per_voice',
        'next_note_per_voice',
        'next_rest_per_voice',
        'previous_any_note_per_voice',
        'previous_note_per_voice',
        'previous_rest_per_voice',
        'chord_analysis',
    )  

    @property
    def truncated_beat_as_str(self, digits=FLOAT_TRUNCATION_DIGITS) -> str:
        return self.truncate_float_as_string(self.beat, digits)
    
    @property
    def truncated_offset_as_str(self, digits=FLOAT_TRUNCATION_DIGITS) -> str:
        return self.truncate_float_as_string(self.offset, digits)
    
    def _calculate_intervals(self) -> Dict[Tuple[int, int], int]:
        """
        Compute intervals between all pairs of notes in the slice.
        Returns a dictionary with keys as tuples of voice indices
        and values as intervals in semitones.
        """
        intervals = {}
        for i, note1 in enumerate(self.notes):
            for j, note2 in enumerate(self.notes):
                if note1 and note2 and note1.note_type == 'note' and note2.note_type == 'note':
                    interval = abs(note1.midi_pitch - note2.midi_pitch)
                    intervals[(i, j)] = interval
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
                if note1 and note2 and note1.note_type == 'note' and note2.note_type == 'note':
                    interval = abs(note1.midi_pitch - note2.midi_pitch) % 12
                    intervals[(i, j)] = interval
        return intervals
    
    
    def _calculate_chord_analysis(self) -> Union[Dict[str, any], None]:
        """
        Analyzes the slice using music21 to identify the sonority (chord or interval).
        Returns a dictionary with details if successful.
        Raises an error if music21 encounters an issue or an unexpected state.
        Returns None if there are no sounding notes.
        Note names in the output (root, bass) will be in Humdrum format (e.g., 'C', 'c', 'a-').
        """
        analysis_result = { # List of original Note objects
            # "bass_note_voice": None,
            "root_note_voice": None,
            
            #"pitch_classes_display": sorted(list(set(p % 12 for p in note_names_for_m21))),
            "sounding_note_names_voice_order": None,
            # Info about the chord
            "common_name_m21": None,
            "quality": None,
            "inversion": None,
            "is_dyad": False,
            "is_triad": False,
            "is_seventh": False,
            "is_major_triad": False,
            "is_minor_triad": False,
            "is_diminished_triad": False,
            "is_augmented_triad": False,
        }

        sounding_notes_voice_order = [note for note in self.notes if note and note.note_type == 'note' and note.midi_pitch != -1]
        #sounding_notes_pitch_order = sorted(sounding_notes_voice_order, key=lambda n: n.midi_pitch)
        analysis_result['sounding_note_names_voice_order'] = sounding_notes_voice_order

        # Use the octave-containing note names for music21. Remove the explicit natural sign from 
        note_names_for_m21_voice_order = [note.note_name.replace("n", "") for note in sounding_notes_voice_order]
        #note_names_for_m21_pitch_order = [note.note_name.replace("n", "") for note in sounding_notes_pitch_order]

        # Set lowest note (some upper voice could cross into the bass)
        # analysis_result['bass_note_voice'] = sorted_sounding_notes_objects[0].voice

        if not sounding_notes_voice_order:
            return None # No notes to analyze, a valid empty case.

        
        

        # Handle single notes directly without music21.chord.Chord
        if len(note_names_for_m21_voice_order) == 1:
            single_note_obj = sounding_notes_voice_order[0]
            analysis_result["common_name_m21"] = "single_note"
            analysis_result["quality"] = "single_note"
            analysis_result["inversion"] = "single_note"
            analysis_result["root_note_voice"] = single_note_obj.voice
            return analysis_result

        try:
            m21_sonority = music21.chord.Chord(note_names_for_m21_voice_order)
            analysis_result["common_name_m21"] = m21_sonority.commonName
            
            # Determine the voice in the slice which has the root 
            m21_root_pitch_obj = m21_sonority.root()
            a=1
            if m21_root_pitch_obj is not None:
                root_name_m21 = m21_root_pitch_obj.nameWithOctave
                # Find lowest note in the slice with the same pitch class as the root detected by music21
                for note in sounding_notes_voice_order:
                    # Compare the note name (must be without octave) to the root name from music21
                    if note.note_name.replace("n", "") == root_name_m21:
                        analysis_result["root_note_voice"] = note.voice
                        break
                else:
                    raise ValueError(f"No note with root name '{root_name_m21}' found in sorted sounding notes: {[n.note_name for n in sounding_notes_pitch_order]}")
            else:
                raise ValueError("Music21 returned None for root pitch object")
                analysis_result["root_note_name"] = "undetermined" # Could raise error if strictness demands

            analysis_result["quality"] = m21_sonority.quality
            
            inv_int = m21_sonority.inversion()
            if isinstance(inv_int, int):
                if inv_int == 0: analysis_result["inversion"] = "root"
                elif inv_int == 1: analysis_result["inversion"] = "1st"
                elif inv_int == 2: analysis_result["inversion"] = "2nd"
                elif inv_int == 3: analysis_result["inversion"] = "3rd"
                else: analysis_result["inversion"] = f"{inv_int}th"
            else:
                # This would be an unexpected return type for .inversion()
                raise TypeError(f"Music21 .inversion() returned unexpected type: {type(inv_int)} for pitches {note_names_for_m21_voice_order}")

            # Chord type flags
            if len(m21_sonority.pitches) == 2:
                analysis_result["is_dyad"] = True
            analysis_result["is_triad"] = m21_sonority.isTriad()
            analysis_result["is_seventh"] = m21_sonority.isSeventh()
            analysis_result["is_major_triad"] = m21_sonority.isMajorTriad()
            analysis_result["is_minor_triad"] = m21_sonority.isMinorTriad()
            analysis_result["is_diminished_triad"] = m21_sonority.isDiminishedTriad()
            analysis_result["is_augmented_triad"] = m21_sonority.isAugmentedTriad()

        except Exception as e: # Catch any other unexpected errors
            raise e
            raise RuntimeError(
                f"Unexpected error during music21 analysis of MIDI pitches {note_names_for_m21_voice_order} "
                f"(Bar: {self.bar}, Beat: {self.beat:.2f}, OrigLine: {self.original_line_num}, Notes: {[n.note_name for n in sounding_notes_pitch_order]}): {str(e)}"
            ) from e
        
        # Catch cases, for debugging
        if analysis_result['is_major_triad']:
            a = 1
        elif analysis_result['is_minor_triad']:
            a = 1
        elif analysis_result['is_diminished_triad']:
            a = 1
        elif analysis_result['is_augmented_triad']:
            a = 1

        return analysis_result

    def add_note(self, note, voice):
        if voice >= self.num_voices:
            raise ValueError(f"Attempt to add note to voice {voice+1}, but slice only has {self.num_voices} voices")
        self.notes[voice] = note

    def check(self):
        raise NotImplementedError("Check method not implemented for SalamiSlice class")
        return

class Note(FloatTruncator):
    midi_to_pitch = MIDI_TO_PITCH
    pitch_to_midi = PITCH_TO_MIDI

    __slots__ = (
        'midi_pitch',
        'octave'
        'spelled_name', # e.g. 
        'duration',
        'bar',
        'is_tied',
        'is_tie_start',
        'is_tie_end',
        'note_type',
        'is_new_occurrence', 
        'is_triplet',
        'is_longa',
        'is_consonance',
        'absolute_interval_to_root',
        'reduced_interval_to_root',
        'voice',
    )

    def __init__(self, 
                # Pitch
                midi_pitch: int = -1,
                octave: int = -1,
                spelled_name: str|None = None,               

                # Rhythm
                duration: float = -1.0,
                bar: int = -1,          # Only used for barlines. Otherwise, the slice has the bar information.
                
                # New tie fields:
                is_tied: bool = False,
                is_tie_start: bool = False,
                is_tie_end: bool = False,

                # Note quality
                note_type: str|None = None,
                is_new_occurence: bool = True,
                is_triplet: bool = False,    
                is_longa: bool = False,  

                # Harmonic properties
                is_consonance: bool|None = None, 
                interval_to_root: int|None = None, # Interval to root in semitones
                reduced_interval_to_root: int|None = None,

                # Redundant, but for ease of processing
                voice: int|None = None,         
            ) -> None:
        
        # Pitch
        self.midi_pitch = midi_pitch
        self.octave = octave
        self.spelled_name = spelled_name
        self.is_consonance = is_consonance
        # Rhythm
        self.duration = duration    
        self.bar = bar
        # Ties
        self.is_tied = is_tied
        self.is_tie_start = is_tie_start 
        self.is_tie_end = is_tie_end   
        # Note quality
        self.note_type = note_type
        self.is_new_occurrence = is_new_occurence
        self.is_triplet = is_triplet  
        self.is_longa = is_longa
        # Properties for ease of processing
        self.voice = voice

        return
    
    def __repr__(self):
        if self.note_type == 'note':    
            tie_repr = self.tie_repr
            if tie_repr:
                return f"({self.note_type}: {self.truncated_duration_as_str}, {self.note_name}, {self.tie_repr})"
            else:
                return f"({self.note_type}: {self.truncated_duration_as_str}, {self.note_name})"
        else:
            return f"({self.note_type}: {self.truncated_duration_as_str})"
        
    @property
    def octave_reduced(self) -> int:
        """Compute the octave reduced note number based on MIDI pitch where the note A (MIDI 57) is 0."""
        if self.midi_pitch == -1:
            return -1
        return (self.midi_pitch - 9) % 12
    
    @property
    def note_name(self) -> Union[str, None]:
        """Get the note name of the note."""
        if self.midi_pitch == -1:
            return self.note_type
        return self.spelled_name
    
    @property
    def tie_repr(self) -> str:
        """Get a string representation of the tie status."""
        if self.is_tied:
            if self.is_tie_start:
                return "tie start"
            elif self.is_tie_end:
                return "tie end"
            else:
                return "tie mid"
        else:
            return ""

    @property
    def compact_summary(self) -> str:
        """Get a compact summary of the note."""
        if self.note_type == 'note':
            return f"{self.note_name}-{self.duration}"
        else:
            return f"{self.note_type}-{self.duration}"
    
    @property
    def truncated_duration_as_str(self, digits=FLOAT_TRUNCATION_DIGITS) -> str:
        return self.truncate_float_as_string(self.duration, digits)



        
        return
    
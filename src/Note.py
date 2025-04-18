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
    def __init__(self, offset=-1, beat=-1, num_voices=-1, bar=-1, ) -> None:
        self.offset = offset
        self.beat = beat  # Position in bar in musical beats
        self.num_voices = num_voices
        self.notes = [None] * num_voices
        self.bar = bar

        self.absolute_intervals = None # Will be: Dict[Tuple[int, int], int]
        self.reduced_intervals = None # Will be: Dict[Tuple[int, int], int]

    def add_note(self, note, voice):
        if voice >= self.num_voices:
            raise ValueError(f"Attempt to add note to voice {voice+1}, but slice only has {self.num_voices} voices")
        self.notes[voice] = note

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
        # Show both offset and beat in the representation
        offset_str = f"offset={self.truncated_offset_as_str}, beat={self.truncated_beat_as_str}, bar={self.bar}, "
        notes_str = "[" + ", ".join([str(note) for note in self.notes]) + "]"
        return f"{offset_str}{notes_str}\n"



    def check(self):
        if self.num_voices == -1:
            raise ValueError("Number of voices is not set")

class Note(FloatTruncator):
    _POSSIBLE_TYPES = ('note', 'rest', 'period', 'barline', 'final_barline')
    midi_to_pitch = MIDI_TO_PITCH
    pitch_to_midi = PITCH_TO_MIDI

    def __init__(self, 
                # Pitch
                midi_pitch: int = -1,

                # Rhythm
                duration: float = -1,
                bar: int = -1,
                
                # New tie fields:
                is_tied: bool = False,
                is_tie_start: bool = False,
                is_tie_end: bool = False,

                # Note quality
                note_type: str = None,
                new_occurence: bool = True,
                is_triplet: bool = False,                 
            ) -> None:
        
        # Pitch
        self.midi_pitch = midi_pitch
        # Rhythm
        self.duration = duration
        self.bar = bar
        # Ties
        self.is_tied = is_tied       # True if part of a tie (start, middle, or end)
        self.is_tie_start = is_tie_start 
        self.is_tie_end = is_tie_end   
        # Note quality
        self.note_type = note_type
        self.new_occurrence = new_occurence
        self.is_triplet = is_triplet  

        return
    
    def __repr__(self):
        if self.note_type == 'note':
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
            return None
        return self.midi_to_pitch[self.midi_pitch]
    
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


    # Checks to make sure the note is set
    def check(self):
        if self.midi_pitch == -1 or self.duration == -1: 
            raise ValueError("Note pitch or duration is not set")
        elif self.note_type not in self._POSSIBLE_TYPES:
            raise ValueError("Note type is not set")
        
        return
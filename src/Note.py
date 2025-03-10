from typing import Dict, Tuple, Union

from constants import DURATION_MAP, PITCH_TO_MIDI, MIDI_TO_PITCH


class SalamiSlice:
    def __init__(self, num_voices=-1, bar=-1) -> None:
        self.offset = 0
        self.num_voices = num_voices
        self.notes = [None] * num_voices
        self.bar = bar

        self.absolute_intervals = None # Will be: Dict[Tuple[int, int], int]
        self.reduced_intervals = None # Will be: Dict[Tuple[int, int], int]

    def add_note(self, note, voice):
        self.notes[voice] = note

    
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
        print_str = "[" + ", ".join([str(note) for note in self.notes]) + "]\n"
        return print_str


    def check(self):
        if self.num_voices == -1:
            raise ValueError("Number of voices is not set")

class Note():
    _POSSIBLE_TYPES = ('note', 'rest', 'period', 'barline', 'final_barline')
    midi_to_pitch = MIDI_TO_PITCH
    pitch_to_midi = PITCH_TO_MIDI
    # __slots__ = ('midi_pitch', 'duration', 'note_type', 'new_occurrence')

    def __init__(self, 
                 midi_pitch: int = -1,
                 duration: float = -1,
                 note_type: str = None,
                 new_occurence: bool = True) -> None:
        
        self.midi_pitch = midi_pitch
        self.duration = duration
        self.note_type = note_type
        self.new_occurrence = new_occurence
        return
    
    def __repr__(self):
        if self.note_type == 'note':
            return f"({self.note_type}: {self.duration}, {self.midi_pitch})"
        else:
            return f"({self.note_type}: {self.duration})"
        
    @property
    def octave_reduced(self) -> int:
        """Compute the octave reduced note number based on MIDI pitch where A (MIDI 57) is 0."""
        if self.midi_pitch == -1:
            return -1
        return (self.midi_pitch - 9) % 12
    
    @property
    def note_name(self) -> Union[str, None]:
        """Get the note name of the note."""
        if self.midi_pitch == -1:
            return None
        return self.midi_to_pitch[self.midi_pitch]
    
    # Checks to make sure the note is set
    def check(self):
        if self.midi_pitch == -1 or self.duration == -1: 
            raise ValueError("Note pitch or duration is not set")
        elif self.note_type not in self._POSSIBLE_TYPES:
            raise ValueError("Note type is not set")
        
        return
from Note import SalamiSlice


class RuleViolation:
    def __init__(self, rule_name: str, rule_id: int, slice_index: int, bar: int,
                 voice_indices: tuple|str, voice_names: tuple|str,
                 note_names: tuple|str, beat: float, original_line_num: int) -> None:
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
    
class CounterpointRulesBase: 
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
        'contratenor1': 'alto',
        'contratenor2': 'alto',

        'contra': 'alto',
        'altus': 'alto',
        'contratenoraltus': 'alto',

        'tenor': 'tenor',

        'bass': 'bass',
        'bassus': 'bass',
        'contratenorbassus': 'bass',
    }

    """ Helper functions """

    
    @staticmethod
    def _chronological_slices_cross_section(slices_in_order: list[SalamiSlice | None], metadata: dict) -> bool:
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

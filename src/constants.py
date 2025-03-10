

# Duration mapping
DURATION_MAP = {
    '0': 2.0,  # Special case; longa
    '1': 1.0,  # Whole note
    '2': 0.5,  # Half note
    '4': 0.25,
    '8': 0.125,
    '16': 0.0625,
    '32': 0.03125, 
}
TRIPLET_DURATION_MAP = {
    '3': 0.333,
    '6': 0.166,
}

PITCH_TO_MIDI = {
    'CCC': 24, 'DDD-': 25, 'DDD': 26, 'EEE-': 27, 'EEE': 28, 'FFF': 29, 'GGG-': 30, 'GGG': 31, 'AAA-': 32, 'AAA': 33, 'BBB-': 34, 'BBB': 35,
    'CC': 36, 'DD-': 37, 'DD': 38, 'EE-': 39, 'EE': 40, 'FF': 41, 'GG-': 42, 'GG': 43, 'AA-': 44, 'AA': 45, 'BB-': 46, 'BB': 47,
    'C': 48, 'D-': 49, 'D': 50, 'E-': 51, 'E': 52, 'F': 53, 'G-': 54, 'G': 55, 'A-': 56, 'A': 57, 'B-': 58, 'B': 59,
    'c': 60, 'd-': 61, 'd': 62, 'e-': 63, 'e': 64, 'f': 65, 'g-': 66, 'g': 67, 'a-': 68, 'a': 69, 'b-': 70, 'b': 71,
    'cc': 72, 'dd-': 73, 'dd': 74, 'ee-': 75, 'ee': 76, 'ff': 77, 'gg-': 78, 'gg': 79, 'aa-': 80, 'aa': 81, 'bb-': 82, 'bb': 83,
    'ccc': 84, 'ddd-': 85, 'ddd': 86, 'eee-': 87, 'eee': 88, 'fff': 89, 'ggg-': 90, 'ggg': 91, 'aaa-': 92, 'aaa': 93, 'bbb-': 94, 'bbb': 95,
    'cccc': 96, 'dddd-': 97, 'dddd': 98, 'eeee-': 99, 'eeee': 100, 'ffff': 101, 'gggg-': 102, 'gggg': 103, 'aaaa-': 104, 'aaaa': 105, 'bbbb-': 106, 'bbbb': 107,
}

MIDI_TO_PITCH = {v: k for k, v in PITCH_TO_MIDI.items()}

_DIATONIC_PITCH_TO_MIDI = {
    'CCC': 24, 'DDD': 26, 'EEE': 28, 'FFF': 29, 'GGG': 31, 'AAA': 33, 'BBB': 35,
    'CC': 36, 'DD': 38, 'EE': 40, 'FF': 41, 'GG': 43, 'AA': 45, 'BB': 47,
    'C': 48, 'D': 50, 'E': 52, 'F': 53, 'G': 55, 'A': 57, 'B': 59,
    'c': 60, 'd': 62, 'e': 64, 'f': 65, 'g': 67, 'a': 69, 'b': 71,
    'cc': 72, 'dd': 74, 'ee': 76, 'ff': 77, 'gg': 79, 'aa': 81, 'bb': 83,
    'ccc': 84, 'ddd': 86, 'eee': 88, 'fff': 89, 'ggg': 91, 'aaa': 93, 'bbb': 95,
    'cccc': 96, 'dddd': 98, 'eeee': 100, 'ffff': 101, 'gggg': 103, 'aaaa': 105, 'bbbb': 107,
}
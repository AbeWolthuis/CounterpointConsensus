FLOAT_TRUNCATION_DIGITS = 5
BEAT_GRID_DIVISIONS = 48
INFINITY_BAR = 1e6

# Duration mapping
DURATION_MAP = {
    # The longa is encoded explicity in kern files, as two 0s, with a the user-specified 'l' (for longa) marker.
    '0': 2.0,   # Brevis, spans one measure in mensural notation
    '1': 1.0,   # Whole note
    '2': 0.5,   # Half note
    '4': 0.25,  # Quarter note
    '8': 0.125, # Eighth note
    '16': 0.0625,
    '32': 0.03125, 
}
TRIPLET_DURATION_MAP = {
    '3': 0.3333333, # 7 digits of 3
    '6': 0.1666666, # 6 digits of 6
}
TIME_SIGNATURE_NORMALIZATION_MAP = {
    1: 4,
    2: 2,
    4: 1,
    8: 0.5,
}
REDUCED_INTERVAL_CONSONANCE_MAP = {
    0: True,  # Unison
    1: False, # Minor 2nd
    2: False, # Major 2nd
    3: True,  # Minor 3rd
    4: True,  # Major 3rd
    5: False, # Perfect 4th
    6: False, # Augmented 4th / dimished 5th
    7: True,  # Perfect 5th
    8: True,  # Minor 6th
    9: True,  # Major 6th
    10: False, # Minor 7th
    11: False, # Major 7th
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

TIME_SIGNATURE_STRONG_BEAT_MAP = {
    # enumerator : { denominator1 : [strong beats], denominator2 : [strong beats] }
    '9':
        {
            '2': [1, 4, 7], # 9 half notes per bar (e.g. Jos2608)
        },
    '6': {
            '1' : [1, 3, 5], # 6 whole notes per bar; (C.|)(5), (C2)(5), (C|)(5), (O)(5), (O2)(5), (2)(4), (C.)(2), (C|r)(1), (O|)(1)
            '2' : [1, 4]     # 6 half notes per bar; (C.)(3)
        },
    '4' : 
        {
            1 : [1, 2, 3, 4], 
        }, 
    '3': {
            '1' :   [1, 2, 3],
            '3%2' : [1, 2, 3],  # These bars should have 3 notes of duration 2/3.
            '2' :   [1], #        # With mar2097 in mind
            '0' :   [1], # 
        },
    '2': {
            '1' : [1, 2], # 2 whole notes per bar
            '0' : [1],    # 2 brevis per bar
        }
    }   

    # 9/0, 1;  6/0, 1;   *M3/2%3


"""
Time signatures and their mensurations found in a subset of the data (larger than the used subset):

Time Signature | Count  | %     | Common Mensurations
---------------+--------+-------+----------------
*M2/1          | 8096   |  69.5% | (C|)(6165), (C)(1156), (C2)(676), (O2)(119), (O)(103), ()(52), (3)(52), (C.)(47), (O.)(24), (C|2)(15), (Cr)(9), (O3/2)(8), (C.3/8)(4), (C2/3)(4), (C|.)(4), (O3)(4), (c)(4), (С|)(4), (C|) (3), (O/3)(3), (2)(2), (O|)(1)
*M3/1          | 3383   |  29.0% | (O)(1884), (3)(707), (O|)(295), (C|3)(179), (O/3)(138), ()(131), (C3)(96), (C.)(95), (O.)(38), (C|)(27), (O|/3)(12), (O|3)(10), (3/2)(8), (C.3/2)(8), (C)(6), (C|/3)(6), (O3/2)(5), (C2)(3), (O3)(3), (O2)(2)
*M3/3%2        | 39     |   0.3% | (3)(22), ()(9), (C|)(9), (C|3)(4), (O|3/2)(3), (3/2)(1)
*M6/1          | 21     |   0.2% | (C.|)(5), (C2)(5), (C|)(5), (O)(5), (O2)(5), (2)(4), (C.)(2), (C|r)(1), (O|)(1)
*M9/2          | 21     |   0.2% | (O.)(6), ()(5), (3)(5), (O3)(4), (3/2)(1), (C|)(1)
*M4/4          | 16     |   0.1% | (C|)(4)
*M12/1         | 10     |   0.1% | (2)(5), (C)(5)
*M4/1          | 7      |   0.1% | (C|2)(4), (C.)(3), (C|r)(3)
*M6/2          | 7      |   0.1% | (C.)(3)
*M9/1          | 7      |   0.1% | (O.)(7), (O)(4), (C|)(2), (O|)(1)    # (found in all of Jos0603xa-e, and Jos0501c)
*M3/0          | 6      |   0.1% | (O)(5), (C|)(3), (O.)(1), (O2)(1), (O|)(1)
*M4/2          | 6      |   0.1% | (C|)(6)
*M18/2         | 5      |   0.0% | (O2)(5)                  # remove Oke1013e
*M2/0          | 5      |   0.0% | (C|)(5), (C)(1)          # e.g. Jos0901e, 131/135
*M30/2         | 5      |   0.0% | (O.)(5)                  # remove Oke1013e
*M2/2          | 4      |   0.0% | (C|)(4)
*M3/2          | 4      |   0.0% | (3)(4)               
*M12/0         | 2      |   0.0% | (C.)(1), (C2)(1), (C|)(1), (O)(1) (Jos0603a, Jos0903b)
*M1/1          | 1      |   0.0% | (C)(1), (C.)(1)          remove (Rue1030d)
*M3/2%3        | 1      |   0.0% | (3)(1)                   remove (Rue1023c)
*M6/0          | 1      |   0.0% | (O)(1)                   remove (Jos0603e, Jos0903b)
*M9/0          | 1      |   0.0% | (C|)(1), (O.)(1);        remove (Jos0603e)
"""
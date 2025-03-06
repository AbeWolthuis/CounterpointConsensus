import os
from pprint import pprint

from kern_parser import parse_kern, post_process_salami_slices
from counterpoint_rules import CounterpointRules

from data_preparation import violations_to_df


def main():
    # Settings stuff

    # Load kern

    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, "..", "data", "test", "extra_parFifth_rue1024a.krn")
    # filepath = os.path.join("..", "data", "test", "extra_parFifth_rue1024a.krn")
    
    # Parse
    salami_slices, metadata = parse_kern(filepath)
    salami_slices, metadata = post_process_salami_slices(salami_slices, metadata)

    # Analyze
    cp_rules = CounterpointRules()
    violations = cp_rules.validate_all_rules(salami_slices, metadata, cp_rules)
    
    print()
    pprint(violations)
    df = violations_to_df(violations, metadata)


    # Classify


    # Analyze classification


    return


if __name__ == "__main__":
    main()


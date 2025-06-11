import inspect
from collections import defaultdict


# Import the counterpoint rule classes
from counterpoint_rules_base import CounterpointRulesBase, RuleViolation

from counterpoint_rules_most import CounterpointRulesMost
from counterpoint_rules_normalization import CounterpointRulesNormalization

DEBUG = True



class CounterpointRules:
    """Main class that orchestrates all counterpoint rule categories."""

    def __init__(self):
        # Initialize all rule category classes
        self.most_rules = CounterpointRulesMost()
        self.normalization_rules = CounterpointRulesNormalization()
        
        # Create a list of all rule classes for easy iteration
        self.rule_classes = [self.most_rules, self.normalization_rules]

    def validate_all_rules(self, *args, **kwargs) -> dict[str, list[RuleViolation]]:
        only_validate_rules = kwargs['only_validate_rules'] if 'only_validate_rules' in kwargs else None

        violations = defaultdict(list)

        # Iterate through each rule class
        for rule_class in self.rule_classes:
            # Get all static methods from each rule class
            for name, func in inspect.getmembers(rule_class, predicate=inspect.isfunction):
                # Skip private methods and the base class methods

                if name != "validate_all_rules" and not name.startswith("_"):
                    # If only_validate_rules is provided, check if the rule is in the list
                    if (not only_validate_rules) or (name in only_validate_rules):
                        result = func(name, **kwargs)
                        violations[name].extend(result)

        return dict(violations)

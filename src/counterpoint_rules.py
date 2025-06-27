import inspect
from collections import defaultdict


# Import the counterpoint rule classes
from counterpoint_rules_base import RuleViolation

from counterpoint_rules_most import CounterpointRulesMost
from counterpoint_rules_motion import CounterpointRulesMotion
from counterpoint_rules_normalization import CounterpointRulesNormalization


DEBUG = True


class CounterpointRuleError(Exception):
    """Custom exception that includes rule context information."""
    def __init__(self, rule_name, original_error, rule_class_name=None):
        self.rule_name = rule_name
        self.original_error = original_error
        self.rule_class_name = rule_class_name
        
        # Create a descriptive error message
        error_msg = f"Error in rule '{rule_name}'"
        if rule_class_name:
            error_msg += f" (from {rule_class_name})"
        error_msg += f": {type(original_error).__name__}: {str(original_error)}"
        
        super().__init__(error_msg)


class CounterpointRules:
    """Main class that orchestrates all counterpoint rule categories."""

    def __init__(self):
        # Initialize all rule category classes
        self.most_rules = CounterpointRulesMost()
        self.motion_rules = CounterpointRulesMotion()
        self.normalization_rules = CounterpointRulesNormalization()
        
        # Create a list of all rule classes for easy iteration
        self.rule_classes = [CounterpointRulesMost, CounterpointRulesMotion, CounterpointRulesNormalization]

    def validate_all_rules(self, *args, **kwargs) -> dict[str, list[RuleViolation]]:
        only_validate_rules = kwargs['only_validate_rules'] if 'only_validate_rules' in kwargs else None

        violations = defaultdict(list)

        # Iterate through each rule class
        for rule_class in self.rule_classes:
            rule_class_name = rule_class.__name__
            
            # Get all static methods from each rule class
            for name, func in inspect.getmembers(rule_class, predicate=inspect.isfunction):
                # Skip private methods and the base class methods
                if name != "validate_all_rules" and not name.startswith("_"):
                    # Always run normalization functions (starting with 'norm_')
                    # For other rules, check if they're in the only_validate_rules list
                    if name.startswith('norm_') or (not only_validate_rules) or (name in only_validate_rules):
                        try:
                            result = func(name, **kwargs)
                            violations[name].extend(result)
                        except Exception as e:
                            # Re-raise with rule context
                            raise CounterpointRuleError(name, e, rule_class_name) from e

        return dict(violations)

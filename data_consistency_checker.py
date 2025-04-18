#!/usr/bin/env python3
"""
Data Consistency Checker for Counterpoint Corpus

This script validates the consistency of kern files across different composers
in the CounterpointConsensus dataset.
"""

import os
import re
import pandas as pd
from collections import defaultdict
import glob
from typing import Dict, List, Set, Tuple
import argparse

# Define paths - modify these to match your actual structure
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Primary data location - this is where we'll look first
DATASET_PATH = os.path.join(DATA_DIR, "full", "more_than_10", "SELECTED")

# Fallback to test directory if the main dataset doesn't exist
TEST_DIR = os.path.join(DATA_DIR, "test")

# Define expected structure and patterns
COMPOSER_CODE_PATTERN = re.compile(r'^[A-Z]{3}$')  # Three uppercase letters
FILE_PATTERN = re.compile(r'.*\.krn$')  # .krn files

DEBUG = False
MAX_EXAMPLES = 5  # Maximum number of examples to show in reports


# Define consistency checks
class ConsistencyChecker:
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path
        self.composers = set()
        self.all_files = []
        self.placeholder_files = []  # Track placeholder files
        self.consistency_issues = defaultdict(list)
        
    def find_all_composers(self):
        """Find all composer directories following the pattern."""
        composer_dirs = []
        
        # Also track lowercase versions for matching
        composer_codes_upper = set()
        
        if DEBUG:
            print(f"Looking for files in: {self.dataset_path}")
            print(f"Directory exists: {os.path.exists(self.dataset_path)}")
            if os.path.exists(self.dataset_path):
                print(f"Directory contents: {os.listdir(self.dataset_path)}")
        
        # First attempt: Try to find composer directories based on structure
        if os.path.exists(self.dataset_path):
            for root, dirs, files in os.walk(self.dataset_path):
                # Find any directories with composer-like names
                for dir_name in dirs:
                    if COMPOSER_CODE_PATTERN.match(dir_name):
                        composer_dirs.append(os.path.join(root, dir_name))
                        self.composers.add(dir_name)
                        composer_codes_upper.add(dir_name.upper())
                
                # Also look for .krn files directly
                krn_files = [f for f in files if f.endswith('.krn')]
                if krn_files:
                    composer_dirs.append(root)
                    # Try to extract composer code from filenames
                    for file in krn_files:
                        match = re.match(r'([A-Za-z]{3})\d+.*\.krn', file)
                        if match:
                            code = match.group(1).upper()
                            self.composers.add(code)
                            composer_codes_upper.add(code)
                            
        # Store uppercase versions of all composer codes for comparison
        self.composer_codes_upper = composer_codes_upper
                            
        # If no structured directories found, just find all directories with .krn files
        if not composer_dirs:
            for root, dirs, files in os.walk(self.dataset_path):
                krn_files = [f for f in files if f.endswith('.krn')]
                if krn_files:
                    composer_dirs.append(root)
                    
        print(f"Found {len(composer_dirs)} directories with potential .krn files")
        return composer_dirs
    
    def collect_all_files(self, composer_dirs):
        """Collect all .krn files from composer directories."""
        # Clear existing files list
        self.all_files = []
        
        # If no composer directories were found, scan the entire dataset path
        if not composer_dirs:
            for root, dirs, files in os.walk(self.dataset_path):
                self.all_files.extend([
                    os.path.join(root, f) for f in files if f.endswith('.krn')
                ])
        else:
            # Otherwise scan the identified composer directories
            for composer_dir in composer_dirs:
                if os.path.isdir(composer_dir):
                    for root, dirs, files in os.walk(composer_dir):
                        self.all_files.extend([
                            os.path.join(root, f) for f in files if f.endswith('.krn')
                        ])
                elif os.path.isfile(composer_dir) and composer_dir.endswith('.krn'):
                    # Handle case where composer_dir is actually a file
                    self.all_files.append(composer_dir)
                
        print(f"Found {len(self.all_files)} .krn files")
        if DEBUG:
            if self.all_files:
                print(f"Example files: {[os.path.basename(f) for f in self.all_files[:3]]}")
        return self.all_files
    
    '''Checks'''

    def check_file_naming_convention(self, files):
        """Check if file names follow expected convention."""
        for file_path in files:
            filename = os.path.basename(file_path)
            # Extract composer code from file name (assuming convention: ABR####.krn)
            match = re.match(r'([A-Za-z]{3})(\d+[a-z]?).*\.krn', filename)
            if not match:
                self.consistency_issues["file_naming"].append(
                    f"File doesn't match naming pattern: {filename}"
                )
                continue
                
            composer_code = match.group(1).upper()  # Convert to uppercase for consistent comparison
            
            # Check if composer code exists in composers set (case-insensitive)
            if composer_code not in self.composers and \
               not any(code.upper() == composer_code for code in self.composers):
                self.consistency_issues["file_naming"].append(
                    f"File {filename} has composer code {composer_code} not matching any known composer directory"
                )
    
    def check_file_header_consistency(self, files):
        """Check consistency of headers across files."""
        # Define required headers to check for
        required_headers = {'JRPID', 'VOICES'}
        headers_by_composer = defaultdict(lambda: defaultdict(set))
        
        for file_path in files:
            composer_code = None
            file_headers = set()
            
            # Extract composer code from filename or path
            filename = os.path.basename(file_path)
            match = re.match(r'([A-Za-z]{3})\d+.*\.krn', filename)
            if match:
                composer_code = match.group(1).upper()  # Convert to uppercase for comparison
            else:
                # Try to extract from path
                composer_code = None
                for part in os.path.normpath(file_path).split(os.sep):
                    if COMPOSER_CODE_PATTERN.match(part):
                        composer_code = part.upper()  # Convert to uppercase
                        break
            
            if not composer_code:
                self.consistency_issues["header_consistency"].append(
                    f"Could not determine composer code for {file_path}"
                )
                continue
            
            # Read file headers
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.startswith('!!!'):
                            break
                        
                        # Check for any required header appearing in the line
                        for header in required_headers:
                            # Case-insensitive check for header in the line
                            if f'!!!{header.lower()}:' in line.lower():
                                file_headers.add(header)
                        
                        # Also capture any other headers for consistency checking
                        header_match = re.match(r'!!!([A-Za-z]+):', line)
                        if header_match:
                            header_name = header_match.group(1).upper()
                            file_headers.add(header_name)
                            
                # Check for missing required headers
                missing_headers = required_headers - file_headers
                if missing_headers:
                    self.consistency_issues["missing_headers"].append(
                        f"File {filename} missing required headers: {', '.join(missing_headers)}"
                    )
                
                # Store headers for composer-level consistency check
                for header in file_headers:
                    headers_by_composer[composer_code][header].add(filename)
                    
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {filename}: {str(e)}"
                )
        
        # Check for header consistency within composers
        for composer, headers in headers_by_composer.items():
            all_files_count = len(set([f for files in headers.values() for f in files]))
            for header, files in headers.items():
                if len(files) < all_files_count:
                    self.consistency_issues["header_consistency"].append(
                        f"Header {header} only present in {len(files)}/{all_files_count} files for composer {composer}"
                    )
    
    def check_time_signature_consistency(self, files):
        """Check if time signatures are properly formatted."""
        for file_path in files:
            filename = os.path.basename(file_path)
            time_sigs = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Skip *MM lines (metronome markings) which start with *M but aren't time signatures
                        if line.startswith('*M') and not line.startswith('*MM'):
                            time_sigs.append(line.strip())
                            # Check if each time signature in the line is properly formatted
                            tokens = line.strip().split('\t')
                            for token in tokens:
                                if token.startswith('*M') and not token.startswith('*MM'):
                                    time_sig = token[2:]  # Remove '*M' prefix
                                    if not (re.match(r'^\d+/\d+$', time_sig) or  # e.g. "2/1"
                                           re.match(r'^\d+$', time_sig) or     # e.g. "3"
                                           re.match(r'^\d+/\d+%\d+$', time_sig) or  # e.g. "3/3%2"
                                           time_sig == '*'):                   # Copy previous
                                        self.consistency_issues["time_signature"].append(
                                            f"Invalid time signature format in {filename}: {token}"
                                        )
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {filename} for time signature check: {str(e)}"
                )
    
    def check_voice_consistency(self, files):
        """Check that voice counts match the specified number."""
        for file_path in files:
            filename = os.path.basename(file_path)
            declared_voices = None
            actual_voices = None
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Check for voice count declaration
                        if line.startswith('!!!voices:'):
                            try:
                                declared_voices = int(line.split(':')[1].strip())
                            except ValueError:
                                self.consistency_issues["voice_count"].append(
                                    f"Invalid voice count format in {filename}: {line.strip()}"
                                )
                            
                        # Count actual voices by checking **kern occurrences in a line
                        if line.startswith('**'):
                            actual_voices = line.count('**kern')
                            break
                            
                # Compare declared vs actual voice count
                if declared_voices is not None and actual_voices is not None:
                    if declared_voices != actual_voices:
                        self.consistency_issues["voice_count"].append(
                            f"Voice count mismatch in {filename}: declared={declared_voices}, actual={actual_voices}"
                        )
                            
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {filename} for voice consistency check: {str(e)}"
                )
    
    def check_percent_sign_lines(self, files):
        """Check for lines containing percent signs (special mensuration)."""
        for file_path in files:
            filename = os.path.basename(file_path)
            percent_lines = []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if '%' in line:
                            percent_lines.append(i)
                            
                if percent_lines:
                    self.consistency_issues["percent_signs"].append(
                        f"File {filename} has % signs on lines: {percent_lines}"
                    )
                            
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {filename} for percent sign check: {str(e)}"
                )
    
    def identify_placeholder_files(self, files):
        """Identify old placeholder files (files with less than 3 lines)."""
        valid_files = []
        
        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read up to 3 lines to check if file is a placeholder
                    lines = [line for _, line in zip(range(3), f)]
                    if len(lines) < 3:
                        self.placeholder_files.append(file_path)
                        self.consistency_issues["placeholder_files"].append(
                            f"File {filename} is a placeholder file with only {len(lines)} lines"
                        )
                    else:
                        valid_files.append(file_path)
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {filename}: {str(e)}"
                )
                # If we can't read the file, don't include it in further checks
                
        print(f"Found {len(self.placeholder_files)} placeholder files (excluded from further checks)")
        return valid_files
    
    def check_voice_indicator_format(self, files):
        """Check that voice indicators follow the standard format '!!!voices: N' where N is an integer."""
        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.startswith('!!!voices:'):
                            # Extract everything after the colon and strip whitespace
                            voice_value = line.split(':', 1)[1].strip()
                            
                            # Check if it's a simple integer
                            if not re.match(r'^\d+$', voice_value):
                                self.consistency_issues["voice_format"].append(
                                    f"Non-standard voice indicator in {filename} (line {line_num}): {line.strip()}. "
                                )
            except Exception as e:
                self.consistency_issues["file_reading"].append(f"Error reading {filename} for voice indicator check: {str(e)}")
    
    def check_for_duplicate_voice_declarations(self, files):
        """Check for files that declare the voices metadata header multiple times."""
        for file_path in files:
            filename = os.path.basename(file_path)
            voice_declaration_lines = []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.lower().startswith('!!!voices:'):
                            voice_declaration_lines.append((line_num, line.strip()))
                
                if len(voice_declaration_lines) > 1:
                    declaration_info = ", ".join([f"line {line_num}: '{text}'" for line_num, text in voice_declaration_lines])
                    self.consistency_issues["duplicate_voice_declarations"].append(
                        f"File {filename} has {len(voice_declaration_lines)} voice declarations: {declaration_info}"
                    )
                    
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {filename} for duplicate voice declarations check: {str(e)}")
    
    def check_voice_indicator_lines(self, files):
        """Check for multiple lines that start with *Ivo, *I", or *I' in each file."""
        for file_path in files:
            filename = os.path.basename(file_path)
            voice_indicator_lines = []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        # Check for lines that start with voice indicators
                        if line.startswith('*Ivo') or line.startswith('*I"') or line.startswith("*I'"):
                            voice_indicator_lines.append((line_num, line.strip()))
                
                # Group the indicators by type for better reporting
                by_type = defaultdict(list)
                for line_num, line in voice_indicator_lines:
                    if line.startswith('*Ivo'):
                        by_type['*Ivo'].append((line_num, line))
                    elif line.startswith('*I"'):
                        by_type['*I"'].append((line_num, line))
                    elif line.startswith("*I'"):
                        by_type["*I'"].append((line_num, line))
                
                # Check for issues: each type should appear at most once per file
                for indicator_type, occurrences in by_type.items():
                    if len(occurrences) > 1:
                        # More than one line of this type in the file
                        details = ", ".join([f"line {ln}: '{text}'" for ln, text in occurrences[:MAX_EXAMPLES]])
                        if len(occurrences) > MAX_EXAMPLES:
                            details += f", ... and {len(occurrences) - MAX_EXAMPLES} more"
                        
                        self.consistency_issues["voice_indicators"].append(
                            f"File {filename} has {len(occurrences)} '{indicator_type}' indicator lines (should have at most 1): {details}"
                        )
                    
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {filename} for voice indicator check: {str(e)}"                )
    
    def check_for_single_voice_files(self, files):
        """Identify files that contain only a single voice."""
        for file_path in files:
            filename = os.path.basename(file_path)
            has_single_voice = False
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    declared_voices = None
                    actual_voices = None
                    
                    for line in f:
                        # Check for voice count declaration
                        if line.startswith('!!!voices:'):
                            try:
                                declared_voices = int(line.split(':')[1].strip())
                                if declared_voices == 1:
                                    has_single_voice = True
                            except ValueError:
                                # Already handled by check_voice_indicator_format
                                pass
                            
                        # Check actual voices in the data
                        if line.startswith('**'):
                            actual_voices = line.count('**kern')
                            if actual_voices == 1:
                                has_single_voice = True
                            break
                    
                    if has_single_voice:
                        declared_info = f"declared={declared_voices}" if declared_voices is not None else "declaration not found"
                        actual_info = f"actual={actual_voices}" if actual_voices is not None else "actual count not found"
                        self.consistency_issues["single_voice_files"].append(
                            f"File {filename} contains only a single voice ({declared_info}, {actual_info})"
                        )
                        
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {filename} for single voice check: {str(e)}"
                )
        
    def run_all_checks(self):
        """Run all consistency checks."""
        composer_dirs = self.find_all_composers()
        all_files = self.collect_all_files(composer_dirs)
        
        # First identify and filter out placeholder files
        valid_files = self.identify_placeholder_files(all_files)
        
        print(f"Running consistency checks on {len(valid_files)} valid files...")
        
        self.check_file_naming_convention(valid_files)
        self.check_file_header_consistency(valid_files)
        self.check_time_signature_consistency(valid_files)
        self.check_voice_consistency(valid_files)
        self.check_voice_indicator_format(valid_files)
        self.check_voice_indicator_lines(valid_files)  # Add new check
        self.check_for_duplicate_voice_declarations(valid_files)
        self.check_percent_sign_lines(valid_files)
        self.check_for_single_voice_files(valid_files)  # Add new check
        
        return self.consistency_issues

    def generate_report(self):
        """Generate a report of all consistency issues."""
        if not self.consistency_issues:
            self.run_all_checks()
            
        print("\n=== CONSISTENCY CHECK REPORT ===\n")
        
        # Define a priority order for reporting issues
        # Critical issues first, informational issues last
        priority_order = [
            "placeholder_files",
            "duplicate_voice_declarations",
            "voice_indicators",
            "voice_format",
            "single_voice_files",
            "voice_count",
            "missing_headers",
            "file_naming",
            "header_consistency",
            "time_signature",
            "file_reading",
            "percent_signs",  # Informational only
            
        ]
        
        # Check if there are any issues at all
        if not any(self.consistency_issues.values()):
            print("✅ No consistency issues found!")
            return
        
        # Count total issues for the summary at the end
        critical_issues = 0
        informational_issues = 0
        
        # First report critical issues by priority
        print("\n=== CRITICAL ISSUES ===")
        has_critical_issues = False
        
        for check_type in priority_order:
            # Skip percent_signs as they're informational
            if check_type == "percent_signs":
                continue
                
            issues = self.consistency_issues.get(check_type, [])
            if issues:
                has_critical_issues = True
                print(f"\n--- {check_type.replace('_', ' ').title()} ({len(issues)} issues) ---")
                for issue in issues[:MAX_EXAMPLES]:
                    print(f"  • {issue}")
                
                if len(issues) > MAX_EXAMPLES:
                    print(f"  ... and {len(issues) - MAX_EXAMPLES} more issues")
                    
                critical_issues += len(issues)
            else:
                # Report zero issues for all check types
                check_name = check_type.replace('_', ' ').title()
                print(f"\n--- {check_name} (0 issues) ✅ ---")
        
        if not has_critical_issues:
            print("✅ No critical issues found!")
        
        # Then report informational issues (percent signs)
        print("\n=== INFORMATIONAL NOTICES ===")
        percent_issues = self.consistency_issues.get("percent_signs", [])
        if percent_issues:
            print(f"\n--- Percent Signs (Mensuration Symbols) ({len(percent_issues)} files) ---")
            for issue in percent_issues[:MAX_EXAMPLES]:  # Use MAX_EXAMPLES instead of hardcoded 10
                print(f"  • {issue}")
            
            if len(percent_issues) > MAX_EXAMPLES:
                print(f"  ... and {len(percent_issues) - MAX_EXAMPLES} more files")
                
            informational_issues = len(percent_issues)
        else:
            print("✅ No informational notices found!")
                
        # Report complete summary count
        print(f"\n=== SUMMARY ===")
        print(f"• Critical issues requiring attention: {critical_issues}")
        print(f"• Informational notices: {informational_issues}")
        #print(f"• Total files checked: {len(self.all_files)}")
        print(f"• Placeholder files found: {len(self.placeholder_files)}")
        print(f"• Valid files processed: {len(self.all_files) - len(self.placeholder_files)}")

def main():
    parser = argparse.ArgumentParser(description='Check consistency of CounterpointConsensus dataset')
    parser.add_argument('--path', help=f'Path to dataset directory (default: {DATASET_PATH})')
    args = parser.parse_args()
    
    # Use the specified path or default path
    if args.path:
        dataset_path = args.path
    else:
        # Check if the default path exists and has .krn files
        if os.path.exists(DATASET_PATH) and os.path.isdir(DATASET_PATH):
            try:
                # Check if there are any .krn files in this directory (or subdirectories)
                has_kern_files = any(f.endswith('.krn') for f in os.listdir(DATASET_PATH)) or \
                                any(f.endswith('.krn') for path, _, files in os.walk(DATASET_PATH) for f in files)
                if has_kern_files:
                    dataset_path = DATASET_PATH
                    print(f"Using default dataset path: {dataset_path}")
                else:
                    # No .krn files in default path, try test directory
                    dataset_path = TEST_DIR
                    print(f"No .krn files found in default path, using test directory: {dataset_path}")
            except Exception:
                # If there's any error accessing the default path, fall back to test directory
                dataset_path = TEST_DIR
                print(f"Error accessing default path, using test directory: {dataset_path}")
        else:
            # Default path doesn't exist, use test directory
            dataset_path = TEST_DIR
            print(f"Default path doesn't exist, using test directory: {dataset_path}")
    
    print()
    print(f"Checking dataset at: {dataset_path}")
    checker = ConsistencyChecker(dataset_path)
    checker.run_all_checks()
    checker.generate_report()

if __name__ == "__main__":
    main()

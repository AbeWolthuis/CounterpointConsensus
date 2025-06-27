#!/usr/bin/env python3
"""
Data Consistency Checker for Counterpoint Corpus (updated for critical vs informational distinction)

This script validates the consistency of **kern files in the CounterpointConsensus
dataset.  All functions run by `run_critical_checks()` now accept an optional
`file_failures` argument so they can tag files that fail a critical test.  
Only files with **no** critical failures are passed on to the informational
checks.
"""

import os
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# ─────────────────────────────── Paths & constants ──────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT_DIR, "data")

DATASET_PATH = os.path.join(DATA_DIR, "full", "more_than_10", "SELECTED")
TEST_DIR     = os.path.join(DATA_DIR, "test")           # fallback

COMPOSER_CODE_PATTERN = re.compile(r'^[A-Z]{3}$')       # three upper‑case letters
FILE_PATTERN          = re.compile(r'.*\.krn$')         # *.krn files

DEBUG        = False
MAX_EXAMPLES = 30


# ═════════════════════════════════ Main checker class ═══════════════════════════
class ConsistencyChecker:
    # ───────────────────────────── initialisation ──────────────────────────────
    def __init__(self, dataset_path: str = DATASET_PATH):
        self.dataset_path: str                    = dataset_path
        self.composers:          Set[str]         = set()
        self.all_files:          List[str]        = []
        self.placeholder_files:  List[str]        = []
        self.consistency_issues: Dict[str, List[str]] = defaultdict(list)

        # tracking for the new critical/informational split
        self.critical_failures: Dict[str, List[str]] = {}   # file → [reasons]
        self.valid_files:      Set[str]            = set()  # files with 0 critical failures

        # Initialize time signature related counts here
        self.time_signature_counts = defaultdict(int)
        self.mensuration_counts = defaultdict(int)
        self.ts_mensuration_pairs = defaultdict(lambda: defaultdict(int))
        self.mensuration_position = {"before": 0, "after": 0, "both": 0, "none": 0}
        # Note: signature_examples is local to the check function, no need to init here

    # ────────────────────────────── discovery ──────────────────────────────────
    def find_all_composers(self) -> List[str]:
        """Return a list of directories that appear to belong to individual composers."""
        composer_dirs: List[str] = []
        composer_codes_upper: Set[str] = set()

        if DEBUG:
            print(f"[DEBUG] Searching: {self.dataset_path} (exists={os.path.exists(self.dataset_path)})")

        if os.path.exists(self.dataset_path):
            for root, dirs, files in os.walk(self.dataset_path):
                # pick up composer directories named “ABC”
                for d in dirs:
                    if COMPOSER_CODE_PATTERN.match(d):
                        composer_dirs.append(os.path.join(root, d))
                        self.composers.add(d)
                        composer_codes_upper.add(d.upper())

                # extract composer code from lone *.krn files in otherwise flat dirs
                krn_files = [f for f in files if f.endswith(".krn")]
                if krn_files:
                    composer_dirs.append(root)
                    for f in krn_files:
                        m = re.match(r'([A-Za-z]{3})\d+.*\.krn', f)
                        if m:
                            code = m.group(1).upper()
                            self.composers.add(code)
                            composer_codes_upper.add(code)

        # if none found, fall back to scanning every directory for krn files
        if not composer_dirs:
            for root, _, files in os.walk(self.dataset_path):
                if any(f.endswith(".krn") for f in files):
                    composer_dirs.append(root)

        self.composer_codes_upper = composer_codes_upper
        print(f"Found {len(composer_dirs)} directories with potential .krn files")
        return composer_dirs

    def collect_all_files(self, composer_dirs: List[str]) -> List[str]:
        """Gather every *.krn file in the candidate directories."""
        self.all_files.clear()

        if not composer_dirs:
            # no dedicated composer dirs – scan the entire dataset
            for root, _, files in os.walk(self.dataset_path):
                self.all_files += [os.path.join(root, f) for f in files if f.endswith(".krn")]
        else:
            for d in composer_dirs:
                if os.path.isdir(d):
                    for root, _, files in os.walk(d):
                        self.all_files += [os.path.join(root, f) for f in files if f.endswith(".krn")]
                elif d.endswith(".krn"):
                    self.all_files.append(d)

        print(f"Found {len(self.all_files)} .krn files")
        if DEBUG and self.all_files:
            print(f"[DEBUG] Example files: {[os.path.basename(f) for f in self.all_files[:3]]}")
        return self.all_files

    # ──────────────────────────── critical checks ──────────────────────────────
    # NOTE: every critical‑check method now accepts `file_failures` to note
    #       per‑file reasons for invalidation.  Add to it only for *critical*
    #       failures – informational findings are handled separately.

    # ✓ already updated by the user
    def check_file_naming_convention(self, files: List[str], file_failures=None):
        """Verify that filenames follow the ABC####.krn convention and map to a known composer."""
        for fp in files:
            fn = os.path.basename(fp)
            m = re.match(r'([A-Za-z]{3})(\d+[a-z]?).*\.krn', fn)
            if not m:
                self.consistency_issues["file_naming"].append(
                    f"File doesn't match naming pattern: {fn}"
                )
                if file_failures is not None:
                    file_failures[fp].append("file_naming")
                continue

            composer_code = m.group(1).upper()
            if composer_code not in self.composers and \
               not any(c.upper() == composer_code for c in self.composers):
                self.consistency_issues["file_naming"].append(
                    f"File {fn} has composer code {composer_code} not recognised"
                )
                if file_failures is not None:
                    file_failures[fp].append("file_naming")

    def check_file_header_consistency(self, files: List[str], file_failures=None):
        """Ensure that mandatory reference records appear and are consistent within each composer."""
        mandatory = {'JRPID', 'VOICES'}
        headers_by_composer: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        for fp in files:
            fn = os.path.basename(fp)
            # composer code from filename or path
            m = re.match(r'([A-Za-z]{3})\d+.*\.krn', fn)
            composer = m.group(1).upper() if m else next(
                (part.upper() for part in os.path.normpath(fp).split(os.sep)
                 if COMPOSER_CODE_PATTERN.match(part)), None)

            if not composer:
                self.consistency_issues["header_consistency"].append(
                    f"Cannot determine composer code for {fp}"
                )
                if file_failures is not None:
                    file_failures[fp].append("header_consistency")
                continue

            present_headers: Set[str] = set()
            try:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if not line.startswith("!!!"):
                            break
                        # case‑insensitive search for mandatory headers
                        for h in mandatory:
                            if line.lower().startswith(f"!!!{h.lower()}:"):
                                present_headers.add(h)
                        m_hdr = re.match(r'!!!([A-Za-z]+):', line)
                        if m_hdr:
                            headers_by_composer[composer][m_hdr.group(1).upper()].add(fn)

                missing = mandatory - present_headers
                if missing:
                    # Treat missing metadata headers as informational only
                    self.consistency_issues["missing_headers"].append(
                        f"File {fn} missing: {', '.join(sorted(missing))}"
                    )
                    # do NOT mark as critical failure:
                    # if file_failures is not None:
                    #     file_failures[fp].append("missing_headers")

            except Exception as e:
                self.consistency_issues["file_reading"].append(f"Error reading {fn}: {e}")
                if file_failures is not None:
                    file_failures[fp].append("file_reading")

        # composer‑level header consistency (informative but still critical)
        for comp, hmap in headers_by_composer.items():
            all_files = {f for files in hmap.values() for f in files}
            total = len(all_files)
            for hdr, files_with_hdr in hmap.items():
                if len(files_with_hdr) < total:
                    self.consistency_issues["header_consistency"].append(
                        f"Composer {comp}: header {hdr} appears in "
                        f"{len(files_with_hdr)}/{total} files"
                    )
                    # do NOT mark missing headers here as critical either
                    # if file_failures is not None:
                    #     for lf in lacking:
                    #         file_failures[lf].append("header_consistency")

    def check_time_signature_consistency(self, files: List[str], file_failures=None):
        """Validate *M time‑signature tokens."""
        ts_re_simple   = re.compile(r'^\d+/\d+$')      # e.g. 4/2
        ts_re_integer  = re.compile(r'^\d+$')          # single integer (mensural)
        ts_re_complex  = re.compile(r'^\d+/\d+%\d+$')  # e.g. 3/3%2

        for fp in files:
            fn = os.path.basename(fp)
            try:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("*M") and not line.startswith("*MM"):
                            for token in line.rstrip().split('\t'):
                                if token.startswith("*M") and not token.startswith("*MM"):
                                    sig = token[2:]
                                    ok = (
                                        ts_re_simple.match(sig) or
                                        ts_re_integer.match(sig) or
                                        ts_re_complex.match(sig) or
                                        sig == "*"
                                    )
                                    if not ok:
                                        self.consistency_issues["time_signature"].append(
                                            f"Invalid time sig in {fn}: {token}"
                                        )
                                        if file_failures is not None:
                                            file_failures[fp].append("time_signature")
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for time‑signature check: {e}"
                )
                if file_failures is not None:
                    file_failures[fp].append("file_reading")

    def check_voice_consistency(self, files: List[str], file_failures=None):
        """Ensure !!!voices: N matches the number of **kern spines on the first data line."""
        for fp in files:
            fn = os.path.basename(fp)
            declared = None
            actual = None
            try:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("!!!voices:"):
                            try:
                                declared = int(line.split(":", 1)[1].strip())
                            except ValueError:
                                self.consistency_issues["voice_count"].append(
                                    f"Invalid voices meta in {fn}: {line.strip()}"
                                )
                                if file_failures is not None:
                                    file_failures[fp].append("voice_count")
                        # Find the first interpretation line (starts with '**')
                        if line.startswith("**"):
                            # Count the number of spines that are exactly '**kern'
                            actual = sum(1 for token in line.strip().split('\t') if token == "**kern")
                            break
                if declared is not None and actual is not None and declared != actual:
                    self.consistency_issues["voice_count"].append(
                        f"Voice mismatch in {fn}: declared {declared}, found {actual}"
                    )
                    if file_failures is not None:
                        file_failures[fp].append("voice_count")
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for voice‑count check: {e}"
                )
                if file_failures is not None:
                    file_failures[fp].append("file_reading")

    def check_voice_indicator_format(self, files: List[str], file_failures=None):
        """Check that !!!voices: is an integer (not e.g. 'three')."""
        for fp in files:
            fn = os.path.basename(fp)
            try:
                with open(fp, encoding="utf-8") as f:
                    for ln, line in enumerate(f, 1):
                        if line.startswith("!!!voices:"):
                            # Extract everything after the colon and strip whitespace
                            voice_value = line.split(':', 1)[1].strip()
                            
                            # Check if it's a simple integer
                            if not re.match(r'^\d+$', voice_value):
                                self.consistency_issues["voice_format"].append(
                                    f"Non‑integer voices in {fn} (line {ln}): {line.strip()}"
                                )
                                if file_failures is not None:
                                    file_failures[fp].append("voice_format")
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for voice‑format check: {e}"
                )
                if file_failures is not None:
                    file_failures[fp].append("file_reading")

    def check_voice_indicator_lines(self, files: List[str], file_failures=None):
        """Each of *Ivo, *I", *I' may appear at most once."""
        for fp in files:
            fn = os.path.basename(fp)
            try:
                with open(fp, encoding="utf-8") as f:
                    occurrences = defaultdict(list)   # indicator → [(ln, text)]
                    for ln, line in enumerate(f, 1):
                        if line.startswith("*Ivo"):
                            occurrences["*Ivo"].append((ln, line.strip()))
                        elif line.startswith('*I"'):
                            occurrences['*I"'].append((ln, line.strip()))
                        elif line.startswith("*I'"):
                            occurrences["*I'"].append((ln, line.strip()))

                for ind, occ in occurrences.items():
                    if len(occ) > 1:
                        detail = ", ".join([f"line {ln}" for ln, _ in occ[:MAX_EXAMPLES]])
                        if len(occ) > MAX_EXAMPLES:
                            detail += f", … (+{len(occ)-MAX_EXAMPLES})"
                        self.consistency_issues["voice_indicators"].append(
                            f"{fn}: {ind} appears {len(occ)}× ({detail})"
                        )
                        if file_failures is not None:
                            file_failures[fp].append("voice_indicators")
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for voice‑indicator check: {e}"
                )
                if file_failures is not None:
                    file_failures[fp].append("file_reading")

    def check_for_duplicate_voice_declarations(self, files: List[str], file_failures=None):
        """Detect multiple !!!voices: lines."""
        for fp in files:
            fn = os.path.basename(fp)
            decl_lines: List[int] = []
            try:
                with open(fp, encoding="utf-8") as f:
                    for ln, line in enumerate(f, 1):
                        if line.lower().startswith("!!!voices:"):
                            decl_lines.append(ln)
                if len(decl_lines) > 1:
                    self.consistency_issues["duplicate_voice_declarations"].append(
                        f"{fn}: multiple !!!voices: ({', '.join(map(str, decl_lines))})"
                    )
                    if file_failures is not None:
                        file_failures[fp].append("duplicate_voice_declarations")
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for duplicate‑voices check: {e}"
                )
                if file_failures is not None:
                    file_failures[fp].append("file_reading")

    def check_for_single_voice_files(self, files: List[str], file_failures=None):
        for fp in files:
            fn = os.path.basename(fp)
            single = False
            declared = actual = None
            try:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("!!!voices:"):
                            try:
                                declared = int(line.split(":", 1)[1].strip())
                                if declared == 1:
                                    single = True
                            except ValueError:
                                pass
                        if line.startswith("**"):
                            actual = line.count("**kern")
                            if actual == 1:
                                single = True
                            break
                if single:
                    if file_failures is not None:
                        file_failures[fp].append(
                        f"{fn} is single‑voice (declared={declared}, actual={actual})"
                        )
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for single‑voice check: {e}"
                )
                if file_failures is not None:
                    file_failures[fp].append("file_reading")



    # ────────────────────────── informational checks ───────────────────────────
    def check_percent_sign_lines(self, files: List[str]):
        for fp in files:
            fn = os.path.basename(fp)
            try:
                with open(fp, encoding="utf-8") as f:
                    lines = [i for i, line in enumerate(f, 1) if '%' in line]
                if lines:
                    self.consistency_issues["percent_signs"].append(
                        f"{fn} contains % on lines: {lines[:MAX_EXAMPLES]}{' …' if len(lines) > MAX_EXAMPLES else ''}"
                    )
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for %‑sign check: {e}"
                )
    
    def check_for_tuplet_markers(self, files: List[str]):
        for fp in files:
            fn = os.path.basename(fp)
            markers: List[Tuple[int, str]] = []
            try:
                with open(fp, encoding="utf-8") as f:
                    for ln, line in enumerate(f, 1):
                        if line.startswith(("!", "*", "=")):
                            continue
                        if 'V' in line or 'Z' in line:
                            toks = [t for t in line.strip().split('\t')
                                    if t.endswith('V') or t.endswith('Z')]
                            if toks:
                                markers.append((ln, ", ".join(toks)))
                if markers:
                    first = "; ".join([f"line {ln}: {txt}" for ln, txt in markers[:MAX_EXAMPLES]])
                    if len(markers) > MAX_EXAMPLES:
                        first += f"; … (+{len(markers)-MAX_EXAMPLES})"
                    self.consistency_issues["tuplet_markers"].append(
                        f"{fn}: tuplet markers found ({first})"
                    )
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for tuplet‑marker check: {e}"
                )

    def check_time_signature_indications(self, files: List[str]):
        """
        Collect all time signature indications (*M tokens) across files and
        check if they have corresponding mensuration indicators (*met) in the line before OR after.
        Counts EVERY occurrence of each time signature, not just unique ones per file.
        """
        # Clear previous counts before recalculating for this run
        self.time_signature_counts.clear()
        self.mensuration_counts.clear()
        self.ts_mensuration_pairs.clear()
        self.mensuration_position = {"before": 0, "after": 0, "both": 0, "none": 0} # Re-init simple dict
        signature_examples = defaultdict(list) # Local variable, init here is fine
        
        for fp in files:
            fn = os.path.basename(fp)
            
            try:
                with open(fp, encoding="utf-8") as f:
                    lines = f.readlines()
                    
                    for line_num, line in enumerate(lines):
                        # Check for time signature tokens
                        if line.startswith("*M") and not line.startswith("*MM"):
                            # Get all time signatures in this line, not just unique ones
                            time_sig_tokens = []
                            for token in line.strip().split('\t'):
                                if token.startswith("*M") and not token.startswith("*MM"):
                                    time_sig_tokens.append(token)
                                    # Count each occurrence immediately
                                    self.time_signature_counts[token] += 1
                            
                            # For each time signature in this line (allows duplicates)
                            for sig in time_sig_tokens:
                                # Track unique mensuration tokens before and after
                                mensuration_before = set()
                                mensuration_after = set()
                                
                                # Check previous line for mensurations
                                if line_num > 0:
                                    prev_line = lines[line_num-1]
                                    for token in prev_line.strip().split('\t'):
                                        if token.startswith("*met"):
                                            mensuration_before.add(token)
                                
                                # Check next line for mensurations
                                if line_num < len(lines) - 1:
                                    next_line = lines[line_num+1]
                                    for token in next_line.strip().split('\t'):
                                        if token.startswith("*met"):
                                            mensuration_after.add(token)
                                
                                # Count each unique mensuration token only once per time signature occurrence
                                for met_token in mensuration_before:
                                    self.mensuration_counts[met_token] += 1
                                    self.ts_mensuration_pairs[sig][f"{met_token} (before)"] += 1
                                    
                                for met_token in mensuration_after:
                                    self.mensuration_counts[met_token] += 1
                                    self.ts_mensuration_pairs[sig][f"{met_token} (after)"] += 1
                                
                                # Track position statistics
                                if mensuration_before and mensuration_after:
                                    self.mensuration_position["both"] += 1
                                elif mensuration_before:
                                    self.mensuration_position["before"] += 1
                                elif mensuration_after:
                                    self.mensuration_position["after"] += 1
                                else:
                                    self.mensuration_position["none"] += 1
                                
                                # Store example for reporting
                                if len(signature_examples[sig]) < MAX_EXAMPLES:
                                    signature_examples[sig].append((fn, line_num+1))
                    
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for time signature indications check: {e}"
                )
        
        # Sort signatures by frequency for reporting
        sorted_sigs = sorted(self.time_signature_counts.items(), key=lambda x: (-x[1], x[0]))
        
        if sorted_sigs:
            self.consistency_issues["time_signature_indications"] = []
            self.consistency_issues["time_signature_indications"].append(
                f"Found {len(sorted_sigs)} unique time signature indications with {sum(self.time_signature_counts.values())} total occurrences:"
            )
            
            for sig, count in sorted_sigs:
                examples = signature_examples[sig]
                example_str = "; ".join([f"{f} (line {l})" for f, l in examples[:3]])
                if len(examples) > 3:
                    example_str += f"; and {len(examples) - 3} more"
                
                # Report on mensurations
                mensuration_info = ""
                paired_mensuration = self.ts_mensuration_pairs.get(sig, {})
                if paired_mensuration:
                    paired_str = ", ".join([f"{m} ({c})" for m, c in sorted(
                        paired_mensuration.items(), key=lambda x: (-x[1], x[0]))])
                    mensuration_info = f" [Mensurations: {paired_str}]"
                
                self.consistency_issues["time_signature_indications"].append(
                    f"  {sig}: appears {count} total times.{mensuration_info} Examples: {example_str}"
                )
        
        return self.time_signature_counts

    def check_for_rscale_tokens(self, files: List[str]):
        """
        Add an informational notice if any token in any line starts with '*rscale'.
        """
        for fp in files:
            fn = os.path.basename(fp)
            try:
                with open(fp, encoding="utf-8") as f:
                    for ln, line in enumerate(f, 1):
                        tokens = line.strip().split('\t')
                        for token in tokens:
                            if token.startswith("*rscale"):
                                self.consistency_issues["rscale_tokens"].append(
                                    f"{fn}: '*rscale' found on line {ln}"
                                )
                                # Only need to report once per file
                                raise StopIteration
            except StopIteration:
                continue
            except Exception as e:
                self.consistency_issues["file_reading"].append(
                    f"Error reading {fn} for rscale check: {e}"
                )

    # ─────────────────────────── placeholder filter ────────────────────────────
    def identify_placeholder_files(self, files: List[str], file_failures=None) -> List[str]:
        """Identify files with minimal data content (likely placeholders)."""
        valid: List[str] = []
        MIN_DATA_LINES = 4  # Threshold for considering a file non-placeholder

        for fp in files:
            fn = os.path.basename(fp)
            data_line_count = 0
            is_placeholder = False
            try:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        line_strip = line.strip()
                        # Count lines that are not comments, headers, interpretations, or spine manipulators
                        if line_strip and not line_strip.startswith(('!', '*', '=')):
                            data_line_count += 1
                
                if data_line_count < MIN_DATA_LINES:
                    is_placeholder = True
                    self.placeholder_files.append(fp)
                    self.consistency_issues["placeholder_files"].append(
                        f"{fn} is likely a placeholder (< {MIN_DATA_LINES} data lines)"
                    )
                    if file_failures is not None:
                        file_failures[fp].append("placeholder_file")
                else:
                    valid.append(fp)

            except Exception as e:
                self.consistency_issues["file_reading"].append(f"Error reading {fn} for placeholder check: {e}")
                if file_failures is not None:
                    # Consider read errors as critical failures too
                    file_failures[fp].append("file_reading_placeholder")

        print(f"Identified {len(self.placeholder_files)} potential placeholder files based on content.")
        return valid

    # ───────────────────────── driver routines ────────────────────────────────
    def run_critical_checks(self, files: List[str]):
        print(f"Running critical checks on {len(files)} files…")
        self.critical_failures.clear()
        self.valid_files.clear()

        file_failures: defaultdict[str, List[str]] = defaultdict(list)

        # Run placeholder check first as it's a fundamental validity issue
        files_after_placeholder_check = self.identify_placeholder_files(files, file_failures)

        # Run other Critical checks only on files that are not placeholders
        self.check_file_naming_convention(files_after_placeholder_check, file_failures)
        self.check_file_header_consistency(files_after_placeholder_check, file_failures)
        self.check_time_signature_consistency(files_after_placeholder_check, file_failures)
        self.check_voice_consistency(files_after_placeholder_check, file_failures)
        self.check_voice_indicator_format(files_after_placeholder_check, file_failures)
        self.check_voice_indicator_lines(files_after_placeholder_check, file_failures)
        self.check_for_duplicate_voice_declarations(files_after_placeholder_check, file_failures)
        self.check_for_single_voice_files(files_after_placeholder_check, file_failures)
        

        # classify all originally passed files
        for fp in files: # Iterate over the original list to capture placeholder failures too
            if fp in file_failures and file_failures[fp]:
                self.critical_failures[fp] = list(set(file_failures[fp])) # Ensure unique reasons
            elif fp in files_after_placeholder_check: # Only add if it passed placeholder and other checks
                 self.valid_files.add(fp)


        return self.valid_files, self.critical_failures

    def run_informational_checks(self, valid_files: List[str]):
        print(f"Running informational checks on {len(valid_files)} files…")
        self.check_percent_sign_lines(valid_files)
        
        self.check_for_tuplet_markers(valid_files)
        self.check_time_signature_indications(valid_files)
        self.check_for_rscale_tokens(valid_files)

    def run_all_checks(self):
        dirs  = self.find_all_composers()
        files = self.collect_all_files(dirs)
        # Placeholder check is now integrated into run_critical_checks

        valid, _ = self.run_critical_checks(files) # Pass the full list here
        self.run_informational_checks(list(valid)) # Convert set to list for iteration
        return self.consistency_issues, valid, self.critical_failures

    # ──────────────────────── convenience accessors ───────────────────────────
    def get_valid_files(self) -> List[str]:
        return list(self.valid_files)

    def get_invalid_files(self) -> List[str]:
        return list(self.critical_failures)
    
    def get_time_signature_counts(self) -> Dict[str, int]:
        """Returns the dictionary mapping time signatures to their occurrence counts."""
        # Assumes check_time_signature_indications has already been run by run_informational_checks
        return self.time_signature_counts

    # ────────────────────────────── reporting ─────────────────────────────────
    def generate_report(self):
        if not self.consistency_issues and not self.critical_failures:
            self.run_all_checks()

        print("\n=== CONSISTENCY CHECK REPORT ===\n")

        critical_categories = [
            "placeholder_files", "duplicate_voice_declarations", "voice_indicators",
            "voice_format", "voice_count", "file_naming",
            "header_consistency", "time_signature", "file_reading", "single_voice_files",
        ]

        informational_categories = [
            "percent_signs", "tuplet_markers", "time_signature_indications",
            "rscale_tokens", "missing_headers",
            ]

        crit_total = sum(len(self.consistency_issues[c]) for c in critical_categories)
        info_total = sum(len(self.consistency_issues[c]) for c in informational_categories)

        # Critical section
        print("─── CRITICAL ISSUES ───")
        for cat in critical_categories:
            issues = self.consistency_issues.get(cat, [])
            name   = cat.replace("_", " ").title()
            if issues:
                print(f"\n{name} ({len(issues)})")
                for i in issues[:MAX_EXAMPLES]:
                    print(f"  • {i}")
                more = len(issues) - MAX_EXAMPLES
                if more > 0:
                    print(f"  … +{more} more")
            else:
                print(f"{name}: 0 ✅")

        # Informational section
        print("\n─── INFORMATIONAL NOTICES ───")
        for cat in informational_categories:
            issues = self.consistency_issues.get(cat, [])
            name   = cat.replace("_", " ").title()
            if issues:
                print(f"\n{name} ({len(issues)})")
                for i in issues[:MAX_EXAMPLES]:
                    print(f"  • {i}")
                more = len(issues) - MAX_EXAMPLES
                if more > 0:
                    print(f"  … +{more} more")
            else:
                print(f"{name}: 0 ✅")

        print("\n")
        self.report_time_signatures()

        # Summary
        print("\n─── SUMMARY ───")
        print(f"Critical issues:        {crit_total}")
        print(f"Informational notices:  {info_total}")
        print(f"Total files checked:    {len(self.all_files)}")
        print(f"Invalid (critical‑fail):{len(self.critical_failures)}")
        print(f"Valid files:            {len(self.valid_files)}")


    def report_time_signatures(self):
        """Generate a specialized report of time signature usage with mensuration information."""
        time_sigs = self.get_time_signature_counts()
        
        if not hasattr(self, 'mensuration_counts'):
            # Ensure mensuration data is collected
            self.check_time_signature_indications(list(self.valid_files))
        
        if not time_sigs:
            print("No time signatures found.")
            return
        
        # Report on position statistics
        total_positions = sum(self.mensuration_position.values())
        if total_positions > 0:
            print("\n--- Mensuration Position Relative to Time Signatures ---")
            print(f"Before only: {self.mensuration_position['before']} ({self.mensuration_position['before']/total_positions*100:.1f}%)")
            print(f"After only:  {self.mensuration_position['after']} ({self.mensuration_position['after']/total_positions*100:.1f}%)")
            print(f"Both:        {self.mensuration_position['both']} ({self.mensuration_position['both']/total_positions*100:.1f}%)")
            print(f"None:        {self.mensuration_position['none']} ({self.mensuration_position['none']/total_positions*100:.1f}%)")
        
        # Sort signatures by frequency for reporting
        sorted_sigs = sorted(time_sigs.items(), key=lambda x: (-x[1], x[0]))
        
        # Calculate column widths
        max_sig_width = max(len(sig) for sig in time_sigs.keys())
        max_count_width = max(len(str(count)) for count in time_sigs.values())
        
        # Print header
        pad_time_sig_column_len = 5
        print(f"\n{'Time Signature':<{max_sig_width+2}} | {'Count':<{max_count_width+2}} | {'%':<5} | {'Common Mensurations'}")
        print(f"{'-'*(max_sig_width+2) + '-'*pad_time_sig_column_len}-+-{'-'*(max_count_width+2)}-+-------+----------------")
        
        # Print rows
        total_occurrences = sum(time_sigs.values())
        for sig, count in sorted_sigs:
            percentage = (count / total_occurrences) * 100
            
            # Get associated mensurations
            paired_mensurations = self.ts_mensuration_pairs.get(sig, {})
            if paired_mensurations:
                # Clean and aggregate mensuration tokens by removing position indicators and prefixes
                clean_paired = defaultdict(int)
                for token, mensuration_count in paired_mensurations.items():
                    # Remove the position indicator "(before)" or "(after)"
                    clean_token = token.split(" (")[0]
                    # Remove "*met" prefix to save space
                    if clean_token.startswith("*met"):
                        clean_token = clean_token[4:]
                    clean_paired[clean_token] += mensuration_count
                    
                # Sort by frequency (highest first) and create comma-separated text of ALL mensurations
                sorted_mens = sorted(clean_paired.items(), key=lambda x: (-x[1], x[0]))
                mensuration_text = ", ".join([f"{m}({c})" for m, c in sorted_mens])
            else:
                mensuration_text = "None found"
                
            print(f"{sig:<{max_sig_width+2+pad_time_sig_column_len}} | {count:<{max_count_width+2}} | {percentage:5.1f}% | {mensuration_text}")
        
        # Also show mensuration statistics, grouping by base token without position info
        if self.mensuration_counts:
            print("\n--- Mensuration Indicators ---")
            
            # Clean up position information for display
            clean_mensuration_counts = defaultdict(int)
            for token, count in self.mensuration_counts.items():
                # Clean token - remove "*met" prefix
                clean_token = token
                if clean_token.startswith("*met"):
                    clean_token = clean_token[4:]
                clean_mensuration_counts[clean_token] += count
                
            sorted_mens = sorted(clean_mensuration_counts.items(), key=lambda x: (-x[1], x[0]))
            
            max_men_width = max(len(m) for m in clean_mensuration_counts.keys())
            print(f"{'Mensuration':<{max_men_width+2}} | {'Count':<{max_count_width+2}} | {'%':<5}")
            print(f"{'-'*(max_men_width+2)}-+-{'-'*(max_count_width+2)}-+------")
            
            total_mens = sum(clean_mensuration_counts.values())
            for men, count in sorted_mens:
                percentage = (count / total_mens) * 100
                print(f"{men:<{max_men_width+2}} | {count:<{max_count_width+2}} | {percentage:5.1f}%")
        
        print("\nTime signatures as a set:")
        print(f"[{', '.join(repr(sig) for sig, _ in sorted_sigs)}]")
        
        if self.mensuration_counts:
            print("\nMensuration indicators as a set:")
            # Clean tokens for set representation
            clean_tokens = set()
            for token in self.mensuration_counts.keys():
                clean_token = token
                if clean_token.startswith("*met"):
                    clean_token = clean_token[4:]
                clean_tokens.add(clean_token)
            print(f"[{', '.join(repr(token) for token in sorted(clean_tokens))}]")
    
# ═════════════════════════════════ ═ CLI ═ ═══════════════════════════════════════
def main():
    # --- REMOVE argparse and CLI parsing ---
    # parser = argparse.ArgumentParser(description="Consistency checker for CounterpointConsensus")
    # parser.add_argument("--path", help=f"Dataset path (default {DATASET_PATH})")
    # parser.add_argument("--output", help="Directory to write valid/invalid lists")
    # args = parser.parse_args()

    # --- Hardcode dataset path logic (use DATASET_PATH or fallback to TEST_DIR) ---
    def has_kerns(p):
        return any(f.endswith(".krn") for f in os.listdir(p)) or \
               any(f.endswith(".krn") for _, _, fs in os.walk(p) for f in fs)
    if os.path.isdir(DATASET_PATH) and has_kerns(DATASET_PATH):
        dataset_path = DATASET_PATH
        print(f"Using default dataset path {dataset_path}")
    else:
        dataset_path = TEST_DIR
        print(f"Default path lacks .krn files – using {dataset_path}")

    print(f"\nChecking dataset at {dataset_path}\n")
    checker = ConsistencyChecker(dataset_path)
    issues, valid_files_set, critical_failures_dict = checker.run_all_checks()
    checker.generate_report()

    # --- Supply a list of externally invalid JRP-IDs ---
    # Example: supply your own list here
    externally_invalid_jrpids = [
        "Oke1013e", "Jos0603a", 
    ]

    # --- Print valid and invalid file lists ---
    valid_files_list = checker.get_valid_files()
    invalid_files_list = checker.get_invalid_files()
    PRINT_VALID_FLAG = False

    # --- Collect JRP-IDs from invalid files ---
    invalid_ids = set()
    if invalid_files_list:
        for file_path in invalid_files_list:
            filename = os.path.basename(file_path)
            jrpid = filename.split('-', 1)[0]
            invalid_ids.add(jrpid)

    # --- Merge with externally supplied invalid JRP-IDs ---
    invalid_ids.update(externally_invalid_jrpids)

    # --- Remove any now-invalid files from the valid list ---
    filtered_valid_files_list = [
        fp for fp in valid_files_list
        if os.path.basename(fp).split('-', 1)[0] not in invalid_ids
    ]

    print(f"\n--- Valid Files (JRP-IDs) ---")
    if PRINT_VALID_FLAG:
        if filtered_valid_files_list:
            valid_ids = set()
            for file_path in filtered_valid_files_list:
                filename = os.path.basename(file_path)
                jrpid = filename.split('-', 1)[0]
                valid_ids.add(jrpid)
            sorted_valid_ids = sorted(list(valid_ids))
            print(sorted_valid_ids)
        else:
            print("[] # No files passed all critical checks.")

    print(f"\n--- Invalid Files (JRP-IDs); {len(invalid_ids)} files ---")
    sorted_invalid_ids = sorted(list(invalid_ids))
    print(sorted_invalid_ids)   

if __name__ == "__main__":
    main()

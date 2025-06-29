import re
import difflib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FieldConfidence(Enum):
    HIGH = "high"  # 0.8+
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"  # <0.5


@dataclass
class ExtractionResult:
    """
    Represents the result of the NID extraction process.
    """
    name: str
    dob: str
    nid: str
    name_confidence: float
    dob_confidence: float
    nid_confidence: float
    extraction_method: str
    raw_text: str


class EnhancedNIDExtractor:
    """
    A class to extract Name, Date of Birth, and NID from OCR text
    with enhanced robustness against common OCR errors and formatting variations.
    """

    def __init__(self):
        # Enhanced regex patterns based on OCR data analysis
        self.date_patterns = [
            # Standard formats with flexible spacing
            r'\b(\d{1,2})\s*[-/\s]*([A-Za-z]{3,9})\s*[-/\s]*(\d{4})\b',  # e.g., 11 May 1965, 10May1987
            r'\b(\d{1,2})\s*[-/\s]*(\d{1,2})\s*[-/\s]*(\d{4})\b',  # e.g., 11/05/1965, 11-05-1965
            r'\b(\d{4})\s*[-/\s]*(\d{1,2})\s*[-/\s]*(\d{1,2})\b',  # e.g., 1965-05-11

            # Concatenated date formats (OCR common errors)
            r'\b(\d{1,2})([A-Za-z]{3,9})(\d{4})\b',  # e.g., 10May1987, 15Apr2000
            r'\b(\d{1,2})([A-Za-z]{3,9})\s+(\d{4})\b',  # e.g., 10May 1987 (space after month)
            r'\b(\d{1,2})\s+([A-Za-z]{3,9})(\d{4})\b',  # e.g., 10 May1987 (space before month)
        ]

        # NID patterns - handle various formats and OCR errors
        self.nid_patterns = [
            # Continuous digits (10, 13, 17 digits)
            r'\b(\d{10})\b',  # 10-digit NID
            r'\b(\d{13})\b',  # 13-digit NID
            r'\b(\d{17})\b',  # 17-digit NID
            r'\b(\d{10,17})\b',  # Any 10-17 digit numbers

            # With ID/NID prefixes, allowing for flexible spacing and punctuation
            r'(?:ID\s*NO?[:\.]?\s*|NID\s*NO?[:\.]?\s*|1D\s*NO?[:\.]?\s*|IDNO[:\.]?\s*|NIDNO[:\.]?\s*)\s*(\d{10,17})',

            # Spaced formats - comprehensive patterns for all combinations
            r'(\d{3}\s+\d{3}\s+\d{4})',  # 3+3+4 = 10 digits
            r'(\d{4}\s+\d{3}\s+\d{3})',  # 4+3+3 = 10 digits
            r'(\d{6}\s+\d{4})',  # 6+4 = 10 digits
            r'(\d{4}\s+\d{6})',  # 4+6 = 10 digits
            r'(\d{5}\s+\d{5})',  # 5+5 = 10 digits
            r'(\d{3}\s+\d{4}\s+\d{6})',  # 3+4+6 = 13 digits
            r'(\d{4}\s+\d{4}\s+\d{5})',  # 4+4+5 = 13 digits
            r'(\d{6}\s+\d{7})',  # 6+7 = 13 digits
            r'(\d{8}\s+\d{9})',  # 8+9 = 17 digits

            # Hyphenated formats
            r'(\d{3,6}[\s-]\d{3,6}[\s-]\d{3,7})',  # Flexible hyphenated (e.g., xxx-xxxx-xxx)
            r'(\d{6}[\s-]\d{4})',  # 6-4 format
            r'(\d{4}[\s-]\d{6})',  # 4-6 format

            # Handle periods and other separators
            r'(\d{3,6}[.\s]\d{3,6}[.\s]\d{3,7})',  # With periods (e.g., xxx.xxx.xxxx)

            # Next line patterns (when ID NO is on one line, number on next)
            r'(?:ID|NID).*\n\s*(\d{10,17})',  # Number on next line
            r'(?:ID|NID).*\n\s*(\d{3,6}[\s.-]\d{3,6}[\s.-]\d{3,7})',  # Spaced/hyphenated on next line
        ]

        # Name extraction keywords
        self.name_keywords = [
            'name:', 'নাম:', 'name', 'নাম', 'name of'  # Added 'name of' for cases like "Name of Applicant"
        ]

        # Date keywords
        self.date_keywords = [
            'date of birth:', 'date of birth', 'dob:', 'birth:', 'birth date:',
            'জন্ম তারিখ:', 'জন্ম তারিখ', 'জন্ম:'
        ]

        # Words to exclude from names (case-insensitive)
        self.name_exclusions = {
            'father', 'mother', 'husband', 'wife', 'son', 'daughter',
            'পিতা', 'মাতা', 'স্বামী', 'স্ত্রী', 'পুত্র', 'কন্যা',
            'not found', 'government', 'bangladesh', 'republic',
            'national', 'id', 'card', 'birth', 'date', 'number', 'no',
            'signature', 'photo', 'issue', 'date', 'office', 'chairman'
        }

        # Month mappings for date normalization (case-insensitive keys)
        self.month_mappings = {
            'jan': 'Jan', 'january': 'Jan',
            'feb': 'Feb', 'february': 'Feb',
            'mar': 'Mar', 'march': 'Mar',
            'apr': 'Apr', 'april': 'Apr',
            'may': 'May',
            'jun': 'Jun', 'june': 'Jun',
            'jul': 'Jul', 'july': 'Jul',
            'aug': 'Aug', 'august': 'Aug',
            'sep': 'Sep', 'september': 'Sep', 'sept': 'Sep',
            'oct': 'Oct', 'october': 'Oct',
            'nov': 'Nov', 'november': 'Nov',
            'dec': 'Dec', 'december': 'Dec'
        }

        # Common Bangladeshi name components for smart splitting (all uppercase for comparison)
        self.common_first_names = {
            'MD', 'MD.', 'MOHAMMAD', 'MOHAMMED', 'ABDUL', 'ABU', 'SHAH', 'SYED', 'MIRZA',
            'KAZI', 'MAULANA', 'HAFEZ', 'QAZI', 'AMINUL', 'RAFIQUL', 'NAZRUL',
            'SAIFUL', 'KAMAL', 'JAMAL', 'FAZLUL', 'NURUL', 'SHAMSUL', 'ANWARUL',
            'MAHBUB', 'MAHBUBUL', 'HABIB', 'HABIBUL', 'RASHID', 'RASHIDUL',
            'NAZMUL', 'NAZMUR', 'NAZMA', 'NASIR', 'NASIRUL', 'MASUM', 'MASUD',
            'MONTAZ', 'MONTAJ', 'MOSA', 'MUSA', 'SHAMIM', 'SHAHIN', 'SAHIN',
            'ISHFAQ', 'TANZIL', 'GOLAM', 'MOHIUDDIN', 'SESO', 'SHAHID', 'FOYSAL',
            'RUBEL', 'RIPON', 'RASHED', 'RAKIB', 'RAJON', 'ROBIN', 'SUMON', 'MOHSIN'
        }

        self.common_last_names = {
            'RAHMAN', 'AHMED', 'ALI', 'HASAN', 'HOSSAIN', 'ISLAM', 'KHAN', 'SHEIKH',
            'BEGUM', 'KHATUN', 'AKTER', 'AKTAR', 'SULTANA', 'PARVIN', 'NASREEN',
            'FATEMA', 'RASHIDA', 'RAIHAN', 'ROBBANI', 'MONDOL', 'MIAH', 'MIYA',
            'UDDIN', 'ULLAH', 'BANU', 'BIBI', 'KHANAM', 'SIDDIQUE', 'CHOWDHURY',
            'SARKAR', 'DAS', 'ROY', 'PAUL', 'CHAKRABORTY', 'BARUA', 'TALUKDER', 'SHEKH',
            'PRODHAN', 'MISTRY', 'BISWAS', 'MAJUMDER', 'BHUIYAN', 'MALLICK', 'GHOSH'
        }

        self.name_titles = {'MST', 'MR', 'MRS', 'DR', 'PROF', 'A.B.M.', 'ABM', 'MOST', 'M.A.', 'MD.'}

    def preprocess_text(self, text: str) -> str:
        """
        Cleans and normalizes OCR text with enhanced preprocessing.
        """
        if not text:
            return ""

        # Step 1: Apply common spacing fixes and character replacements
        text = self.fix_common_spacing(text)

        # Step 2: Split into lines and process each line
        lines = text.split('\n')
        processed_lines = []

        for line in lines:
            if line.strip():  # Only process non-empty lines
                # Remove excessive whitespace within each line
                line = re.sub(r'[ \t]+', ' ', line.strip())

                # Fix common OCR errors for keywords
                line = line.replace('1D NO', 'ID NO').replace('1D No', 'ID No')
                line = line.replace('IDNO', 'ID NO').replace('NIDNo', 'NID No')
                line = line.replace('ofBirth', 'of Birth').replace('DateofBirth', 'Date of Birth')

                # Normalize colons (ensure space after colon if present)
                line = re.sub(r'(\w)\s*:\s*', r'\1: ', line)

                processed_lines.append(line)

        return '\n'.join(processed_lines)

    def fix_common_spacing(self, text: str) -> str:
        """
        Fixes common OCR spacing and character issues with rule-based preprocessing.
        """
        # Fix 0: Handle common OCR punctuation errors in names (e.g., A.B.M: Foysal -> A.B.M. Foysal)
        text = re.sub(r'\b(A\.B\.M|MD|MST|MR|MRS|DR|PROF):\s*', r'\1. ', text)

        # Fix weird characters in names: replace problematic punctuation with space
        text = re.sub(r'([A-Z][a-z]*[A-Z]*)\s*[!@#$%^&*();,?/\\|+=~`<>{}[\]_-]+\s*([A-Z][a-z]*)', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])([=])([A-Za-z])', r'\1 \3', text)  # Fix A=B -> A B

        # Fix 1: Add proper spacing after common prefixes (e.g., MD.SABUJSHEKH -> MD. SABUJ SHEKH)
        for title in sorted(self.name_titles, key=len, reverse=True):
            # Handle titles followed by concatenated words or just by a space and a word
            # Ensure the dot for titles like MD. is present if that's the intended format.
            text = re.sub(r'\b(' + re.escape(title.replace('.', '\.')) + r')\.?([A-Z]{4,})', r'\1. \2', text)
            text = re.sub(r'\b(' + re.escape(title.replace('.', '\.')) + r')\s+([A-Z]{4,})([A-Z]{4,})', r'\1 \2 \3',
                          text)  # MD. SABUJSHEKH -> MD. SABUJ SHEKH

        # Fix 2: Handle concatenated names with predictable patterns
        # e.g., NAZMURRAIHAN -> NAZMUR RAIHAN, ABDULRAHMAN -> ABDUL RAHMAN
        common_name_splits = [
            (r'(NAZMUR)([A-Z]+)', r'\1 \2'),
            (r'(ABDUL)([A-Z]+)', r'\1 \2'),
            (r'(MOHAMMAD)([A-Z]+)', r'\1 \2'),
            (r'(FAZLUL)([A-Z]+)', r'\1 \2'),
            (r'(RAFIQUL)([A-Z]+)', r'\1 \2'),
            (r'(NAZRUL)([A-Z]+)', r'\1 \2'),
            (r'(ANWARUL)([A-Z]+)', r'\1 \2'),
            (r'(MAHBUBUL)([A-Z]+)', r'\1 \2'),
            (r'(HABIBUL)([A-Z]+)', r'\1 \2'),
            (r'(RASHIDUL)([A-Z]+)', r'\1 \2'),
            (r'(NASIRUL)([A-Z]+)', r'\1 \2'),
            (r'(SHAMSUL)([A-Z]+)', r'\1 \2'),
            (r'(SAIFUL)([A-Z]+)', r'\1 \2'),
            (r'(GOLAM)([A-Z]+)', r'\1 \2'),
        ]
        for pattern, replacement in common_name_splits:
            text = re.sub(pattern, replacement, text)

        # Fix 3: Split obvious concatenated surnames (e.g., ISLAMRAHMAN -> ISLAM RAHMAN)
        surname_splits = [
            (r'(ISLAM)([A-Z]{4,})', r'\1 \2'),
            (r'(RAHMAN)([A-Z]{4,})', r'\1 \2'),
            (r'(AHMED)([A-Z]{4,})', r'\1 \2'),
            (r'(HASAN)([A-Z]{4,})', r'\1 \2'),
            (r'(ALI)([A-Z]{4,})', r'\1 \2'),
            (r'(KHAN)([A-Z]{4,})', r'\1 \2'),
            (r'(BEGUM)([A-Z]{4,})', r'\1 \2'),
            (r'(AKTER)([A-Z]{4,})', r'\1 \2'),
        ]
        for pattern, replacement in surname_splits:
            text = re.sub(pattern, replacement, text)

        # Fix 4: Handle number spacing issues (e.g., "1948176  696" -> "1948176696")
        text = re.sub(r'(\d+)\s{2,}(\d+)', r'\1\2', text)

        # Fix 5: Common OCR keyword fixes before general spacing
        text = text.replace('1D NO', 'ID NO').replace('NIDNO', 'NID NO').replace('IDNO:', 'ID NO:')
        text = text.replace('NID NO:', 'NID NO: ')  # Ensure space after NID NO:

        # Fix 6: Handle date and name keyword formatting
        text = re.sub(r'Date\s*of\s*Birth\s*:?\s*', 'Date of Birth: ', text, flags=re.IGNORECASE)
        text = re.sub(r'Name\s*:?\s*', 'Name: ', text, flags=re.IGNORECASE)

        # Fix 7: Fix concatenated date formats (e.g., 10May1987 -> 10 May 1987)
        text = re.sub(r'\b(\d{1,2})([A-Za-z]{3,9})(\d{4})\b', r'\1 \2 \3', text)

        # Fix 8: Fix partial date spacing (e.g., 10May 1987 -> 10 May 1987)
        text = re.sub(r'\b(\d{1,2})([A-Za-z]{3,9})\s+(\d{4})\b', r'\1 \2 \3', text)
        text = re.sub(r'\b(\d{1,2})\s+([A-Za-z]{3,9})(\d{4})\b', r'\1 \2 \3', text)

        # Fix 9: Normalize multiple spaces (reduce 3+ spaces to single space)
        text = re.sub(r'[ \t]{3,}', ' ', text)

        # Fix 10: Handle A.B.M style names
        text = re.sub(r'A\.B\.M\.?([A-Z])', r'A.B.M. \1', text)
        text = re.sub(r'A\.B\.M:\s*([A-Z])', r'A.B.M. \1', text)

        # Fix 11: Fix incorrect "ABDUL." to "ABDUL" (remove dot if not appropriate for name components)
        text = re.sub(r'\bABDUL\.\s+', 'ABDUL ', text)

        return text

    def smart_split_name(self, concatenated_name: str) -> str:
        """
        Intelligently splits concatenated names using Bangladeshi name patterns.
        """
        if not concatenated_name or len(concatenated_name) < 6:
            return concatenated_name

        name_upper = concatenated_name.upper()

        # Method 1: Try known first name + last name combinations
        for first_name in sorted(self.common_first_names, key=len, reverse=True):
            if name_upper.startswith(first_name):
                remainder = name_upper[len(first_name):]
                if remainder in self.common_last_names:
                    return f"{first_name} {remainder}"
                # Try with common suffixes, e.g., MOHAMMADTAHSINULISLAM -> MOHAMMAD TAHSINUL ISLAM
                for last_name in sorted(self.common_last_names, key=len, reverse=True):
                    if remainder.endswith(last_name):
                        middle_part = remainder[:-len(last_name)]
                        if len(middle_part) >= 2 and middle_part not in self.name_exclusions:
                            return f"{first_name} {middle_part} {last_name}"

        # Method 2: Look for specific common concatenated patterns
        common_splits = [
            ('NAZMUR', 'RAIHAN'), ('ABDUL', 'RAHMAN'), ('MOHAMMAD', 'ALI'),
            ('SAIFUL', 'ISLAM'), ('NURUL', 'ISLAM'), ('GOLAM', 'MOHIUDDIN'),
            ('ABDUL', 'KARIM'), ('ABDUL', 'LATIF'), ('FAZLUL', 'HAQUE'),
            ('RAFIQUL', 'ISLAM'), ('NAZRUL', 'ISLAM'), ('ANWARUL', 'ISLAM')
        ]
        for first, last in common_splits:
            if name_upper == first + last:
                return f"{first} {last}"

        # Method 3: Try to split based on known first names or titles followed by another name component
        for title in sorted(self.name_titles, key=len, reverse=True):
            if name_upper.startswith(title.replace('.', '')):  # Handle MD. vs MD
                remaining = name_upper[len(title.replace('.', '')):]
                if remaining:
                    # Attempt to find another known name component within the remainder
                    for common_part in sorted(list(self.common_first_names) + list(self.common_last_names), key=len,
                                              reverse=True):
                        if remaining.startswith(common_part) and len(remaining) > len(common_part):
                            return f"{title.replace('.', '')} {common_part} {remaining[len(common_part):]}"

        # Method 4: Generic split at the boundary of a common name suffix and a new common name prefix
        # This can be complex and might over-split, use with caution.
        # Example: MOHAMMADSARWARKHAN -> MOHAMMAD SARWAR KHAN
        for i in range(1, len(name_upper) - 1):
            part1 = name_upper[:i]
            part2 = name_upper[i:]
            if part1 in self.common_first_names and part2 in self.common_last_names:
                return f"{part1} {part2}"
            # Check for common middle names or compound surnames
            elif part1 in self.common_first_names and any(
                    part2.startswith(p) for p in self.common_first_names.union(self.common_last_names)):
                # This is a heuristic, needs more refinement
                pass

        return concatenated_name  # Return original if no good split found

    def extract_name(self, text: str) -> Tuple[str, float]:
        """
        Extracts the name with confidence scoring using multiple strategies.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        best_name = "Not Found"
        best_confidence = 0.0

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Method 1: Look for explicit name patterns with keywords (e.g., "Name: John Doe")
            for keyword in self.name_keywords:
                if keyword in line_lower:
                    # Case A: Name on the same line after keyword (with or without colon)
                    name_pattern = r'(?:' + re.escape(keyword) + r')\s*[:=\s]*([A-Z][A-Za-z\s\.\-]+)'
                    match = re.search(name_pattern, line, re.IGNORECASE)
                    if match:
                        name_candidate = match.group(1).strip()
                        confidence = self._validate_name(name_candidate)
                        # Boost confidence for direct keyword match
                        confidence = min(confidence + 0.2, 1.0)
                        if confidence > best_confidence:
                            best_name = name_candidate
                            best_confidence = confidence

                    # Case B: Name on the next line if the current line ends with a keyword/colon
                    if (line_lower.endswith(keyword) or line_lower.endswith(keyword + ':') or line_lower.endswith(
                            keyword + '=')) and i + 1 < len(lines):
                        name_candidate = lines[i + 1].strip()
                        confidence = self._validate_name(name_candidate)
                        # Boost confidence for next-line match after keyword
                        confidence = min(confidence + 0.15, 1.0)
                        if confidence > best_confidence:
                            best_name = name_candidate
                            best_confidence = confidence

            # Method 2: Look for standalone name lines (e.g., "Mst Fulbanu Akter" on its own line)
            # This is a fallback and has lower initial confidence.
            if not any(keyword in line_lower for keyword in list(self.name_exclusions)):
                # Check if this line looks like a typical name (starts with capital or title)
                name_match = re.match(r'^(?:' + '|'.join(
                    re.escape(t.replace('.', '\.')) for t in self.name_titles) + r')?\s*([A-Z][A-Za-z\s\.\-]+)$',
                                      line.strip())
                if name_match:
                    name_candidate = line.strip()
                    confidence = self._validate_name(name_candidate) * 0.7  # Lower confidence for standalone
                    if confidence > best_confidence:
                        best_name = name_candidate
                        best_confidence = confidence

        # Apply smart splitting to the final best name if it appears concatenated
        if ' ' not in best_name and len(best_name) > 6 and best_name != "Not Found":
            split_name = self.smart_split_name(best_name)
            if split_name != best_name:
                logger.info(f"Smart split applied: '{best_name}' -> '{split_name}'")
                best_name = split_name
                best_confidence = min(best_confidence + 0.1, 1.0)  # Small confidence boost for successful split

        return best_name, best_confidence

    def _validate_name(self, name_candidate: str) -> float:
        """
        Validates and scores a name candidate.
        """
        if not name_candidate:
            return 0.0

        # Clean up the name candidate
        name_candidate = name_candidate.strip()

        # Remove common prefixes like "Name:", "নাম:", "=" signs etc.
        for prefix in self.name_keywords + ['=']:
            if name_candidate.lower().startswith(prefix.lower()):
                name_candidate = name_candidate[len(prefix):].strip()
        name_candidate = re.sub(r'^[=:\s]+', '', name_candidate).strip()  # Additional cleaning for leading chars

        # If it's just "name" or empty after cleaning, reject it
        if not name_candidate or name_candidate.lower() in ['name', 'নাম', 'nime', 'nane']:  # Common OCR misreads
            return 0.0

        name_lower = name_candidate.lower()

        # Check for exclusion words (including standalone "name")
        for exclusion in self.name_exclusions:
            if exclusion in name_lower:  # Use 'in' for broader check, but more specific for direct match
                if name_lower == exclusion or name_lower.startswith(exclusion + ' ') or name_lower.endswith(
                        ' ' + exclusion):
                    return 0.0

        # Check if it's just numbers (likely not a name)
        if name_candidate.isdigit() or re.match(r'^\d[\d\s]*\d$', name_candidate):
            return 0.0

        # Score based on characteristics
        score = 0.0

        # Length check
        if 3 <= len(name_candidate) <= 60:  # Extended max length for longer names
            score += 0.2

        # Contains letters
        if re.search(r'[A-Za-z]', name_candidate):
            score += 0.2

        # Proper capitalization or multiple words (common for names)
        # Check for at least one capital letter after first char if it's a multi-word name
        if any(c.isupper() for c in name_candidate) or (
                len(name_candidate.split()) > 1 and name_candidate[0].isupper()):
            score += 0.2

        # Multiple words bonus (names often have multiple parts)
        words = name_candidate.split()
        if len(words) >= 2:
            score += 0.2

        # All caps with multiple words is often a name in OCR
        if name_candidate.isupper() and len(words) >= 2:
            score += 0.1

        # Common name patterns (e.g., A.B.M. Foysal)
        if re.match(r'^[A-Z][A-Z\s\.]+$', name_candidate) or re.match(r'^(?:' + '|'.join(
                re.escape(t.replace('.', '\.')) for t in self.name_titles) + r')\s+[A-Z][A-Za-z\s\.\-]+$',
                                                                      name_candidate):
            score += 0.1

        # Check for known Bangladeshi names/titles
        words_upper = [w.upper() for w in words]
        if any(word in self.common_first_names for word in words_upper):
            score += 0.1
        if any(word in self.common_last_names for word in words_upper):
            score += 0.1
        if any(word.replace('.', '') in self.name_titles for word in words_upper):  # Check for titles like MD, MST
            score += 0.05

        # QUALITY SCORING: Prefer cleaner names
        # Penalize names with unwanted punctuation/characters
        unwanted_chars = ['!', '?', '=', '|', '+', '#', '*', '(', ')', '[', ']', '{', '}', '<', '>', '~']
        if any(char in name_candidate for char in unwanted_chars):
            score -= 0.2  # Significant penalty for messy punctuation

        # Bonus for clean names (only letters, spaces, dots, hyphens, and apostrophes for possessives)
        if re.match(r"^[A-Za-z\s\.\-']+$", name_candidate):
            score += 0.1

        # Prefer names that end with common Bangladeshi name patterns
        name_upper = name_candidate.upper()
        if any(name_upper.endswith(suffix) for suffix in
               ['MONDOL', 'ISLAM', 'RAHMAN', 'AKTER', 'BEGUM', 'KHAN', 'ALI', 'AHMED', 'HASSAN', 'CHOWDHURY', 'SARKAR',
                'UDDIN']):
            score += 0.1

        return min(score, 1.0)  # Cap score at 1.0

    def _is_better_name(self, name1: str, name2: str, conf1: float, conf2: float) -> bool:
        """
        Determines if name1 is a better candidate than name2, considering both quality and confidence.
        """
        if not name1:  # If name1 is empty, name2 is better if it exists
            return False
        if not name2:  # If name2 is empty, name1 is better if it exists
            return True

        # Significant confidence difference - go with higher confidence
        if abs(conf1 - conf2) > 0.3:
            return conf1 > conf2

        # Close confidence - consider quality factors
        quality_score1 = self._calculate_name_quality(name1)
        quality_score2 = self._calculate_name_quality(name2)

        # If quality is significantly different, prefer higher quality
        if abs(quality_score1 - quality_score2) > 0.2:
            return quality_score1 > quality_score2

        # Otherwise, prefer higher confidence (or length if confidences are identical)
        if conf1 == conf2:
            # If confidences are tied, prefer longer names (more complete) or names with more words
            len1 = len(name1.split())
            len2 = len(name2.split())
            if len1 != len2:
                return len1 > len2
            return len(name1) > len(name2)
        return conf1 > conf2

    def _calculate_name_quality(self, name: str) -> float:
        """
        Calculates a quality score for a name (0-1) based on its format and content.
        """
        if not name:
            return 0.0

        score = 0.5  # Base score

        # Penalty for unwanted characters
        unwanted_chars = ['!', '?', '=', '|', '+', '#', '*', '(', ')', '[', ']', '{', '}', '<', '>', '~']
        if any(char in name for char in unwanted_chars):
            score -= 0.3

        # Penalty for remaining prefixes or NID-like patterns
        if name.lower().startswith('name') or '=' in name or re.match(r'^\d{10,17}$', re.sub(r'\s', '', name)):
            score -= 0.3

        # Bonus for clean formatting (only letters, spaces, dots, hyphens, apostrophes)
        if re.match(r"^[A-Za-z\s\.\-']+$", name):
            score += 0.2

        # Bonus for proper Bangladeshi surnames
        common_endings = ['MONDOL', 'ISLAM', 'RAHMAN', 'AKTER', 'BEGUM', 'KHAN', 'ALI', 'AHMED', 'HASSAN', 'CHOWDHURY',
                          'SARKAR', 'UDDIN']
        if any(name.upper().endswith(ending) for ending in common_endings):
            score += 0.2

        # Bonus for multiple words (proper names usually have 2+ words)
        if len(name.split()) >= 2:
            score += 0.1

        # Penalty for very short names that might be false positives
        if len(name) < 5 and len(name.split()) == 1:
            score -= 0.1

        return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1

    def extract_date_of_birth(self, text: str) -> Tuple[str, float]:
        """
        Extracts the date of birth with confidence scoring.
        """
        best_date = "Not Found"
        best_confidence = 0.0

        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check if line contains date keywords (or if it's the line immediately after a potential date keyword)
            has_date_keyword = any(keyword in line_lower for keyword in self.date_keywords)
            if not has_date_keyword and i > 0:
                prev_line_lower = lines[i - 1].lower()
                if any(keyword in prev_line_lower for keyword in self.date_keywords) and not re.search(r'\d',
                                                                                                       prev_line_lower):
                    has_date_keyword = True  # Previous line indicated a date, but no digits there

            # Try all date patterns on the current line
            for pattern in self.date_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    date_candidate = match.group(0)
                    confidence = self._validate_date(date_candidate, has_date_keyword)
                    if confidence > best_confidence:
                        best_date = self._normalize_date(date_candidate)
                        best_confidence = confidence

        return best_date, best_confidence

    def _validate_date(self, date_str: str, has_keyword: bool) -> float:
        """
        Validates and scores a date candidate.
        """
        if not date_str:
            return 0.0

        score = 0.0

        # Higher score if near a date keyword
        if has_keyword:
            score += 0.3

        # Check year range (e.g., 1900-Current Year + 1)
        current_year = 2025  # As per current time context
        year_match = re.search(r'\b(19\d{2}|20[0-2]\d|20[0-1]\d)\b', date_str)  # Adjusted for 2000-2019, 2020-2025
        if year_match:
            year = int(year_match.group(1))
            if 1900 <= year <= current_year + 1:  # Allow slightly future dates in case of scan error or future context
                score += 0.4
                # Bonus for realistic birth years (e.g., 1950-2010 for active NID holders)
                if 1950 <= year <= 2010:
                    score += 0.1

        # Check month format (name or number)
        if re.search(r'\b[A-Za-z]{3,9}\b', date_str):  # Month name (e.g., Jan, January)
            score += 0.3
        elif re.search(r'\b(0?[1-9]|1[0-2])\b', date_str):  # Month number (01-12)
            score += 0.2

        # Check day format (01-31)
        if re.search(r'\b(0?[1-9]|[12]\d|3[01])\b', date_str):
            score += 0.1

        # Bonus for complete date patterns (e.g., "10 Mar 1987")
        if re.match(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', date_str):
            score += 0.2

        # Penalize if it looks like a year-only or fragment
        if re.fullmatch(r'\d{4}', date_str.strip()):
            score -= 0.5

        # Penalize if it contains other non-date related keywords (e.g. "Name 1987")
        if any(kw in date_str.lower() for kw in self.name_keywords + ['id', 'nid', 'number']):
            score -= 0.4

        return min(max(0.0, score), 1.0)  # Ensure score is between 0 and 1

    def _normalize_date(self, date_str: str) -> str:
        """
        Normalizes the date format to "DD Mon YYYY" (e.g., "01 Jan 1970").
        """
        # Prioritize MM/DD/YYYY or DD/MM/YYYY numeric format for initial parsing
        match = re.search(r'(\d{1,2})\s*[-/\s]*(\d{1,2})\s*[-/\s]*(\d{4})', date_str)
        if match:
            day_or_month_1, day_or_month_2, year = match.groups()
            # Heuristic: Assume DD/MM/YYYY is more common for Bangladesh context.
            # If first part > 12, it's likely a day. If second part > 12, it's likely a day and first is month.
            # This is simplified; proper date parsing needs try-except blocks or external libraries.
            if int(day_or_month_1) > 12:  # Assume DD
                day = day_or_month_1
                month_num = int(day_or_month_2)
            elif int(day_or_month_2) > 12:  # Assume DD
                day = day_or_month_2
                month_num = int(day_or_month_1)
            else:  # Ambiguous, default to DD/MM
                day = day_or_month_1
                month_num = int(day_or_month_2)

            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_abbr = month_names[month_num] if 1 <= month_num <= 12 else str(month_num).zfill(2)
            return f"{day.zfill(2)} {month_abbr} {year}"

        # Try day-month_name-year format
        match = re.search(r'(\d{1,2})\s*[-/\s]*([A-Za-z]{3,9})\s*[-/\s]*(\d{4})', date_str, re.IGNORECASE)
        if match:
            day, month_name, year = match.groups()
            normalized_month = self.month_mappings.get(month_name.lower(), month_name.title())
            return f"{day.zfill(2)} {normalized_month} {year}"

        # Try year-month-day format
        match = re.search(r'(\d{4})\s*[-/\s]*(\d{1,2})\s*[-/\s]*(\d{1,2})', date_str)
        if match:
            year, month_num, day = match.groups()
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_abbr = month_names[int(month_num)] if 1 <= int(month_num) <= 12 else str(month_num).zfill(2)
            return f"{day.zfill(2)} {month_abbr} {year}"

        return date_str  # Return original if no recognizable format

    def extract_nid(self, text: str) -> Tuple[str, float]:
        """
        Extracts the NID number with enhanced fragmented pattern detection and confidence scoring.
        """
        best_nid = "Not Found"
        best_confidence = 0.0

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Method 1: Try regex patterns on full text first (most reliable for clean data)
        for pattern in self.nid_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Clean candidate by removing spaces, hyphens, periods
                nid_candidate = re.sub(r'[\s.-]+', '', match.group(1) if match.groups() else match.group(0))
                confidence = self._validate_nid(nid_candidate, text)
                if confidence > best_confidence:
                    best_nid = nid_candidate
                    best_confidence = confidence

        # Method 2: Enhanced bidirectional line-by-line analysis for fragmented NIDs
        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check if line contains NID keywords
            nid_keywords = ['id no', 'nid no', 'id:', 'nid:', 'idno', 'nidno', '1d no', 'national id']
            has_nid_keyword = any(keyword in line_lower for keyword in nid_keywords)

            # Consider lines around the keyword
            if has_nid_keyword:
                digit_sequences = []

                # Current line digits
                current_line_digits = re.findall(r'\d+', line)
                digit_sequences.extend(current_line_digits)

                # Look in previous and next lines for fragmented parts
                for j in range(max(0, i - 2), min(len(lines), i + 3)):  # Search 2 lines before, 2 lines after
                    if i == j: continue  # Skip current line as already processed
                    nearby_line = lines[j].strip()
                    if nearby_line:
                        # Skip lines clearly containing other information
                        if self._line_contains_date_info(nearby_line) or self._line_contains_name_info(nearby_line):
                            continue
                        if any(stop_word in nearby_line.lower() for stop_word in
                               ['father', 'mother', 'signature', 'photo', 'issue', 'office', 'chairman']):
                            continue

                        nearby_digits = re.findall(r'\d+', nearby_line)
                        digit_sequences.extend(nearby_digits)

                # Generate candidates by combining digit sequences
                nid_candidates = self._generate_nid_candidates(digit_sequences)

                for nid_candidate in nid_candidates:
                    confidence = self._validate_nid(nid_candidate, text)
                    # High boost for numbers found near NID keyword
                    confidence = min(confidence + 0.2, 1.0)  # Boost confidence if found near keyword
                    if confidence > best_confidence:
                        best_nid = nid_candidate
                        best_confidence = confidence

        # Method 3: Post-processing to find the best among very similar NIDs if multiple were found
        if best_nid != "Not Found" and best_confidence < 1.0:
            all_found_nids_with_confidence = []
            for pattern in self.nid_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    candidate = re.sub(r'[\s.-]+', '', match.group(1) if match.groups() else match.group(0))
                    conf = self._validate_nid(candidate, text)
                    if conf > 0:
                        all_found_nids_with_confidence.append((candidate, conf))

            if len(all_found_nids_with_confidence) > 1:
                # Sort by confidence descending
                all_found_nids_with_confidence.sort(key=lambda x: x[1], reverse=True)

                # Filter for candidates with high similarity to the best one, if they have decent confidence
                top_nid = all_found_nids_with_confidence[0][0]
                top_conf = all_found_nids_with_confidence[0][1]

                for candidate, conf in all_found_nids_with_confidence[1:]:
                    similarity = difflib.SequenceMatcher(None, top_nid, candidate).ratio()
                    if similarity > 0.8 and conf >= (top_conf * 0.8):  # Similar and good confidence
                        # Prefer longer or more structured NIDs if confidence is very close
                        if len(candidate) > len(top_nid):
                            top_nid = candidate
                            top_conf = conf
                        elif len(candidate) == len(top_nid) and conf > top_conf:
                            top_nid = candidate
                            top_conf = conf
                best_nid = top_nid
                best_confidence = top_conf

        return best_nid, best_confidence

    def _validate_nid(self, nid_str: str, full_text: str) -> float:
        """
        Validates and scores an NID candidate based on length, digits, and context.
        """
        if not nid_str or not nid_str.isdigit():
            return 0.0

        length = len(nid_str)
        score = 0.0

        # Base score on length
        if length == 10:
            score += 0.9
        elif length == 13:
            score += 0.8
        elif length == 17:
            score += 1.0  # 17 digits is the new standard
        elif 9 <= length <= 18:  # Allow slight variations around common lengths
            score += 0.5 - abs(length - 13) * 0.05  # Penalize deviation from 13, 17, 10
            if length == 12 or length == 16:  # common OCR errors for 13 or 17
                score += 0.1

        # Check for non-NID like context (e.g., if it's a phone number or date)
        if re.search(r'(?:phone|mobile|contact|tel|mob)[:\s]*' + nid_str, full_text, re.IGNORECASE):
            score -= 0.5
        if self._line_contains_date_info(nid_str):  # Check if the string itself looks like a date
            score -= 0.5

        # Check for proximity to "ID No" / "NID No" keywords in the full text
        nid_keywords_regex = r'(?:ID|NID|1D)\s*NO?[:\.]?\s*|\bNID\b|\bID\b'
        # Find all occurrences of keywords
        keyword_matches = [m.start() for m in re.finditer(nid_keywords_regex, full_text, re.IGNORECASE)]
        # Find all occurrences of the NID candidate
        nid_matches = [m.start() for m in re.finditer(re.escape(nid_str), full_text)]

        # If the NID candidate is very close to a keyword, boost confidence
        for kw_pos in keyword_matches:
            for nid_pos in nid_matches:
                if abs(kw_pos - nid_pos) < 50:  # Arbitrary distance, adjust as needed
                    score += 0.2
                    break  # Only need one close keyword
            if score > 0.2: break

        # Deduct if it appears multiple times in very different contexts (might be a generic number)
        if len(nid_matches) > 1 and len(set(keyword_matches)) == 0:  # If repeated without NID keywords
            score -= 0.1

        return min(max(0.0, score), 1.0)  # Ensure score is between 0 and 1

    def _generate_nid_candidates(self, digit_sequences: List[str]) -> List[str]:
        """
        Generates potential NID candidates by combining fragmented digit sequences.
        """
        candidates = set()
        cleaned_digits = [re.sub(r'\D', '', d) for d in digit_sequences if re.sub(r'\D', '', d)]  # Ensure only digits

        # Try to combine pairs or triplets of digit sequences
        for i in range(len(cleaned_digits)):
            # Single sequence check
            if 10 <= len(cleaned_digits[i]) <= 17:
                candidates.add(cleaned_digits[i])

            for j in range(i + 1, len(cleaned_digits)):
                combined = cleaned_digits[i] + cleaned_digits[j]
                if 10 <= len(combined) <= 17:
                    candidates.add(combined)

                # Try reverse combination (e.g. if parts are out of order)
                combined_rev = cleaned_digits[j] + cleaned_digits[i]
                if 10 <= len(combined_rev) <= 17:
                    candidates.add(combined_rev)

                for k in range(j + 1, len(cleaned_digits)):
                    combined_triple = cleaned_digits[i] + cleaned_digits[j] + cleaned_digits[k]
                    if 10 <= len(combined_triple) <= 17:
                        candidates.add(combined_triple)
                    combined_triple_alt = cleaned_digits[i] + cleaned_digits[k] + cleaned_digits[j]
                    if 10 <= len(combined_triple_alt) <= 17:
                        candidates.add(combined_triple_alt)

        return list(candidates)

    def _line_contains_date_info(self, line: str) -> bool:
        """Helper to check if a line likely contains date information."""
        line_lower = line.lower()
        if any(re.search(p, line_lower) for p in self.date_patterns):
            return True
        if any(keyword in line_lower for keyword in self.date_keywords):
            return True
        # Simple check for DD/MM/YYYY or similar
        if re.search(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{4}', line):
            return True
        return False

    def _filter_date_digits(self, digits: List[str], line_context: str) -> List[str]:
        """Filters out digit sequences that look like dates from a list of digits."""
        filtered = []
        for d in digits:
            if re.fullmatch(r'\d{1,2}', d) and self._line_contains_date_info(line_context):
                continue  # Likely a day/month, not a full NID part
            if re.fullmatch(r'\d{4}', d) and self._line_contains_date_info(line_context):
                continue  # Likely a year, not a full NID part
            filtered.append(d)
        return filtered

    def _line_contains_name_info(self, line: str) -> bool:
        """Helper to check if a line likely contains name information."""
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in self.name_keywords):
            return True
        # Heuristic: If it contains many words and starts with a capital letter
        words = line.strip().split()
        if len(words) >= 2 and words[0] and words[0][0].isupper():
            if not any(excl in line_lower for excl in ['id', 'nid', 'number', 'birth', 'date']):
                return True
        return False

    def extract_information(self, ocr_text: str) -> ExtractionResult:
        """
        Extracts all NID-related information from the OCR text.
        """
        logger.info("Starting information extraction...")
        raw_text = ocr_text  # Store original for result

        # Step 1: Preprocess the OCR text
        processed_text = self.preprocess_text(ocr_text)
        logger.debug(f"Preprocessed text:\n{processed_text}")

        # Step 2: Extract Name
        name, name_conf = self.extract_name(processed_text)
        logger.info(f"Extracted Name: '{name}' (Confidence: {name_conf:.2f})")

        # Step 3: Extract Date of Birth
        dob, dob_conf = self.extract_date_of_birth(processed_text)
        logger.info(f"Extracted DOB: '{dob}' (Confidence: {dob_conf:.2f})")

        # Step 4: Extract NID
        nid, nid_conf = self.extract_nid(processed_text)
        logger.info(f"Extracted NID: '{nid}' (Confidence: {nid_conf:.2f})")

        return ExtractionResult(
            name=name,
            dob=dob,
            nid=nid,
            name_confidence=name_conf,
            dob_confidence=dob_conf,
            nid_confidence=nid_conf,
            extraction_method="Enhanced Regex + Heuristics",
            raw_text=raw_text
        )






# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_test(ocr_text: str, test_name: str):
    """Helper function to run an extraction test and print results."""
    logger.info(f"\n--- Running Test: {test_name} ---")
    extractor = EnhancedNIDExtractor()
    result = extractor.extract_information(ocr_text)

    logger.info(f"Test '{test_name}' Results:")
    logger.info(f"  Name: {result.name} (Confidence: {result.name_confidence:.2f})")
    logger.info(f"  DOB: {result.dob} (Confidence: {result.dob_confidence:.2f})")
    logger.info(f"  NID: {result.nid} (Confidence: {result.nid_confidence:.2f})")
    logger.info(f"  Method: {result.extraction_method}")
    logger.info(f"  Raw Text Sample: {result.raw_text[:100]}...")  # Print a snippet of raw text


if __name__ == "__main__":
    # --- Test Cases ---

    # Test Case 1: Standard well-formatted OCR text
    ocr_text_2 = """
    Government of the People's Republic of Bangladesh
    National ID Card
    JTA
    Name: JUIEAL CHANDRANATH
    Date of Birth: 10 May 1983
    NID NO: 5963485064
    """
    run_test(ocr_text_2, "Standard Format Test")


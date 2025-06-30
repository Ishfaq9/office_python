import re


def extract_id_after_date(text):
    print("Raw text:", text)
    # Regex to match a date (DD MMM YYYY) followed by a 10, 13, or 17-digit ID (with optional spaces, dashes, etc.)
    combined_pattern = re.compile(
        r'(?:\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\b\s*[^0-9]*((?:\d[-\s@#]*?){10,17}(?=\b)))|((?:\d[-\s@#]*?){10,17}(?=\b))',
        re.IGNORECASE | re.DOTALL
    )
    match = combined_pattern.search(text)
    if match:
        # Use group(1) if date pattern exists, otherwise group(2) for standalone ID
        raw_id = match.group(1) if match.group(1) else match.group(2)
        print("Raw ID:", raw_id)
        # Clean the ID by removing non-digit characters
        cleaned_id = re.sub(r'[-\s@#]', '', raw_id)
        print("Cleaned ID:", cleaned_id)
        # Verify the length of the cleaned ID
        if len(cleaned_id) in [10, 13, 17]:
            return cleaned_id
        else:
            print(f"Invalid ID length: {len(cleaned_id)} digits")
            return None
    print("No valid ID found")
    return None

# Test with raw data
raw_text = """Government of/thePeople's Republic of Bangladesh /NationalDCard Name MST.SAHERA BEGUM Date of Birth  -59-5 966681@87681 """
result = extract_id_after_date(raw_text)
print("Final ID:", result)

# Test with another example
# raw_text2 = """Some random text. Name: Ishfaq Rahman  02 Feb 2021 12345678 90 And more text."""
# result2 = extract_id_after_date(raw_text2)
# print("Final ID:", result2)
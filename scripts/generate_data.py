"""CLI script to generate synthetic data."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "data" / "samples" / "sample_candidates.json"

FIRST_NAMES = [
    "Aarav", "Aditi", "Amit", "Ananya", "Arjun", "Deepika", "Ishaan", "Kavya", 
    "Nikhil", "Pooja", "Rahul", "Riya", "Rohan", "Sanjana", "Siddharth", 
    "Tanvi", "Varun", "Vidyut", "Yash", "Zoya"
]
LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Mehra", "Joshi", "Patel", "Shah", "Reddy", 
    "Nair", "Iyer", "Rao", "Kumar", "Singh", "Das", "Choudhury", "Bose", 
    "Banerjee", "Chatterjee", "Sen", "Menon"
]


def generate_random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def main() -> None:
    if not SAMPLE_PATH.exists():
        print(f"Sample file not found at {SAMPLE_PATH}")
        return

    with open(SAMPLE_PATH) as f:
        existing_profiles = json.load(f)

    print(f"Loaded {len(existing_profiles)} existing profiles.")

    # Import constants from the project
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.core.constants import INDIAN_CITIES, INDIAN_COMPANIES, INDIAN_UNIVERSITIES

    target_count = 1000
    needed = target_count - len(existing_profiles)
    
    generated = list(existing_profiles)
    for _ in range(needed):
        # Clone an existing profile as template
        template = random.choice(existing_profiles)
        cloned = json.loads(json.dumps(template))
        
        # Unique ID
        new_id = f"CAND_{len(generated) + 1:07d}"
        cloned["candidate_id"] = new_id
        
        # Randomize personal info
        new_name = generate_random_name()
        cloned["profile"]["anonymized_name"] = new_name
        
        # Randomize location
        cloned["profile"]["location"] = random.choice(INDIAN_CITIES)
        cloned["profile"]["country"] = "India"
        
        # Randomize experience years slightly
        exp = round(max(0.5, template["profile"]["years_of_experience"] + random.uniform(-2, 2)), 1)
        cloned["profile"]["years_of_experience"] = exp
        
        # Randomize company
        company = random.choice(INDIAN_COMPANIES)
        cloned["profile"]["current_company"] = company
        
        # Randomize career history
        if "career_history" in cloned and cloned["career_history"]:
            for job in cloned["career_history"]:
                job["company"] = random.choice(INDIAN_COMPANIES)
        
        # Randomize education
        if "education" in cloned and cloned["education"]:
            for edu in cloned["education"]:
                edu["institution"] = random.choice(INDIAN_UNIVERSITIES)
                
        generated.append(cloned)

    # Save back to file
    with open(SAMPLE_PATH, "w") as f:
        json.dump(generated, f, indent=2)

    print(f"Successfully generated and saved {len(generated)} profiles to {SAMPLE_PATH}.")


if __name__ == "__main__":
    main()

# ruff: noqa: E501
#!/usr/bin/env python3
"""Check data quality issues in the candidate dataset."""
import os
import sys

sys.path.insert(0, '/home/nanda/India-Runs')
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.core.config import DATA_DIR
from src.core.profile_store import ProfileStore

profiles = ProfileStore()
profiles.load_sample(DATA_DIR / "samples" / "sample_candidates.json")
all_p = profiles.get_all_sample()

print(f"Total profiles: {len(all_p)}")

zero_roles = []
no_signals = []
has_exp_but_no_roles = []

for pid, p in all_p.items():
    if len(p.experience) == 0:
        zero_roles.append(pid)
        if p.professional and p.professional.total_experience_years:
            has_exp_but_no_roles.append(pid)

    sig = p.signals
    has_any_signal = bool(
        (sig.saved_by_recruiters_30d or 0) > 0 or
        sig.recruiter_response_rate is not None or
        sig.profile_completeness_score is not None or
        sig.open_to_work or
        sig.verified_email or
        sig.verified_phone or
        sig.notice_period_days is not None
    )
    if not has_any_signal:
        no_signals.append(pid)

print(f"\nProfiles with 0 experience roles: {len(zero_roles)}/{len(all_p)}")
print(f"  Which also have total_experience_years: {len(has_exp_but_no_roles)}")
print(f"Profiles with NO behavioral signals: {len(no_signals)}/{len(all_p)}")

# Show first few of each
print("\n--- Zero roles but have exp years ---")
for pid in has_exp_but_no_roles[:10]:
    p = all_p[pid]
    na = p.personal.name if p.personal else "?"
    exp_y = p.professional.total_experience_years if p.professional else 0
    title = p.professional.current_title if p.professional else "?"
    company = p.professional.current_company if p.professional else "?"
    print(f"  {pid}: {na} — {title} @ {company} ({exp_y}y)")

# Check raw text for presence of experience info
print("\n--- Checking raw_text for exp info in CAND_0000100 ---")
p100 = all_p["CAND_0000100"]
print(f"Total exp years: {p100.professional.total_experience_years}")
print(f"Experience list length: {len(p100.experience)}")
if len(p100.raw_text) > 200:
    print(f"Raw text (first 600 chars):\n{p100.raw_text[:600]}")

# Check if role info exists in raw text but wasn't parsed into experience
print("\n--- Checking raw data structure ---")
import json  # noqa: E402

samples = DATA_DIR / "samples" / "sample_candidates.json"
with open(samples) as f:
    data = json.load(f)

for pid in has_exp_but_no_roles[:3]:
    if pid in data:
        # Check various possible keys for experience
        raw = data[pid]
        for key in raw:
            val = raw[key]
            if isinstance(val, list) and len(val) > 0:
                print(f"{pid}.{key}: list of {len(val)} items")
                if isinstance(val[0], dict):
                    print(f"  keys: {list(val[0].keys())[:8]}")
            elif isinstance(val, dict):
                if any(k in str(val).lower() for k in ['experience', 'job', 'company', 'role']):
                    print(f"{pid}.{key}: dict with relevant keys: {list(val.keys())[:10]}")

# Also check if there's a 'career' or 'roles' field
print("\n--- Checking raw data keys across samples ---")
all_keys = set()
for pid in has_exp_but_no_roles[:10]:
    if pid in data:
        all_keys.update(data[pid].keys())
print(f"All raw data keys in samples: {sorted(all_keys)}")

# Check the first profile with roles for comparison
for pid in all_p:
    if len(all_p[pid].experience) > 0:
        p = all_p[pid]
        if pid in data:
            print(f"\n--- {pid} (has roles) ---")
            print(f"Experience: {len(p.experience)} roles")
            raw_exp_key = None
            for key in data[pid]:
                val = data[pid][key]
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and 'title' in str(val[0]).lower():
                    raw_exp_key = key
                    break
            print(f"Raw data experience key: {raw_exp_key}")
            if raw_exp_key:
                print(f"First entry: {json.dumps(data[pid][raw_exp_key][0], indent=2)[:300]}")
        break

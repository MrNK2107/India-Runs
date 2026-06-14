from __future__ import annotations

from src.core.config import DATA_DIR

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
}

PROFILE_SOURCES = ("linkedin", "naukri", "github", "resume_pdf", "career_page", "manual")

SKILL_CATEGORIES = (
    "programming_language",
    "framework",
    "tool",
    "soft_skill",
    "domain_knowledge",
    "certification",
)

PROFICIENCY_LEVELS = ("beginner", "intermediate", "advanced", "expert")

SKILL_IMPORTANCE = ("required", "preferred", "nice_to_have")

EMPLOYMENT_TYPES = ("full_time", "part_time", "contract", "freelance", "student")

MATCH_RECOMMENDATIONS = ("strong_match", "good_match", "potential_match", "weak_match")

SEARCH_METHODS = ("hybrid", "vector_only", "keyword_only")

INDIAN_COMPANIES = [
    "Flipkart", "Zoho", "Freshworks", "TCS", "Infosys", "Wipro", "HCL",
    "Razorpay", "PhonePe", "Swiggy", "Zomato", "Ola", "Paytm", "BYJU'S",
    "PolicyBazaar", "Dream11", "Meesho", "CRED", "Postman", "Hasura",
    "CitrusPay", "Mu Sigma", "Fractal Analytics", "Postman", "BrowserStack",
    "Chargebee", "Zenoti", "InMobi", "BigBasket", "UrbanClap", "Vedantu",
    "Unacademy", "Cult.fit", "Cars24", "Delhivery", "BlackBuck",
    "Icertis", "Druva", "Sigmoid", "Tiger Analytics", "Fractal",
    "Publicis Sapient", "Infosys BPM", "Mindtree", "Mphasis",
]

INDIAN_CITIES = [
    "Bangalore", "Hyderabad", "Pune", "Chennai", "Noida", "Gurgaon",
    "Mumbai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Kochi",
    "Indore", "Bhopal", "Coimbatore", "Visakhapatnam", "Thiruvananthapuram",
    "Mysore", "Nagpur", "Chandigarh",
]

INDIAN_UNIVERSITIES = [
    "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
    "BITS Pilani", "NIT Trichy", "NIT Warangal", "IIIT Hyderabad",
    "VIT Vellore", "SRM University", "Manipal Institute of Technology",
    "Delhi Technological University", "Punjab Engineering College",
    "Anna University", "Osmania University", "JNTU Hyderabad",
    "University of Mumbai", "Pune University", "Christ University Bangalore",
]

DEFAULT_MIN_EXPERIENCE = 0
DEFAULT_MAX_EXPERIENCE = 50
DEFAULT_LOCATION = None

FAISS_INDEX_PATH = DATA_DIR / "indexes" / "faiss_index.bin"
FAISS_ID_MAP_PATH = DATA_DIR / "indexes" / "faiss_id_map.json"
BM25_INDEX_PATH = DATA_DIR / "indexes" / "bm25_index.pkl"
PROFILES_PATH = DATA_DIR / "profiles" / "profiles.json"
QUERIES_PATH = DATA_DIR / "queries" / "queries.json"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth" / "ground_truth.json"

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

INDIAN_MNCS = [
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix",
    "Uber", "Twitter", "LinkedIn", "Salesforce", "Adobe", "Oracle",
    "IBM", "Intel", "Cisco", "Dell", "HP", "SAP", "Visa", "Mastercard",
    "PayPal", "JP Morgan", "Goldman Sachs", "Morgan Stanley",
    "Accenture", "Deloitte", "PwC", "EY", "KPMG", "McKinsey", "BCG",
    "Bain & Company", "Qualcomm", "NVIDIA", "AMD", "Intel",
]

INDIAN_PRODUCT_COMPANIES = [
    "Flipkart", "Zoho", "Freshworks", "Razorpay", "PhonePe", "Swiggy",
    "Zomato", "Ola", "Paytm", "BYJU'S", "PolicyBazaar", "Dream11",
    "Meesho", "CRED", "Postman", "Hasura", "BrowserStack", "Chargebee",
    "Zenoti", "InMobi", "BigBasket", "Urban Company", "Vedantu",
    "Unacademy", "Cult.fit", "Cars24", "Delhivery", "BlackBuck",
    "Icertis", "Druva", "Rivigo", "Porter", "Licious", "Nykaa",
    "FirstCry", "Fractal Analytics", "Tiger Analytics", "Sigmoid",
    "Mu Sigma", "CitrusPay", "Fyle", "Slintel", "Whatfix",
    "Innovaccer", "HealthPlix", "Practo", "PharmEasy", "1mg",
    "OYO", "MakeMyTrip", "ixigo", "Cleartrip", "Rapido",
]

INDIAN_IT_SERVICES = [
    "TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "LTI",
    "Mindtree", "Mphasis", "L&T Infotech", "Cognizant", "Capgemini",
    "Publicis Sapient", "Persistent Systems", "Cyient", "KPIT",
    "Coforge", "Hexaware", "Sonata Software", "Birlasoft",
    "Zensar Technologies", "NIIT Technologies", "Oracle Financial Services",
]

INDIAN_BANKS_FINTECH = [
    "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra", "Yes Bank",
    "SBI", "Bank of Baroda", "Punjab National Bank", "BharatPe",
    "Groww", "Zerodha", "Upstox", "Angel One", "Kuvera", "Smallcase",
    "INDmoney", "Jupiter Money", "CRED", "MobiKwik", "Freecharge",
]

INDIAN_STARTUPS = [
    "Zepto", "Blinkit", "Dunzo", "ShareChat", "Moj", "Koo",
    "Apna", "Gupshup", "LeadSquared", "Yellow.ai", "Yellow Messenger",
    "Observe.ai", "Uniphore", "Mad Street Den", "Haptik", "Niki.ai",
    "Locus.sh", "FarEye", "ElasticRun", "Ninjacart", "WayCool",
    "Aye Finance", "Lendingkart", "Indifi", "NeoGrowth",
    "Rebel Foods", "Chaayos", "Third Wave Coffee", "Blue Tokai",
    "Ola Electric", "Ather Energy", "Bounce", "Yulu",
    "Pristyn Care", "MFine", "Cure.fit", "Tata 1mg",
]

INDIAN_COMPANIES = (
    INDIAN_MNCS + INDIAN_PRODUCT_COMPANIES + INDIAN_IT_SERVICES
    + INDIAN_BANKS_FINTECH + INDIAN_STARTUPS
)

INDIAN_CITIES = [
    # Metro cities
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad",
    # State capitals
    "Thiruvananthapuram", "Bengaluru", "Bhopal", "Bhubaneswar",
    "Chandigarh", "Dehradun", "Dispur", "Gandhinagar", "Gangtok",
    "Guwahati", "Hyderabad", "Imphal", "Itanagar", "Jaipur",
    "Jammu", "Kohima", "Lucknow", "Panaji", "Patna", "Raipur",
    "Ranchi", "Shillong", "Shimla", "Srinagar", "Amaravati",
    # Tier-2 tech hubs
    "Coimbatore", "Indore", "Nagpur", "Visakhapatnam", "Vadodara",
    "Surat", "Nashik", "Aurangabad", "Madurai", "Tiruchirappalli",
    "Mysore", "Mangalore", "Belgaum", "Hubli", "Kochi",
    "Kozhikode", "Kannur", "Kollam", "Alappuzha",
    # NCR region
    "Gurgaon", "Gurugram", "Noida", "Faridabad", "Ghaziabad",
    "Greater Noida", "Dwarka",
    # Tier-2 North
    "Agra", "Aligarh", "Allahabad", "Amritsar", "Bareilly",
    "Chandigarh", "Gorakhpur", "Jalandhar", "Jodhpur", "Kanpur",
    "Kota", "Ludhiana", "Meerut", "Moradabad", "Udaipur",
    "Varanasi", "Haridwar", "Rishikesh", "Mathura", "Vrindavan",
    # Tier-2 East
    "Asansol", "Bokaro", "Cuttack", "Dhanbad", "Durgapur",
    "Jamshedpur", "Patna", "Puri", "Rourkela", "Siliguri",
    # Tier-2 West
    "Bhavnagar", "Bhuj", "Gandhidham", "Jamnagar", "Junagadh",
    "Porbandar", "Rajkot", "Surendranagar", "Valsad", "Vapi",
    # Tier-2 South
    "Guntur", "Kakinada", "Kurnool", "Nellore", "Rajahmundry",
    "Tirupati", "Vijayawada", "Warangal", "Kadapa", "Anantapur",
    # Tier-2 Central
    "Bilaspur", "Gwalior", "Jabalpur", "Raipur", "Ujjain",
    # Tier-2 Northeast
    "Agartala", "Aizawl", "Dimapur", "Shillong", "Silchar",
    "Tinsukia", "Jorhat",
    # Satellite towns
    "Electronic City", "Whitefield", "Marathahalli", "Kharadi",
    "Hinjewadi", "Navi Mumbai", "Thane", "Kalyan", "Vasai",
    "Panvel", "Pimpri Chinchwad", "Salt Lake City", "Rajarhat",
    "New Town Kolkata", "Sriperumbudur", "Siruseri", "Oragadam",
    "Devanahalli",
]

INDIAN_UNIVERSITIES = [
    # IITs
    "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
    "IIT Roorkee", "IIT Guwahati", "IIT Hyderabad", "IIT Jodhpur",
    "IIT Patna", "IIT Bhubaneswar", "IIT Mandi", "IIT Ropar",
    "IIT Gandhinagar", "IIT Palakkad", "IIT Tirupati", "IIT Bhilai",
    "IIT Goa", "IIT Dharwad", "IIT Jammu",
    # NITs
    "NIT Trichy", "NIT Warangal", "NIT Surathkal", "NIT Calicut",
    "NIT Rourkela", "NIT Durgapur", "NIT Kurukshetra", "NIT Silchar",
    "NIT Allahabad", "NIT Srinagar", "NIT Patna", "NIT Raipur",
    "NIT Nagpur", "NIT Jamshedpur", "NIT Bhopal", "NIT Hamirpur",
    "NIT Jaipur", "NIT Meghalaya", "NIT Manipur", "NIT Mizoram",
    # IIITs
    "IIIT Hyderabad", "IIIT Bangalore", "IIIT Delhi", "IIIT Allahabad",
    "IIIT Guwahati", "IIIT Lucknow", "IIIT Pune", "IIIT Kancheepuram",
    "IIITDM Jabalpur", "IIITDM Chennai", "IIIT Vadodara", "IIIT Kota",
    "IIIT Manipur", "IIIT Nagpur", "IIIT Ranchi", "IIIT Sri City",
    # BITS & Pilani
    "BITS Pilani", "BITS Goa", "BITS Hyderabad", "BITS Dubai",
    # Premier engineering
    "VIT Vellore", "SRM University", "Manipal Institute of Technology",
    "Delhi Technological University", "NSUT Delhi", "IIIT Delhi",
    "Punjab Engineering College", "Thapar Institute", "LPU Jalandhar",
    "Shiv Nadar University", "Ashoka University", "Amrita University",
    # State universities
    "Anna University", "Osmania University", "JNTU Hyderabad",
    "University of Mumbai", "Pune University", "Savitribai Phule Pune University",
    "University of Delhi", "Jamia Millia Islamia", "AMU Aligarh",
    "BHU Varanasi", "Calcutta University", "Jadavpur University",
    "University of Madras", "University of Kerala", "Goa University",
    "Gujarat University", "MS University Baroda", "Sardar Patel University",
    # Private universities
    "Christ University Bangalore", "BML Munjal University",
    "Amity University", "SRM IST Chennai", "KIIT Bhubaneswar",
    "SOA Bhubaneswar", "D.Y. Patil University", "Symbiosis Pune",
    "NMIMS Mumbai", "VIT Chennai", "Sathyabama Chennai",
    "Hindustan Institute of Technology", "SASTRA Thanjavur",
    "PES University Bangalore", "RV College Bangalore",
    "BMS College Bangalore", "Dayananda Sagar Bangalore",
    "M.S. Ramaiah Bangalore", "CMR University Bangalore",
    # NITTE
    "NITTE University Mangalore", "MIT Manipal",
    # IIMs (for signal value)
    "IIM Ahmedabad", "IIM Bangalore", "IIM Calcutta", "IIM Lucknow",
    "IIM Kozhikode", "IIM Indore", "IIM Shillong", "IIM Rohtak",
    "IIM Ranchi", "IIM Raipur", "IIM Tiruchirappalli", "IIM Udaipur",
    "IIM Nagpur", "IIM Jammu", "IIM Bodh Gaya", "IIM Amritsar",
    "IIM Sambalpur", "IIM Sirmaur", "IIM Visakhapatnam",
    # ISB
    "ISB Hyderabad", "ISB Mohali",
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

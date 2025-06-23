"""
Academic subject-based categorization system
Based on traditional academic disciplines and subject areas
"""

from typing import List, Optional, Dict

# Comprehensive academic subject areas covering all knowledge domains
ACADEMIC_SUBJECT_CATEGORIES = [
    # STEM Fields
    "Mathematics",
    "Physics", 
    "Chemistry",
    "Biology",
    "Computer Science",
    "Engineering",
    "Medicine",
    "Environmental Science",
    "Astronomy",
    "Geology",
    
    # Social Sciences
    "Psychology",
    "Sociology", 
    "Anthropology",
    "Political Science",
    "Economics",
    "Geography",
    "Linguistics",
    "Archaeology",
    
    # Humanities
    "History",
    "Philosophy",
    "Literature",
    "Languages",
    "Religious Studies",
    "Art History",
    "Music Theory",
    "Theater Arts",
    
    # Applied & Professional Fields
    "Business Administration",
    "Law",
    "Education",
    "Communications",
    "Journalism",
    "Architecture",
    "Agriculture",
    "Nutrition",
    
    # Arts & Creative
    "Visual Arts",
    "Music",
    "Creative Writing",
    "Film Studies",
    "Photography",
    "Design",
    
    # Health & Physical
    "Public Health",
    "Physical Education",
    "Sports Science",
    "Therapy & Rehabilitation",
    
    # Interdisciplinary & Modern
    "Data Science",
    "Cybersecurity",
    "Biotechnology",
    "Cognitive Science",
    "International Studies",
    "Gender Studies",
    "Urban Planning",
    "General Studies"
]

# Subject area descriptions for better LLM understanding
SUBJECT_DESCRIPTIONS = {
    "Mathematics": "Number theory, algebra, calculus, statistics, geometry, applied mathematics, mathematical modeling",
    "Physics": "Mechanics, thermodynamics, electromagnetism, quantum physics, relativity, particle physics, astrophysics",
    "Chemistry": "Organic chemistry, inorganic chemistry, physical chemistry, biochemistry, analytical chemistry, materials science",
    "Biology": "Cell biology, genetics, ecology, evolution, microbiology, botany, zoology, molecular biology",
    "Computer Science": "Programming, algorithms, data structures, software engineering, artificial intelligence, machine learning, databases",
    "Engineering": "Mechanical, electrical, civil, chemical, software engineering, systems design, manufacturing",
    "Medicine": "Anatomy, physiology, pathology, pharmacology, clinical medicine, medical research, healthcare",
    "Environmental Science": "Ecology, climate science, conservation, environmental policy, sustainability, renewable energy",
    "Astronomy": "Planetary science, stellar astronomy, cosmology, space exploration, observational astronomy",
    "Geology": "Earth science, mineralogy, petrology, geophysics, paleontology, geological processes",
    
    "Psychology": "Cognitive psychology, behavioral psychology, developmental psychology, social psychology, clinical psychology",
    "Sociology": "Social theory, social institutions, social movements, demographics, social research methods",
    "Anthropology": "Cultural anthropology, physical anthropology, archaeological methods, ethnography",
    "Political Science": "Government systems, political theory, international relations, public policy, comparative politics",
    "Economics": "Microeconomics, macroeconomics, econometrics, financial markets, economic policy, behavioral economics",
    "Geography": "Physical geography, human geography, cartography, geographic information systems, urban geography",
    "Linguistics": "Language structure, phonetics, syntax, semantics, sociolinguistics, computational linguistics",
    "Archaeology": "Archaeological methods, cultural heritage, artifact analysis, historical reconstruction",
    
    "History": "World history, historical analysis, historiography, specific historical periods and events",
    "Philosophy": "Ethics, logic, metaphysics, epistemology, political philosophy, philosophy of mind",
    "Literature": "Literary analysis, literary theory, creative writing, poetry, prose, literary criticism",
    "Languages": "Foreign language learning, translation, linguistic analysis, cultural studies",
    "Religious Studies": "Theology, comparative religion, religious history, spiritual practices",
    "Art History": "Art movements, artistic techniques, cultural context of art, museum studies",
    "Music Theory": "Composition, musical analysis, music history, performance techniques",
    "Theater Arts": "Drama, performance, theatrical production, dramatic literature",
    
    "Business Administration": "Management, strategy, operations, organizational behavior, business analytics",
    "Law": "Legal theory, jurisprudence, legal research, case analysis, legal writing",
    "Education": "Pedagogy, curriculum development, educational psychology, teaching methods, learning theory",
    "Communications": "Media studies, public relations, advertising, digital communication, interpersonal communication",
    "Journalism": "News writing, investigative reporting, media ethics, broadcast journalism, digital journalism",
    "Architecture": "Architectural design, urban planning, building technology, architectural history",
    "Agriculture": "Crop science, animal husbandry, agricultural technology, sustainable farming",
    "Nutrition": "Dietetics, food science, nutritional biochemistry, public health nutrition",
    
    "Visual Arts": "Painting, sculpture, drawing, digital art, graphic design, illustration",
    "Music": "Musical performance, composition, music production, music technology",
    "Creative Writing": "Fiction writing, poetry, screenwriting, narrative techniques, creative process",
    "Film Studies": "Cinematography, film theory, video production, film history, documentary making",
    "Photography": "Photographic techniques, digital photography, photo editing, visual storytelling",
    "Design": "Graphic design, user interface design, industrial design, design thinking",
    
    "Public Health": "Epidemiology, health policy, disease prevention, community health, global health",
    "Physical Education": "Exercise science, sports coaching, fitness training, movement analysis",
    "Sports Science": "Athletic performance, biomechanics, sports psychology, sports medicine",
    "Therapy & Rehabilitation": "Physical therapy, occupational therapy, speech therapy, rehabilitation methods",
    
    "Data Science": "Data analysis, machine learning, statistical modeling, data visualization, big data",
    "Cybersecurity": "Information security, network security, cryptography, security policy, threat analysis",
    "Biotechnology": "Genetic engineering, bioinformatics, medical technology, bioprocessing",
    "Cognitive Science": "Mind and brain studies, artificial intelligence, neuroscience, consciousness studies",
    "International Studies": "Global affairs, international relations, cultural studies, globalization",
    "Gender Studies": "Gender theory, feminist studies, social identity, equality studies",
    "Urban Planning": "City planning, transportation planning, zoning, sustainable development",
    "General Studies": "Interdisciplinary knowledge, life skills, personal development, general knowledge"
}

def get_all_subject_categories() -> List[str]:
    """Get all academic subject categories"""
    return ACADEMIC_SUBJECT_CATEGORIES.copy()

def get_subject_categories_prompt_text() -> str:
    """Get formatted subject categories for LLM prompts"""
    categories_text = "Available academic subject categories:\n\n"
    for i, category in enumerate(ACADEMIC_SUBJECT_CATEGORIES, 1):
        description = SUBJECT_DESCRIPTIONS.get(category, "")
        categories_text += f"{i:2d}. {category}\n"
        if description:
            categories_text += f"    → {description}\n"
        categories_text += "\n"
    return categories_text

def validate_subject_category(category: str) -> bool:
    """Check if a subject category is valid"""
    return category in ACADEMIC_SUBJECT_CATEGORIES

def get_subject_category_fallback(sub_category: str) -> str:
    """
    Simple fallback function if LLM doesn't provide subject category
    Maps keywords to academic subjects
    """
    if not sub_category:
        return "General Studies"
    
    sub_lower = sub_category.lower()
    
    # STEM subjects
    if any(word in sub_lower for word in ['math', 'mathematics', 'algebra', 'calculus', 'geometry', 'statistics']):
        return "Mathematics"
    elif any(word in sub_lower for word in ['physics', 'mechanics', 'quantum', 'relativity', 'particle']):
        return "Physics"
    elif any(word in sub_lower for word in ['chemistry', 'chemical', 'molecule', 'reaction', 'compound']):
        return "Chemistry"
    elif any(word in sub_lower for word in ['biology', 'biological', 'cell', 'genetic', 'organism', 'ecosystem']):
        return "Biology"
    elif any(word in sub_lower for word in ['programming', 'software', 'algorithm', 'computer', 'coding', 'ai', 'machine learning']):
        return "Computer Science"
    elif any(word in sub_lower for word in ['engineering', 'mechanical', 'electrical', 'civil', 'design', 'construction']):
        return "Engineering"
    elif any(word in sub_lower for word in ['medicine', 'medical', 'health', 'disease', 'treatment', 'clinical']):
        return "Medicine"
    elif any(word in sub_lower for word in ['environment', 'climate', 'ecology', 'conservation', 'sustainability']):
        return "Environmental Science"
    elif any(word in sub_lower for word in ['space', 'astronomy', 'planet', 'star', 'cosmic', 'universe']):
        return "Astronomy"
    elif any(word in sub_lower for word in ['earth', 'geology', 'rock', 'mineral', 'geological']):
        return "Geology"
    
    # Social Sciences
    elif any(word in sub_lower for word in ['psychology', 'psychological', 'behavior', 'mental', 'cognitive']):
        return "Psychology"
    elif any(word in sub_lower for word in ['sociology', 'social', 'society', 'culture', 'community']):
        return "Sociology"
    elif any(word in sub_lower for word in ['anthropology', 'anthropological', 'ethnography', 'cultural']):
        return "Anthropology"
    elif any(word in sub_lower for word in ['politics', 'political', 'government', 'policy', 'governance']):
        return "Political Science"
    elif any(word in sub_lower for word in ['economics', 'economic', 'economy', 'market', 'finance', 'financial']):
        return "Economics"
    elif any(word in sub_lower for word in ['geography', 'geographical', 'map', 'location', 'spatial']):
        return "Geography"
    elif any(word in sub_lower for word in ['language', 'linguistic', 'grammar', 'syntax', 'phonetic']):
        return "Linguistics"
    elif any(word in sub_lower for word in ['archaeology', 'archaeological', 'artifact', 'excavation']):
        return "Archaeology"
    
    # Humanities
    elif any(word in sub_lower for word in ['history', 'historical', 'past', 'ancient', 'medieval', 'modern']):
        return "History"
    elif any(word in sub_lower for word in ['philosophy', 'philosophical', 'ethics', 'logic', 'metaphysics']):
        return "Philosophy"
    elif any(word in sub_lower for word in ['literature', 'literary', 'novel', 'poetry', 'prose', 'author']):
        return "Literature"
    elif any(word in sub_lower for word in ['religion', 'religious', 'theology', 'spiritual', 'faith']):
        return "Religious Studies"
    elif any(word in sub_lower for word in ['art', 'artistic', 'painting', 'sculpture', 'artwork']):
        return "Art History"
    elif any(word in sub_lower for word in ['music', 'musical', 'composition', 'melody', 'harmony']):
        return "Music Theory"
    elif any(word in sub_lower for word in ['theater', 'theatre', 'drama', 'performance', 'acting']):
        return "Theater Arts"
    
    # Applied & Professional
    elif any(word in sub_lower for word in ['business', 'management', 'strategy', 'organization', 'corporate']):
        return "Business Administration"
    elif any(word in sub_lower for word in ['law', 'legal', 'court', 'justice', 'jurisprudence']):
        return "Law"
    elif any(word in sub_lower for word in ['education', 'teaching', 'learning', 'pedagogy', 'curriculum']):
        return "Education"
    elif any(word in sub_lower for word in ['communication', 'media', 'journalism', 'news', 'broadcasting']):
        return "Communications"
    elif any(word in sub_lower for word in ['architecture', 'architectural', 'building', 'structure']):
        return "Architecture"
    elif any(word in sub_lower for word in ['agriculture', 'farming', 'crop', 'livestock']):
        return "Agriculture"
    elif any(word in sub_lower for word in ['nutrition', 'diet', 'food', 'nutritional']):
        return "Nutrition"
    
    # Arts & Creative
    elif any(word in sub_lower for word in ['visual', 'graphic', 'design', 'illustration', 'drawing']):
        return "Visual Arts"
    elif any(word in sub_lower for word in ['writing', 'creative', 'story', 'narrative', 'fiction']):
        return "Creative Writing"
    elif any(word in sub_lower for word in ['film', 'movie', 'cinema', 'video', 'documentary']):
        return "Film Studies"
    elif any(word in sub_lower for word in ['photography', 'photo', 'camera', 'image']):
        return "Photography"
    
    # Health & Physical
    elif any(word in sub_lower for word in ['public health', 'epidemiology', 'health policy']):
        return "Public Health"
    elif any(word in sub_lower for word in ['exercise', 'fitness', 'physical', 'sport', 'athletic']):
        return "Physical Education"
    elif any(word in sub_lower for word in ['therapy', 'rehabilitation', 'treatment']):
        return "Therapy & Rehabilitation"
    
    # Modern/Interdisciplinary
    elif any(word in sub_lower for word in ['data', 'analytics', 'big data', 'data science']):
        return "Data Science"
    elif any(word in sub_lower for word in ['security', 'cyber', 'cybersecurity', 'encryption']):
        return "Cybersecurity"
    elif any(word in sub_lower for word in ['biotechnology', 'biotech', 'genetic engineering']):
        return "Biotechnology"
    elif any(word in sub_lower for word in ['cognitive', 'neuroscience', 'brain', 'mind']):
        return "Cognitive Science"
    elif any(word in sub_lower for word in ['international', 'global', 'world affairs']):
        return "International Studies"
    elif any(word in sub_lower for word in ['gender', 'feminist', 'women studies']):
        return "Gender Studies"
    elif any(word in sub_lower for word in ['urban', 'city planning', 'municipal']):
        return "Urban Planning"
    
    return "General Studies"

# Backward compatibility
def get_main_category(sub_category: str) -> str:
    """Backward compatible function - now maps to academic subjects"""
    return get_subject_category_fallback(sub_category)

def get_all_main_categories() -> List[str]:
    """Backward compatible function"""
    return get_all_subject_categories()
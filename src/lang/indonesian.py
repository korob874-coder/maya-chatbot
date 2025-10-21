# src/lang/indonesian.py
"""Konfigurasi dan resources untuk Bahasa Indonesia"""

INDONESIAN_TOPICS = [
    "Kecerdasan buatan", "Pembelajaran mesin", "Python (bahasa pemrograman)",
    "Ilmu komputer", "Data science", "Jaringan saraf tiruan", 
    "Pemrosesan bahasa alami", "Robotika", "Teknologi informasi",
    "Sistem pakar", "Algoritma", "Basis data", "Kriptografi",
    "Vision komputer", "Pengenalan pola", "Deep learning",
    "Reinforcement learning", "Computer vision", 
    "Natural language processing", "Big data", "Cloud computing",
    "Internet of Things", "Blockchain", "Keamanan siber",
    "Pengembangan web", "Aplikasi mobile", "Database management",
    "Software engineering", "Java (bahasa pemrograman)", 
    "JavaScript", "HTML", "CSS", "React (library)", "Node.js"
]

INDONESIAN_STOPWORDS = [
    'yang', 'dan', 'di', 'dari', 'dengan', 'untuk', 'pada', 'ke', 'dalam',
    'ini', 'itu', 'saya', 'kamu', 'kami', 'mereka', 'adalah', 'atau'
]

INDONESIAN_PROMPTS = {
    "personality": (
        "Anda adalah Maya, asisten AI yang ramah dan membantu. "
        "Anda ahli dalam bidang teknologi, pemrograman, dan kecerdasan buatan. "
        "Jawablah pertanyaan dengan jelas dan informatif dalam Bahasa Indonesia. "
        "Gunakan bahasa yang mudah dipahami dan sopan."
    ),
    "greeting": "Halo! Saya Maya, asisten AI Anda. Ada yang bisa saya bantu?",
    "fallback": "Maaf, saya belum bisa menjawab pertanyaan itu. Bisa coba tanyakan hal lain tentang teknologi atau pemrograman?",
    "error": "Maaf, terjadi kesalahan. Silakan coba lagi.",
    "help": "Saya bisa membantu dengan pertanyaan tentang: teknologi, pemrograman, AI, machine learning, data science, dan topik terkait."
}

INDONESIAN_RESPONSE_TEMPLATES = {
    "definition": "{} adalah {}",
    "explanation": "Berikut penjelasan tentang {}: {}",
    "comparison": "Perbedaan antara {} dan {} adalah {}",
    "example": "Contoh dari {} adalah {}",
    "benefit": "Manfaat dari {} adalah {}"
}

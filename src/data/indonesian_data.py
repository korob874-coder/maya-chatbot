# src/data/indonesian_data.py
"""Data loader khusus Bahasa Indonesia"""
import wikipediaapi
import requests
import re
from ..lang.indonesian import INDONESIAN_TOPICS, INDONESIAN_STOPWORDS

class IndonesianDataLoader:
    def __init__(self, language='id'):
        self.language = language
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
    
    def get_wikipedia_text(self, topic, min_length=300):
        """Ambil teks dari Wikipedia Indonesia"""
        try:
            page = self.wiki.page(topic)
            if page.exists():
                return self.clean_indonesian_text(page.text)
            return None
        except:
            return None
    
    def clean_indonesian_text(self, text):
        """Bersihkan teks Bahasa Indonesia"""
        # Hapus referensi [1], [2]
        text = re.sub(r'\[\d+\]', '', text)
        # Hapus section headers
        text = re.sub(r'=+.*?=+', '', text)
        # Hapus templates
        text = re.sub(r'\{\{.*?\}\}', '', text)
        # Hapus HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_multiple_topics(self, topics=None, min_length=200):
        """Ambil multiple topics"""
        if topics is None:
            topics = INDONESIAN_TOPICS
        
        all_text = ""
        successful_topics = []
        
        for topic in topics:
            text = self.get_wikipedia_text(topic, min_length)
            if text and len(text) >= min_length:
                all_text += text + "\n\n"
                successful_topics.append(topic)
                print(f"✅ {topic}: {len(text)} karakter")
            else:
                print(f"❌ {topic}: gagal atau terlalu pendek")
            
            # Delay untuk menghormati server
            import time
            time.sleep(0.5)
        
        print(f"\n📚 Berhasil mengumpulkan {len(successful_topics)} topik")
        print(f"📊 Total teks: {len(all_text)} karakter")
        
        return all_text, successful_topics

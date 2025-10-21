"""Data loader khusus Bahasa Indonesia"""
import wikipediaapi
import requests
import re

class IndonesianDataLoader:
    def __init__(self, language='id', user_agent="MayaChatbot/1.0"):
        self.language = language
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=user_agent  # ← TAMBAHKAN INI
        )
    
    def get_wikipedia_text(self, topic, min_length=300):
        """Ambil teks dari Wikipedia Indonesia"""
        try:
            page = self.wiki.page(topic)
            if page.exists():
                text = self.clean_indonesian_text(page.text)
                if len(text) >= min_length:
                    return text
            return None
        except Exception as e:
            print(f"Error getting {topic}: {e}")
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
            topics = [
                "Kecerdasan buatan",
                "Pembelajaran mesin", 
                "Python (bahasa pemrograman)",
                "Ilmu komputer",
                "Data science"
            ]
        
        all_text = ""
        successful_topics = []
        
        for topic in topics:
            print(f"  Downloading: {topic}")
            text = self.get_wikipedia_text(topic, min_length)
            if text:
                all_text += text + "\n\n"
                successful_topics.append(topic)
                print(f"  ✅ {topic}: {len(text)} karakter")
            else:
                print(f"  ❌ {topic}: gagal atau terlalu pendek")
            
            # Delay untuk menghormati server
            import time
            time.sleep(1)
        
        print(f"\n📚 Berhasil mengumpulkan {len(successful_topics)} topik")
        print(f"📊 Total teks: {len(all_text)} karakter")
        
        return all_text, successful_topics

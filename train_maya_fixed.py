#!/usr/bin/env python3
"""
Maya Chatbot Training - FIXED Version
"""

import torch
import sys
import os

# Add src to path
sys.path.append('/content/maya-chatbot/src')

print("🇮🇩 Maya Chatbot Training - Bahasa Indonesia")
print("=" * 50)

# Import modules
try:
    from model.transformer_model import MayaTransformer
    from data.dataset import TextDataset
    from data.indonesian_data import IndonesianDataLoader
    from training.trainer import Trainer
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

from torch.utils.data import DataLoader

# 1. Download Wikipedia Indonesia data
print("\n📥 Downloading Indonesian Wikipedia data...")

try:
    data_loader = IndonesianDataLoader()
    indonesian_data, topics = data_loader.get_multiple_topics(min_length=100)
    
    if len(indonesian_data) < 500:
        # Fallback: use simple data
        print("⚠️ Using fallback data (Wikipedia response small)...")
        indonesian_data = """
        Kecerdasan buatan adalah bidang ilmu komputer yang menciptakan mesin pintar.
        Machine learning adalah cabang AI yang fokus pada pembelajaran dari data.
        Deep learning menggunakan jaringan saraf untuk belajar pola kompleks.
        Python adalah bahasa pemrograman populer untuk pengembangan AI.
        Natural language processing memungkinkan komputer memahami bahasa manusia.
        Robotika menggabungkan AI dengan teknik mesin dan elektronika.
        Data science menganalisis data untuk mengambil keputusan.
        Neural network terinspirasi dari cara kerja otak manusia.
        Computer vision membantu komputer memahami gambar dan video.
        Pemrograman adalah proses menulis kode untuk aplikasi komputer.
        Algoritma adalah langkah-langkah logis untuk menyelesaikan masalah.
        Basis data menyimpan dan mengelola informasi digital.
        Jaringan komputer menghubungkan perangkat untuk berbagi data.
        Keamanan siber melindungi sistem dari serangan digital.
        Cloud computing menyediakan layanan melalui internet.
        Internet of Things menghubungkan perangkat fisik ke internet.
        Blockchain adalah teknologi ledger terdistribusi yang aman.
        Big data menganalisis dataset yang sangat besar.
        Virtual reality menciptakan lingkungan digital yang imersif.
        Augmented reality menambahkan elemen digital ke dunia nyata.
        """
except Exception as e:
    print(f"❌ Error downloading data: {e}")
    print("⚠️ Using built-in data...")
    indonesian_data = """
    Kecerdasan buatan adalah teknologi masa depan.
    Machine learning belajar dari data secara otomatis.
    Python bahasa pemrograman yang mudah dipelajari.
    Deep learning untuk pattern recognition yang kompleks.
    Natural language processing memahami teks dan bahasa.
    Computer vision menganalisis gambar dan video.
    Robotika mengotomasi tugas fisik.
    Data science mengambil insight dari data.
    Neural network mirip dengan otak manusia.
    AI membantu di healthcare, finance, dan education.
    """
    
print(f"📊 Total training data: {len(indonesian_data)} characters")

# 2. Create dataset
print("\n📊 Creating dataset...")
dataset = TextDataset(text_data=indonesian_data, max_length=64)
print(f"  Sequences: {len(dataset)}")
print(f"  Vocabulary size: {dataset.get_vocab_size()}")

# 3. Create model
print("\n🧠 Creating model...")
model = MayaTransformer(
    vocab_size=dataset.get_vocab_size(),
    d_model=128,    # Small model for CPU
    n_heads=4,
    n_layers=2,
    max_length=64
)
print(f"  Model parameters: {model.get_num_params():,}")

# 4. Create dataloader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 5. Train
print("\n🏋️ Starting training...")
trainer = Trainer(
    model=model,
    dataloader=dataloader,
    dataset=dataset,
    device="cpu",
    learning_rate=0.001
)

# Train for few epochs
trainer.train(num_epochs=5)

# 6. Save model
print("\n💾 Saving model...")
os.makedirs('/content/maya-chatbot/models', exist_ok=True)
trainer.save_model('/content/maya-chatbot/models/maya_indonesian_v1.pth')

print("✅ Training completed successfully!")
print("🎉 Maya Chatbot Bahasa Indonesia is ready!")

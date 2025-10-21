#!/usr/bin/env python3
# scripts/train_id.py
"""Script training khusus Bahasa Indonesia"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.config import MayaConfig
from src.data.indonesian_data import IndonesianDataLoader
from src.model.transformer_model import MayaTransformer
from src.training.trainer import Trainer
import torch

def main():
    print("🇮🇩 Training Maya Chatbot Bahasa Indonesia...")
    
    # Load config Indonesia
    config = MayaConfig('configs/indonesian_small.yaml')
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data Indonesia
    print("📥 Loading data Bahasa Indonesia...")
    data_loader = IndonesianDataLoader()
    text_data, topics = data_loader.get_multiple_topics(
        config.data.topics, 
        config.data.min_text_length
    )
    
    # Create dataset
    from src.data.dataset import TextDataset
    dataset = TextDataset(
        text_data=text_data,
        max_length=config.model.max_length
    )
    
    print(f"📊 Dataset: {len(dataset)} sequences")
    print(f"🔤 Vocabulary: {dataset.get_vocab_size()}")
    
    # Create model
    model = MayaTransformer(
        vocab_size=dataset.get_vocab_size(),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        max_length=config.model.max_length
    )
    
    print(f"🧠 Model: {model.get_num_params():,} parameters")
    
    # Create trainer dan training
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
    
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        learning_rate=config.training.learning_rate
    )
    
    # Training
    print("🏋️ Starting training...")
    trainer.train(num_epochs=config.training.num_epochs)
    
    # Save model
    os.makedirs(config.training.save_dir, exist_ok=True)
    model_path = os.path.join(config.training.save_dir, "maya_indonesian.pth")
    trainer.save_model(model_path)
    
    print(f"✅ Training selesai! Model disimpan di: {model_path}")

if __name__ == "__main__":
    main()

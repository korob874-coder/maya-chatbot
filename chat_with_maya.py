#!/usr/bin/env python3
"""
Maya Chatbot Interface - Bahasa Indonesia
"""

import torch
import sys
import os
import re

sys.path.append('/content/maya-chatbot/src')

from model.transformer_model import MayaTransformer
from data.dataset import TextDataset

class MayaChatbot:
    def __init__(self, model_path):
        print("🤖 Memuat Maya Chatbot...")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        self.dataset = checkpoint['dataset']
        
        self.model = MayaTransformer(
            vocab_size=self.dataset.get_vocab_size(),
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_length=64
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.conversation_history = []
        
        print("✅ Maya siap! Mari ngobrol...")
        print(f"📊 Vocabulary: {self.dataset.get_vocab_size()} karakter")
    
    def generate_response(self, prompt, max_length=100, temperature=0.7):
        """Generate response dari prompt"""
        with torch.no_grad():
            # Encode prompt
            encoded = []
            for char in prompt:
                if char in self.dataset.char_to_idx:
                    encoded.append(self.dataset.char_to_idx[char])
                else:
                    encoded.append(0)  # unknown character
            
            if len(encoded) == 0:
                return "Maaf, saya tidak memahami input Anda."
            
            input_seq = torch.tensor(encoded).unsqueeze(0)
            generated = prompt
            
            for _ in range(max_length):
                output = self.model(input_seq)
                logits = output[0, -1, :] / temperature
                
                # Apply softmax dan sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                next_char = self.dataset.idx_to_char.get(next_token, '')
                
                if not next_char:
                    break
                
                generated += next_char
                
                # Update input sequence
                new_token = torch.tensor([[next_token]])
                if input_seq.size(1) >= self.model.max_length:
                    input_seq = torch.cat([input_seq[:, 1:], new_token], dim=1)
                else:
                    input_seq = torch.cat([input_seq, new_token], dim=1)
            
            # Extract hanya response bagian baru
            response = generated[len(prompt):].strip()
            return self.clean_response(response)
    
    def clean_response(self, text):
        """Bersihkan response"""
        # Hentikan di akhir kalimat
        for end_char in ['.', '!', '?', '\n']:
            pos = text.find(end_char)
            if pos != -1:
                return text[:pos + 1].strip()
        
        return text.strip()
    
    def chat(self, message):
        """Fungsi chat utama"""
        # Build context dari history
        context = ""
        if self.conversation_history:
            context = "\n".join(self.conversation_history[-4:]) + "\n"
        
        prompt = f"{context}User: {message}\nMaya:"
        response = self.generate_response(prompt, max_length=80, temperature=0.7)
        
        # Simpan ke history
        self.conversation_history.append(f"User: {message}")
        self.conversation_history.append(f"Maya: {response}")
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response

def main():
    print("=" * 50)
    print("🤖 MAYA CHATBOT - Bahasa Indonesia")
    print("=" * 50)
    
    model_path = "/content/maya-chatbot/models/maya_indonesian_v1.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan di: {model_path}")
        print("Jalankan training dulu: python train_maya_fixed.py")
        return
    
    chatbot = MayaChatbot(model_path)
    
    print("\n💬 Mulai ngobrol dengan Maya!")
    print("Ketik 'quit' untuk keluar")
    print("Ketik 'clear' untuk menghapus memory")
    print("-" * 30)
    
    while True:
        try:
            user_input = input("\n👤 Anda: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'keluar']:
                print("🤖 Maya: Sampai jumpa! Terima kasih sudah ngobrol 😊")
                break
            elif user_input.lower() == 'clear':
                chatbot.conversation_history = []
                print("🤖 Maya: Memory dihapus. Mari mulai percakapan baru!")
                continue
            elif user_input == '':
                continue
            
            print("🤖 Maya: ", end='', flush=True)
            response = chatbot.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n🤖 Maya: Dadah! Jumpa lagi 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

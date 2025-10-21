#!/usr/bin/env python3
# scripts/chat_id.py
"""Script chat khusus Bahasa Indonesia"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chatbot.indonesian_bot import MayaIndonesianChatbot

def main():
    print("🤖 Maya Chatbot - Bahasa Indonesia")
    print("=" * 40)
    
    # Load model (default path)
    model_path = "checkpoints/indonesian/maya_indonesian.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan di: {model_path}")
        print("Jalankan training dulu: python scripts/train_id.py")
        return
    
    chatbot = MayaIndonesianChatbot(model_path=model_path)
    
    print("Maya: Halo! Saya Maya, asisten AI Anda. Ada yang bisa saya bantu?")
    print("\nPerintah:")
    print("  'clear' - Hapus riwayat percakapan")
    print("  'history' - Lihat riwayat percakapan") 
    print("  'quit' - Keluar")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n👤 Anda: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'keluar']:
                print("Maya: Sampai jumpa! 👋")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("Maya: Riwayat percakapan dihapus!")
                continue
            elif user_input.lower() == 'history':
                history = chatbot.get_history()
                if history:
                    print("\n--- Riwayat Percakapan ---")
                    for msg in history[-6:]:  # Last 3 exchanges
                        print(msg)
                else:
                    print("Belum ada percakapan.")
                continue
            elif user_input == '':
                continue
            
            response = chatbot.chat(user_input)
            print(f"🤖 Maya: {response}")
            
        except KeyboardInterrupt:
            print("\n\nMaya: Sampai jumpa! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

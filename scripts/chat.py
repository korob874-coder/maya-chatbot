# scripts/chat.py
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chatbot.interface import MayaChatbot

def main():
    print("🤖 Starting Maya Chatbot...")
    
    # Load model (ganti dengan path model Anda)
    model_path = "checkpoints/maya_model.pth"
    
    chatbot = MayaChatbot(model_path=model_path)
    
    print("Maya: Hi! I'm Maya, your AI assistant. How can I help you today?")
    print("Type 'quit' to exit, 'clear' to clear history")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Maya: Goodbye! 👋")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("Maya: Conversation history cleared!")
                continue
            elif user_input == '':
                continue
            
            response = chatbot.chat(user_input)
            print(f"Maya: {response}")
            
        except KeyboardInterrupt:
            print("\n\nMaya: Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()

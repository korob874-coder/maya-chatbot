# src/chatbot/interface.py
import torch
import re
from typing import List, Dict
from ..model.transformer_model import MayaTransformer
from ..data.dataset import TextDataset

class MayaChatbot:
    def __init__(self, model_path: str = None, device: str = "auto"):
        self.device = self._setup_device(device)
        self.model = None
        self.dataset = None
        self.conversation_history = []
        self.personality_traits = {
            "name": "Maya",
            "tone": "friendly",
            "knowledge_domain": "AI and technology"
        }
        
        if model_path:
            self.load_model(model_path)
    
    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self, model_path: str):
        """Load trained model and dataset"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load dataset info
        self.dataset = checkpoint['dataset']
        
        # Initialize model
        self.model = MayaTransformer(
            vocab_size=self.dataset.get_vocab_size(),
            d_model=checkpoint['config']['d_model'],
            n_heads=checkpoint['config']['n_heads'],
            n_layers=checkpoint['config']['n_layers']
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"🤖 Maya model loaded successfully!")
        print(f"📊 Vocabulary size: {self.dataset.get_vocab_size()}")
        print(f"⚙️  Device: {self.device}")
    
    def chat(self, message: str, max_length: int = 150, 
             temperature: float = 0.8, top_k: int = 50) -> str:
        """Main chat method"""
        if not self.model:
            return "Error: Model not loaded. Please load a trained model first."
        
        # Add to history
        self.conversation_history.append(f"User: {message}")
        
        # Build context
        context = self._build_context()
        prompt = self._create_prompt(context, message)
        
        # Generate response
        response = self._generate_response(
            prompt, max_length, temperature, top_k
        )
        
        # Add to history and return
        self.conversation_history.append(f"Maya: {response}")
        return response
    
    def _create_prompt(self, context: str, message: str) -> str:
        """Create formatted prompt for generation"""
        personality = (
            "You are Maya, a friendly AI assistant specialized in "
            "artificial intelligence and technology. "
            "Be helpful, concise, and accurate in your responses."
        )
        
        return f"{personality}\n\n{context}\nUser: {message}\nMaya:"
    
    def _generate_response(self, prompt: str, max_length: int, 
                          temperature: float, top_k: int) -> str:
        """Generate text response"""
        with torch.no_grad():
            encoded = self.dataset.encode_text(prompt)
            input_seq = torch.tensor(encoded).unsqueeze(0).to(self.device)
            
            generated = prompt
            
            for _ in range(max_length):
                output = self.model(input_seq)
                logits = output[0, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                next_char = self.dataset.idx_to_char.get(next_token, '')
                
                if not next_char:
                    break
                
                generated += next_char
                
                # Update input
                new_token = torch.tensor([[next_token]]).to(self.device)
                if input_seq.size(1) >= self.model.max_length:
                    input_seq = torch.cat([input_seq[:, 1:], new_token], dim=1)
                else:
                    input_seq = torch.cat([input_seq, new_token], dim=1)
            
            # Extract only the new response
            return self._extract_response(generated, prompt)
    
    def _build_context(self) -> str:
        """Build context from conversation history"""
        if len(self.conversation_history) <= 2:
            return ""
        return "\n".join(self.conversation_history[-4:])  # Last 2 exchanges
    
    def _extract_response(self, full_text: str, prompt: str) -> str:
        """Extract only the AI response"""
        if prompt in full_text:
            response = full_text[len(prompt):]
        else:
            response = full_text
        
        # Clean up response
        response = re.sub(r'\n+', ' ', response)
        response = re.sub(r'\s+', ' ', response)
        
        # Truncate at last complete sentence
        for end_char in ['.', '!', '?']:
            last_pos = response.rfind(end_char)
            if last_pos != -1:
                return response[:last_pos + 1].strip()
        
        return response.strip()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[str]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def set_personality(self, name: str = None, tone: str = None):
        """Customize chatbot personality"""
        if name:
            self.personality_traits["name"] = name
        if tone:
            self.personality_traits["tone"] = tone

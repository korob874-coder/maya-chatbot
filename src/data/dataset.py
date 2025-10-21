# 2. Buat file dataset
%%writefile /content/maya-chatbot/src/data/dataset.py
import torch
from torch.utils.data import Dataset
import re

class TextDataset(Dataset):
    def __init__(self, text_data, max_length=128):
        self.text_data = text_data
        self.max_length = max_length
        
        self.vocab, self.char_to_idx, self.idx_to_char = self.build_vocab(text_data)
        self.encoded_data = self.encode_text(text_data)
    
    def build_vocab(self, text):
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        return len(chars), char_to_idx, idx_to_char
    
    def encode_text(self, text):
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode_text(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char])
    
    def __len__(self):
        return len(self.encoded_data) - self.max_length
    
    def __getitem__(self, idx):
        x = self.encoded_data[idx:idx + self.max_length]
        y = self.encoded_data[idx + 1:idx + self.max_length + 1]
        return torch.tensor(x), torch.tensor(y)
    
    def get_vocab_size(self):
        return self.vocab

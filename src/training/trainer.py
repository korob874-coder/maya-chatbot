# 3. Buat file trainer
%%writefile /content/maya-chatbot/src/training/trainer.py
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, dataloader, dataset, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.dataset = dataset
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(self.dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)), 
                y.view(-1)
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(self.dataloader)} | Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s')
            
            # Generate sample text setiap 2 epoch
            if epoch % 2 == 0:
                self.generate_sample_text(epoch)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
    def generate_sample_text(self, epoch, length=50):
        self.model.eval()
        start_text = "AI adalah"
        
        with torch.no_grad():
            encoded = [self.dataset.char_to_idx.get(ch, 0) for ch in start_text]
            input_seq = torch.tensor(encoded).unsqueeze(0).to(self.device)
            
            generated = start_text
            
            for _ in range(length):
                output = self.model(input_seq)
                next_token = output[0, -1].argmax().item()
                
                next_char = self.dataset.idx_to_char.get(next_token, '')
                generated += next_char
                
                new_token = torch.tensor([[next_token]]).to(self.device)
                if input_seq.size(1) >= self.model.max_length:
                    input_seq = torch.cat([input_seq[:, 1:], new_token], dim=1)
                else:
                    input_seq = torch.cat([input_seq, new_token], dim=1)
            
            print(f"\n--- Epoch {epoch} Sample ---")
            print(f"{generated}\n")
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'dataset': self.dataset,
        }, path)
        print(f"Model saved to {path}")

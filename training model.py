

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# STABLE Hyperparameters for 2GB GPU
batch_size = 64           # Fixed small batch
block_size = 512             # Short context
max_iters = 18000            # Quick training
eval_interval = 100
learning_rate = 5e-4
n_embd = 256                # Small embeddings
n_head = 8                # Minimal heads
n_layer = 8           # Minimal layers
dropout = 0.1               # No dropout

torch.manual_seed(1337)

# Simple character-level tokenizer
def create_simple_tokenizer(text):
    """Create a simple character-level tokenizer for Hindi"""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    return encode, decode, vocab_size

# Load small Hindi dataset
# Load Hindi dataset safely from large file
print("📖 Loading Hindi dataset...")
try:
    max_chars = 50 * 1024 * 1024  # 50MB worth of characters
    buffer = []
    total_chars = 0

    with open('hindi_1gb_fixed.txt', 'r', encoding='utf-8') as f:
        for line in f:
            buffer.append(line)
            total_chars += len(line)
            if total_chars >= max_chars:
                break

    text = ''.join(buffer)
    print(f"Loaded {len(text):,} characters from hindi_4gb.txt")

except FileNotFoundError:
    text = "नमस्ते आज का दिन अच्छा है। भारत देश महान है। " * 50
    print("Using tiny sample dataset")

# Create simple tokenizer
encode, decode, vocab_size = create_simple_tokenizer(text)
print(f"Vocabulary size: {vocab_size}")

# Encode text
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Encoded tokens: {len(data):,}")

# Split data
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")

# STABLE Data loader - no dynamic changes
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Simple Model
class NanoGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Simple attention + FFN
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            nn.ReLU(),
            nn.Linear(2 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        # Single transformer block
        attn_mask = self.mask[:T, :T] == 0
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model
print("🚀 Initializing NanoGPT...")
model = NanoGPT(vocab_size)
model = model.to(device)

# Print model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params/1e6:.2f} M")

# Simple optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# STABLE Training loop - no dynamic changes
print("🔥 Starting training...")
start_time = time.time()

for iter in range(max_iters):
    # Single batch training (no gradient accumulation)
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Clear memory every step for 2GB GPU
    del xb, yb, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Simple evaluation
    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            train_loss = 0
            for _ in range(10):  # Quick estimate
                xb, yb = get_batch('train')
                _, loss = model(xb, yb)
                train_loss += loss.item()
            train_loss /= 10
            
            val_loss = 0
            for _ in range(10):
                xb, yb = get_batch('val')
                _, loss = model(xb, yb)
                val_loss += loss.item()
            val_loss /= 10
        
        elapsed = (time.time() - start_time) / 60
        
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"Step {iter:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}m | GPU: {memory:.2f}GB")
        else:
            print(f"Step {iter:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}m")
        
        # Generate sample
        if iter % 200 == 0:
            with torch.no_grad():
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=20, temperature=0.8)
                text_out = decode(generated[0].tolist())
                print(f"Sample: {text_out}")
        
        model.train()

# Final generation
print("\n🎭 Final generation:")
model.eval()
with torch.no_grad():
    context = torch.tensor([encode("नमस्ते")], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=1100, temperature=0.7)
    final_text = decode(generated[0].tolist())
    print(f"Final: {final_text}")

# Save model
print("\n💾 Saving model...")
torch.save({
    'model_state': model.state_dict(),
    'config': {
        'n_embd': n_embd, 'n_head': n_head, 
        'block_size': block_size, 'vocab_size': vocab_size
    }
}, 'nano_hindi_gpt.pth')

print("✅ Training complete!")
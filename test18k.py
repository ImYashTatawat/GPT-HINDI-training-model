import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------
# CONFIG (must match training)
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT = "nano_hindi_gpt.pth"
DATA_FILE = "hindi_1gb_fixed.txt"
BLOCK_SIZE = 512

print("Using device:", DEVICE)

# -----------------------------------
# SIMPLE CHARACTER TOKENIZER (same as training)
# -----------------------------------
def create_simple_tokenizer(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s if c in stoi]

    def decode(toks):
        return ''.join([itos[i] for i in toks])

    return encode, decode, len(chars)


# ---------------------------
# Load vocabulary from dataset
# ---------------------------
print("Loading tokenizer dataset...")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    # only load first 50MB to save RAM
    max_chars = 50 * 1024 * 1024
    buf = []
    total = 0
    for line in f:
        buf.append(line)
        total += len(line)
        if total >= max_chars:
            break

text = ''.join(buf)
encode, decode, vocab_size = create_simple_tokenizer(text)

print("Vocab size:", vocab_size)

# ---------------------------
# Rebuild the GPT model
# ---------------------------
class NanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 2*n_embd),
            nn.ReLU(),
            nn.Linear(2*n_embd, n_embd)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Causal mask
        self.register_buffer("mask",
                             torch.tril(torch.ones(block_size, block_size)))

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))

        x = tok + pos

        attn_mask = (self.mask[:T, :T] == 0)
        out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                           attn_mask=attn_mask)
        x = x + out
        x = x + self.ffn(self.ln2(x))

        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens=200, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------
# Load model checkpoint
# ---------------------------
print("Loading model checkpoint...")

chk = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
config = chk["config"]

N_LAYER = 8   # <-- SET YOUR TRAINED NUMBER OF LAYERS

model = NanoGPT(
    vocab_size,
    config["n_embd"],
    config["n_head"],
    N_LAYER,
    config["block_size"]
).to(DEVICE)


model.load_state_dict(chk["model_state"])
model.eval()

print("Model loaded!")

# ---------------------------
# GENERATE TEXT
# ---------------------------

prompt = " नमस्ते आज का दिन अच्छा है। भारत देश महान है"
print("\nPrompt:", prompt)

idx = torch.tensor([encode(prompt)], dtype=torch.long).to(DEVICE)

with torch.no_grad():
    out = model.generate(idx, max_new_tokens=1200, temperature=0.9)

generated_text = decode(out[0].tolist())

print("\nGenerated text:\n")
print(generated_text)
print("\n----------------------------")

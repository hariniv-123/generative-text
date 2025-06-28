from transformers import pipeline

print("Loading GPT-2 model...")
generator = pipeline('text-generation', model='gpt2')

prompt = input("Enter a topic or prompt for GPT-2: ")
gpt2_outputs = generator(prompt, max_length=120, num_return_sequences=1, temperature=0.8)
print("\nGPT-2 Generated Paragraph:\n")
print(gpt2_outputs[0]['generated_text'])

# --- Optional: Minimal LSTM-based Character Generator ---

import torch
import torch.nn as nn
import numpy as np

# Tiny corpus for LSTM (change as needed)
corpus = (
    "Machine learning enables computers to learn from data. "
    "Deep learning is a subset of machine learning. "
    "Natural language processing is a field of AI."
).lower()

# Build vocabulary
chars = sorted(list(set(corpus)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

seq_length = 40
step = 3

sentences = []
next_chars = []
for i in range(0, len(corpus) - seq_length, step):
    sentences.append(corpus[i: i + seq_length])
    next_chars.append(corpus[i + seq_length])

def one_hot_encode(sequence, vocab_size):
    out = np.zeros((len(sequence), vocab_size), dtype=np.float32)
    for i, idx in enumerate(sequence):
        out[i, idx] = 1.0
    return out

X = [one_hot_encode([char2idx[c] for c in seq], vocab_size) for seq in sentences]
y = [char2idx[c] for c in next_chars]
X = torch.tensor(np.array(X))
y = torch.tensor(y)

class LSTMTextGen(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, prev_state=None):
        output, state = self.lstm(x, prev_state) if prev_state else self.lstm(x)
        logits = self.fc(output)
        return logits, state

model = LSTMTextGen(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
epochs = 20

print("\nTraining minimal LSTM model (few seconds)...")
for epoch in range(epochs):
    optimizer.zero_grad()
    output, _ = model(X)
    output = output[:, -1, :]
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def generate_lstm(seed, length=200, temperature=1.0):
    model.eval()
    seed = seed.lower()
    generated = seed
    input_seq = seed[-seq_length:]
    for _ in range(length):
        x = one_hot_encode([char2idx.get(c, 0) for c in input_seq], vocab_size)
        x = torch.tensor(x).unsqueeze(0)
        logits, _ = model(x)
        logits = logits[0, -1, :] / temperature
        probas = torch.softmax(logits, dim=0).detach().numpy()
        idx = np.random.choice(range(vocab_size), p=probas)
        next_char = idx2char[idx]
        generated += next_char
        input_seq = (input_seq + next_char)[-seq_length:]
    return generated

seed_text = input("\nEnter a seed for LSTM text generation: ")
print("\nLSTM Generated Text:\n")
print(generate_lstm(seed_text, length=300, temperature=0.8))
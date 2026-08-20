import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

#build dataset
file_path = Path(__file__).parent / "input.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi(c) for c in s]
decode = lambda l: ''.join[itos(i) for i in l]

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(text))
train_data = data[:n]
val_data = data[n:]


def get_batch(mode):
    data = train_data if mode == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y






%%writefile pre_train.py
import torch
import torch.nn.functional as F
from vengpt.gpt import VenGPT
from vengpt.dataset import PretrainDataset
from tokenizers import Tokenizer

block_size = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = Tokenizer.from_file("tokenizer.json")
BOS_ID = tokenizer.token_to_id("<[BOS]>")
EOS_ID = tokenizer.token_to_id("<[EOS]>")

pd = PretrainDataset()

token_ids = []

for index, text in enumerate(pd()):
  token_ids += [BOS_ID] + tokenizer.encode(text.get("text", "")).ids + [EOS_ID]

data = torch.tensor(token_ids, dtype=torch.long)
vocab_size = tokenizer.get_vocab_size()

model = VenGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4
)
model.to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

def get_batch(data, batch_size=16):
    ix = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(DEVICE), y.to(DEVICE)

for step in range(20000):
    x, y = get_batch(data, batch_size=32)

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
        ignore_index=-100
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "embed_dim": 128,
            "num_heads": 4,
            "num_layers": 4,
        },
    },
    "model.safetensors",
)

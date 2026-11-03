import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from vengpt.gpt import VenGPT
from vengpt.dataset import SFTDataset
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = Tokenizer.from_file("tokenizer.json")
BOS_ID = tokenizer.token_to_id("<[BOS]>")
EOS_ID = tokenizer.token_to_id("<[EOS]>")
EOT_ID = tokenizer.token_to_id("<[EOT]>")
USER_ID = tokenizer.token_to_id("<[{USER}]>")
ASSISTANT_ID = tokenizer.token_to_id("<[{ASSISTANT}]>")

vocab_size = tokenizer.get_vocab_size()

def encode_conv(conv):
    ids, labels = [BOS_ID], [-100]
    for m in conv:
        if m["role"] == "user":
            seg = [USER_ID] + tokenizer.encode(m["content"]).ids + [EOT_ID]
            ids += seg
            labels += [-100] * len(seg)
        else:
            header = [ASSISTANT_ID]
            body = tokenizer.encode(m["content"]).ids + [EOT_ID]
            ids += header + body
            labels += [-100] + body + [EOT_ID]
    ids.append(EOS_ID); labels.append(EOS_ID)
    return ids, labels

checkpoint = torch.load(
    "model.safetensors",
    map_location=DEVICE,
    weights_only=True
)

config = checkpoint["config"]

model = VenGPT(**config)
model.to(DEVICE)
model.load_state_dict(
   checkpoint["model_state_dict"]
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5
)

sd = SFTDataset()

xs, ys = [], []

class SFTTrainDataset(Dataset):
    def __init__(self, source, block_size):
        self.samples = []
        for sft in source():
            x, y = encode_conv(sft["messages"])
            # Chia conversation thành các block
            for i in range(0, len(x), block_size):
                bx = x[i:i + block_size]
                by = y[i:i + block_size]
                if len(bx) < 2:
                    continue
                self.samples.append((bx, by))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

block_size = config["block_size"]
batch_size = 8

def collate_fn(batch):
    xs, ys = zip(*batch)

    x = torch.full(
        (len(xs), block_size),
        EOS_ID,
        dtype=torch.long
    )

    y = torch.full(
        (len(ys), block_size),
        -100,
        dtype=torch.long
    )

    for i, (a, b) in enumerate(zip(xs, ys)):
        x[i, :len(a)] = torch.tensor(a)
        y[i, :len(b)] = torch.tensor(b)

    return x, y

dataset = SFTTrainDataset(sd, block_size)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)


model.train()

for step in range(15000):

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        logits = logits[:, :-1, :]
        targets = y[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=-100
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(step, loss.item())

        break


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": config
    },
    "modelv2.safetensors"
)

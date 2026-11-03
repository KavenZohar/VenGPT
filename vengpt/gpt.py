import torch
import torch.nn as nn
import torch.nn.functional as F
from vengpt.transformer import TransformerBlock

class VenGPT(nn.Module):
        def __init__(self, vocab_size, block_size, embed_dim=64, num_heads=4, num_layers=2):
                    super().__init__()

                            self.block_size = block_size
                                    self.token_embedding = nn.Embedding(vocab_size, embed_dim)
                                            self.position_embedding = nn.Embedding(block_size, embed_dim)
                                                    self.blocks = nn.Sequential(*[
                                                                    TransformerBlock(embed_dim, num_heads, block_size)
                                                                                for _ in range(num_layers)
                                                                                        ])
                                                            self.final_norm = nn.LayerNorm(embed_dim)
                                                                    self.lm_head = nn.Linear(embed_dim, vocab_size)

                                                                        def forward(self, idx):
                                                                                    B, T = idx.shape

                                                                                            positions = torch.arange(T, device=idx.device)

                                                                                                    token_emb = self.token_embedding(idx)
                                                                                                            pos_emb = self.position_embedding(positions)

                                                                                                                    x = token_emb + pos_emb
                                                                                                                            x = self.blocks(x)
                                                                                                                                    x = self.final_norm(x)

                                                                                                                                            logits = self.lm_head(x)
                                                                                                                                                    return logits

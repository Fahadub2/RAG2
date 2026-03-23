import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_heads"]
        self.head_dim = config["hidden_size"] // config["num_heads"]
        self.hidden_size = config["hidden_size"]

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(x)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config["hidden_size"])
        self.ffn_norm = RMSNorm(config["hidden_size"])

    def forward(self, x, mask=None):
        # Pre-norm architecture
        normed = self.attention_norm(x)
        attn_output = self.attention(normed, mask)
        x = x + attn_output

        normed = self.ffn_norm(x)
        ffn_output = self.feed_forward(normed)
        x = x + ffn_output

        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.position_embedding = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config["num_layers"])
        ])

        self.norm = RMSNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(config["dropout"])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape

        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)

        # Create causal mask
        mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {"logits": logits, "loss": loss}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids)
                next_token_logits = outputs["logits"][:, -1, :] / temperature

                # Top-k sampling
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(top_k_values, dim=-1)
                next_token = torch.gather(top_k_indices, -1, torch.multinomial(probs, 1))

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if input_ids.shape[1] > self.config["max_position_embeddings"]:
                    break

        return input_ids
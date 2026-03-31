import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from model.config import SLMConfig
from model.model import LayerNorm, FeedForward

# ==============================
# KV Cache Attention
# ==============================

class CausalMultiHeadAttentionWithCache(nn.Module):
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.W_query = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_key = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_value = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.d_emb, config.d_emb)
        self.n_heads = config.n_heads
        self.head_dim = config.d_emb // config.n_heads

        # KV cache buffers (persistent=False ensures they aren't saved in state_dict)
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

     
        if self.k_cache is None:
            self.k_cache, self.v_cache = K, V
        else:
            self.k_cache = torch.cat([self.k_cache, K], dim=2)
            self.v_cache = torch.cat([self.v_cache, V], dim=2)

        
        K_full, V_full = self.k_cache, self.v_cache
        scores = (Q @ K_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal masking only during the initial prefill (when T > 1)
        if T > 1:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ V_full).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# ==============================
# Transformer Block & GPT Model
# ==============================

class TransformerBlockWithCache(nn.Module):
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.norm1 = LayerNorm(config)
        self.attention = CausalMultiHeadAttentionWithCache(config)
        self.norm2 = LayerNorm(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class GPTWithKVCache(nn.Module):
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_emb)
        self.position_embedding = nn.Embedding(config.n_blocks, config.d_emb)
        self.blocks = nn.ModuleList([TransformerBlockWithCache(config) for _ in range(config.n_layers)])
        self.final_norm = LayerNorm(config)
        self.head = nn.Linear(config.d_emb, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.current_pos = 0 # Track position for incremental generation

    def reset_cache(self):
        self.current_pos = 0
        for block in self.blocks:
            block.attention.k_cache = None
            block.attention.v_cache = None

    def forward(self, x):
        B, T = x.shape
        # Use stored current_pos to get correct positional embeddings for incremental tokens
        positions = torch.arange(self.current_pos, self.current_pos + T, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        self.current_pos += T # Advance position tracker
        return self.head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.reset_cache()
        # PREFILL- Process the entire prompt at once
        logits = self(idx)
        
        # DECODE- Generate tokens one-by-one using only the newest token
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            
            logits = self(idx_next)
        return idx

# ==============================
# Execution Script
# ==============================

if __name__ == "__main__":
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    config = SLMConfig()
    model = GPTWithKVCache(config)

    
    state_dict = torch.load("best_model_parameter.pt", map_location="cpu")
    new_state_dict = {k.replace("transformer_blocks", "blocks").replace("output_head", "head"): v 
                      for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    prompt = "There was a little girl named Lily"
    x = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    
    generated = model.generate(x, max_new_tokens=60, temperature=0.7, top_k=40)
    print(tokenizer.decode(generated[0].tolist()))

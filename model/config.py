class SLMConfig:
    def __init__(self):
        self.vocab_size = 50257      # GPT-2 vocab
        self.n_blocks = 128          # Max sequence length (your model uses this)
        self.n_layers = 8            # Transformer blocks
        self.n_heads = 8             # Attention heads  
        self.d_emb = 512             # Embedding dim
        self.head_d_emb = 64         # 512/8 = 64 per head
        self.drop_rate = 0.1         # Dropout
        self.qkv_bias = True         # QKV bias
        self.is_debug = True         # Enable debug prints for learning

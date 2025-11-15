import math 
from functools import partial
from dataclasses import dataclass

import torch 
import torch.nn.functional as F

from nanochat.common import get_base_dir
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304 # the vocab of the inputs consider alphabet 26
    n_layer: int = 12 # number of blocks layers of transformer
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6
    n_embed: int = 768
    
def norm(x):
    # purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1)),) # normalize across the actual embeddings

def apply_rotary_emb(x, cos, sin):
    assert x.dim == 4 # is a multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out
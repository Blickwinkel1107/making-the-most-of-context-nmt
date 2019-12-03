import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, demb, dropout=0):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)
        self.drop = nn.Dropout(dropout)
        self.scale = demb ** 0.5

    def forward(self, word_emb, pos_seq=None):
        # word_emb: [bsz, len, d]
        # pos_seq: None or [bsz, len]
        sizes = word_emb.size()

        if pos_seq is None:
            pos_seq = torch.arange(word_emb.size(1), device=word_emb.device, dtype=torch.float32)
        
        sinusoid_inp = torch.ger(pos_seq.view(-1), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if pos_seq.dim() > 1:
            pos_emb = pos_emb.view(*sizes)
        else:
            pos_emb = pos_emb[None, :, :].expand_as(word_emb)

        emb = word_emb * self.scale + pos_emb
        return self.drop(emb)


class SegmentEmbedding(nn.Module):
    def __init__(self, d_emb, max_segment=20):
        super().__init__()

        self.d_emb = d_emb
        self.embed = nn.Embedding(max_segment, d_emb)

    def forward(self, segment_ids):
        # segment_ids: [bsz, len]. for each batch, like [0,0,0,1,1,1,1,2,2,...]
        emb = self.embed(segment_ids)
        return emb
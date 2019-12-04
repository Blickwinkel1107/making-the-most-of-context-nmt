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

        nn.init.normal_(self.embed.weight, mean=0, std=1/d_emb**0.5)

    def forward(self, segment_ids):
        # segment_ids: [bsz, len]. for each batch, like [0,0,0,1,1,1,1,2,2,...]
        emb = self.embed(segment_ids)
        return emb


def get_relative_position_matrix(length, max_relative_position, direction, offset=True):
    """ Generate matrix of relative positions between inputs ([..., length])."""
    range_vec = torch.arange(length).long()
    if torch.cuda.is_available():
        range_vec = range_vec.cuda()
    range_mat = range_vec[:, None].expand(length, length)
    distance_mat = range_mat - range_mat.transpose(0, 1)
    if max_relative_position is None:
        distance_mat_clipped = distance_mat
    else:
        distance_mat_clipped = torch.clamp(distance_mat,
                                           -max_relative_position, max_relative_position)
    if direction:
        # Shift values to be >= 0. Each integer still uniquely identifies a relative
        # position difference.
        if offset and max_relative_position is not None:
            final_mat = distance_mat_clipped + max_relative_position
        else:
            final_mat = distance_mat_clipped
    else:
        # Do not distinguish the forward and backward positions.
        # Just leave the absolute relative position representation.
        final_mat = distance_mat_clipped.abs()
    return final_mat


class RelativePositionEmbeddings(nn.Module):
    """ Relative Position Representation in "Self-Attention with Relative Position Representations"
        (https://arxiv.org/pdf/1803.02155.pdf)
    Implementation inspired by
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    def __init__(self,
                 max_relative_position,
                 embedding_dim,
                 dropout=0.0):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings=max_relative_position*2+1,
                                       embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        from src.utils.init import embedding_init
        embedding_init(self.embeddings.weight)

    def forward(self, length):
        """Generate tensor of size [length, length, depth]"""
        relative_position_matrix = get_relative_position_matrix(
            length, self.max_relative_position, direction=True
        )
        embeddings = self.embeddings(relative_position_matrix)
        embeddings = self.dropout(embeddings)
        return embeddings


class RelativeSegmentEmbeddings(RelativePositionEmbeddings):
    def _build_segment_relative_distance(self, segment_ids):
        # segment_ids: [bsz, len]. [[0,0,0,1,1,1,2,2,2,2...], ...]
        # return: [bsz, qlen, mlen] (qlen==mlen). [[[0,0,0,1,1,1,2,2,2,2...], [-1,-1,-1,0,0,0,1,1,1,1...],...]]
        rel_dis = segment_ids[:, None, :] - segment_ids[:, :, None]
        if self.max_relative_position > -1:
            rel_dis = torch.clamp(rel_dis, -self.max_relative_position, self.max_relative_position)

        return rel_dis + self.max_relative_position

    def forward(self, segment_ids):
        relative_position_matrix = \
            self._build_segment_relative_distance(segment_ids)
        embeddings = self.embeddings(relative_position_matrix)
        embeddings = self.dropout(embeddings)
        return embeddings

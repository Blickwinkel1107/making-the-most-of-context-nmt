import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sublayers import MultiHeadedAttention


class MultiHeadedAttentionRelative(MultiHeadedAttention):
    def _compute_relative_attention(self, q, k, v, mask, rel_k, rel_v, dropout):
        """Calculate relative position-aware dot-product self-attention.
        The attention calculation is augmented with learned representations for the
        relative position between each element in q and each element in k and v.
        \alpha = softmax( q(k+rel_k) ); out = \alpha (v+rel_v)
        Args:
            q: a Tensor with shape [batch, heads, qlen, depth].
            k: a Tensor with shape [batch, heads, klen, depth].
            v: a Tensor with shape [batch, heads, klen, depth].
            bias: bias Tensor.
            relative_embedding_keys: a Tensor with shape [(bsz), qlen, klen, depth].
            relative_embedding_values: a Tensor with shape [(bsz), qlen, klen, depth].
            dropout (optional): nn.Dropout.

        Returns:
            Attention weights. [batch, heads, qlen, klen]
            Attention outputs. [batch, heads, qlen, depth]
        """
        QK = torch.einsum("bhqd,bhkd->bhqk", [q, k])
        if rel_k.dim() == 3:
            QR = torch.einsum("bhqd,qkd->bhqk", [q, rel_k])
        elif rel_k.dim() == 4:
            QR = torch.einsum("bhqd,bqkd->bhqk", [q, rel_k])
        logits = QK + QR

        # [bsz, head, qlen, klen]
        if mask is not None:
            logits = logits.masked_fill(mask, -1e18)
        alpha = F.softmax(logits, -1)
        if dropout is not None:
            alpha = dropout(alpha)

        AV = torch.einsum("bhqk,bhkd->bhqd", [alpha, v])
        if rel_v.dim() == 3:
            AR = torch.einsum("bhqk,qkd->bhqd", [alpha, rel_v])
        elif rel_v.dim() == 4:
            AR = torch.einsum("bhqk,bqkd->bhqd", [alpha, rel_v])
        out = AV + AR

        return alpha, out

    def forward(self, key, value, query, mask=None,
                enc_attn_cache=None, self_attn_cache=None,
                rel_attn_kv=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 1) Project key, value, and query.
        if enc_attn_cache is not None:
            key_up, value_up = enc_attn_cache
        else:
            key_up = self._split_heads(self.linear_keys(key)) # [batch_size, num_head, seq_len, dim_head]
            value_up = self._split_heads(self.linear_values(value))

        if self_attn_cache is not None:
            key_up_prev, value_up_prev = self_attn_cache
            # Append current key and value to the cache
            key_up = torch.cat([key_up_prev, key_up], dim=2)
            value_up = torch.cat([value_up_prev, value_up], dim=2)

        query_up = self._split_heads(self.linear_query(query))
        query_up = query_up / math.sqrt(dim_per_head)

        key_len = key_up.size(2)
        query_len = query_up.size(2)

        # 2) Calculate and scale scores.
        if mask is not None:
            mask = mask.unsqueeze(1).expand(
                batch_size, head_count, query_len, key_len)

        # do attention
        attn, context = \
            self._compute_relative_attention(
                q=query_up,
                k=key_up,
                v=value_up,
                mask=mask,
                rel_k=rel_attn_kv[0],
                rel_v=rel_attn_kv[1],
                dropout=self.dropout)

        # 3) Apply attention dropout and compute context vectors.
        # context ([batch, length, d_model])
        context = self._combine_heads(context)

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len).mean(1) \
            .contiguous()
        # END CHECK
        return output, top_attn, [key_up, value_up]

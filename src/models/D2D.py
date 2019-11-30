import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(threshold=10000000)

import src.context_cache as ctx
from src.models.base import NMTModel
from src.models import transformer
from src.models.mem_transformer import MemTransformerLM
from src.modules.transformer_xl_utils.parameter_init import weights_init
from src.modules.embeddings import Embeddings
from src.modules.position_embedding import PositionalEmbedding, SegmentEmbedding
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.utils import nest
from src.modules.sublayers import MultiHeadedAttention, PositionwiseFeedForward
from src.data.vocabulary import PAD, EOS, BOS


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1, **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input
        return self.pos_ffn(out)


class Encoder(transformer.Encoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = Embeddings(num_embeddings=kwargs["n_src_vocab"],
                                     embedding_dim=kwargs["d_word_vec"],
                                     dropout=False,
                                     add_position_embedding=False)
        self.reset_position = kwargs.get("reset_encoder_position", False)
        self.pos_emb = PositionalEmbedding(kwargs["d_word_vec"], dropout=kwargs["dropout"])

        self.segment_embed = SegmentEmbedding(
            kwargs["d_word_vec"],
            max_segment=kwargs["max_encoder_segment_embedding"]) \
            if kwargs["max_encoder_segment_embedding"] > 0 \
            else None

        self.mix_local_global = kwargs.get("mix_encoder_local_global_attention", False)
        if self.mix_local_global:
            self.global_encoder_layer = EncoderBlock(**kwargs)

        self.reset_params()

    def reset_params(self):
        scale = self.embeddings.scale
        nn.init.normal_(self.embeddings.embeddings.weight,
                        mean=0, std=1/scale)
        nn.init.constant_(self.embeddings.embeddings.weight[self.embeddings.padding_idx], 0)
        if self.segment_embed is not None:
            nn.init.normal_(self.segment_embed.embed.weight,
                            mean=0, std=1/scale)

    def build_local_mask(self, segment_ids, enc_mask):
        # segment_ids: [bsz, len]. [[0,0,0,1,1,1,2,2,2,2...], ...]
        # enc_mask: [bsz, qlen, mlen] (qlen==mlen)
        # return: [bsz, qlen, mlen] (qlen==mlen). [[[0,0,0,1,1,1,1,1,1,1...], [1,1,1,0,0,0,1,1,1,1...],...]]
        local_mask = []
        for i in range(segment_ids.size(-1)):
            # [bsz, mlen]
            local_mask.append(
                segment_ids.ne(
                    segment_ids[:, i:i+1].expand_as(segment_ids)
                )
            )
        local_mask = torch.stack(local_mask, 1)
        return (local_mask + enc_mask).gt(0)

    def forward(self, src_seq, position=None, segment_ids=None):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        emb = self.embeddings(src_seq)
        if segment_ids is not None and self.segment_embed is not None:
            seg_emb = self.segment_embed(segment_ids)
            emb += seg_emb
        emb = self.pos_emb(emb, pos_seq=position if self.reset_position else None)

        enc_mask = src_seq.detach().eq(PAD)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)
        if self.mix_local_global:
            local_self_attn_mask = self.build_local_mask(segment_ids, enc_slf_attn_mask)

        out = emb
        for i in range(self.num_layers):
            out = self.block_stack[i](
                out,
                enc_slf_attn_mask if not self.mix_local_global else local_self_attn_mask)

        if self.mix_local_global:
            out = self.global_encoder_layer(out, enc_slf_attn_mask)

        out = self.layer_norm(out)
        return out, enc_mask
    
    
class D2D(NMTModel):
    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None,
            dropout=0.1, proj_share_weight=True, **kwargs):

        super(D2D, self).__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head, **kwargs)

        tgt_len, mem_len, ext_len = 4, 4, 0

        self.decoder = MemTransformerLM(n_token=n_tgt_vocab, n_layer=n_layers, n_head=n_head,
                                        d_model=d_model, d_head=dim_per_head,
                                        d_embed=d_word_vec,
                                        dropout=dropout, dropatt=dropout,
                                        d_inner=d_inner_hid, attn_type=0,
                                        tgt_len=tgt_len, mem_len=mem_len, ext_len=ext_len,
                                        pre_lnorm=True)
        self.decoder.apply(weights_init)
        self.decoder.word_emb.apply(weights_init)

        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        if proj_share_weight:
            self.generator = transformer.Generator(n_words=n_tgt_vocab,
                                                   hidden_size=d_word_vec,
                                                   shared_weight=self.decoder.word_emb.emb_layers[0].weight,
                                                   padding_idx=PAD)

        else:
            self.generator = transformer.Generator(
                n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=PAD)

    def forward(self, src_seq, tgt_seq, log_probs=True):
        enc_output, enc_mask = self.encoder(src_seq)
        dec_inp = tgt_seq

        dec_inp_T = dec_inp.transpose(0, 1)
        enc_out_T = enc_output.transpose(0, 1)
        enc_mask_T = enc_mask.transpose(0, 1)

        dec_pred_T, ctx.memory_cache = self.decoder(dec_inp_T, enc_out_T, enc_mask_T, *ctx.memory_cache)
        dec_pred = dec_pred_T.transpose(0, 1).contiguous()

        return self.generator(dec_pred, log_probs=log_probs)

    def encode(self, src_seq):

        ctx, ctx_mask = self.encoder(src_seq)
        return {"ctx": ctx, "ctx_mask": ctx_mask}

    def decode_train(self, tgt_seq, enc_out, enc_mask, log_probs=True):
        dec_inp = tgt_seq

        dec_inp_T = dec_inp.transpose(0, 1)
        enc_out_T = enc_out.transpose(0, 1)
        enc_mask_T = enc_mask.transpose(0, 1)
        
        if ctx.ENABLE_CONTEXT:
            dec_pred_T, ctx.memory_cache = self.decoder(dec_inp_T, enc_out_T, enc_mask_T, *ctx.memory_cache)
        else:
            dec_pred_T, _ = self.decoder(dec_inp_T, enc_out_T, enc_mask_T)
        dec_pred = dec_pred_T.transpose(0, 1).contiguous()

        return self.generator(dec_pred, log_probs=log_probs)

    def decode(self, tgt_seq, dec_states, log_probs=True):

        enc_output = dec_states["ctx"]
        enc_output_mask = dec_states['ctx_mask']
        enc_attn_caches = dec_states['enc_attn_caches']
        slf_attn_caches = dec_states['slf_attn_caches']

        dec_inp = tgt_seq
        dec_inp_T = dec_inp.transpose(0, 1)
        enc_out_T = enc_output.transpose(0, 1)
        enc_mask_T = enc_output_mask.transpose(0, 1)

        # dec_output, slf_attn_caches, enc_attn_caches = self.decoder(tgt_seq, )
        dec_pred_T, ctx.memory_cache = self.decoder(dec_inp_T, enc_out_T, enc_mask_T, *ctx.memory_cache)
        dec_pred = dec_pred_T.transpose(0, 1).contiguous()

        next_scores = self.generator(dec_pred[:, -1], log_probs=log_probs)

        dec_states['enc_attn_caches'] = enc_attn_caches
        dec_states['slf_attn_caches'] = slf_attn_caches

        return next_scores, dec_states

    def init_decoder(self, enc_outputs, expand_size=1):

        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        if expand_size > 1:
            ctx = tile_batch(ctx, multiplier=expand_size)
            ctx_mask = tile_batch(ctx_mask, multiplier=expand_size)

        return {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "enc_attn_caches": None,
            "slf_attn_caches": None
        }

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):

        slf_attn_caches = dec_states['slf_attn_caches']

        batch_size = slf_attn_caches[0][0].size(0) // beam_size

        n_head = self.decoder.n_head
        dim_per_head = self.decoder.d_head

        slf_attn_caches = nest.map_structure(
            lambda t: tensor_gather_helper(gather_indices=new_beam_indices,
                                           gather_from=t,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, n_head, -1, dim_per_head]),
            slf_attn_caches)

        dec_states['slf_attn_caches'] = slf_attn_caches

        return dec_states

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
from src.modules.position_embedding import PositionalEmbedding, SegmentEmbedding, RelativePositionEmbeddings, RelativeSegmentEmbeddings
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.utils import nest
from src.modules.sublayers import MultiHeadedAttention, PositionwiseFeedForward
from src.modules.relative_attention import MultiHeadedAttentionRelative
from src.data.vocabulary import PAD, EOS, BOS


class GatedConnection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Linear(d_model*2, d_model, True)
    
    def forward(self, t1, t2):
        g = F.sigmoid(self.w(torch.cat([t1, t2], -1)))
        return g*t1 + (1-g)*t2


class EncoderBlock(nn.Module):
    attention_cls = {
        "normal": MultiHeadedAttention,
        "relative": MultiHeadedAttentionRelative
    }
    
    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1, attention_type="normal", **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

        attn_cls = EncoderBlock.attention_cls[attention_type]
        self.slf_attn = attn_cls(head_count=n_head, model_dim=d_model, dropout=dropout,
                                 dim_per_head=dim_per_head)
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None, rel_attn_kv=(None, None)):
        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask, rel_attn_kv=rel_attn_kv)
        out = self.dropout(context) + enc_input
        return self.pos_ffn(out)


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self._build_embeddings(**kwargs)

        self._build_encoder_layers(**kwargs)

        self._build_global_encoder_layer(**kwargs)
        
        self.layer_norm = nn.LayerNorm(kwargs["d_model"])

        self._reset_parameters()

    def _reset_parameters(self):
        scale = self.embeddings.scale
        nn.init.normal_(self.embeddings.embeddings.weight,
                        mean=0, std=1/scale)
        nn.init.constant_(self.embeddings.embeddings.weight[self.embeddings.padding_idx], 0)

    # -------------------------- builds -------------------------------- #
    def _build_embeddings(self, **kwargs):
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
    
    def _build_encoder_layers(self, **kwargs):
        attn_type = kwargs.get("encoder_attention_type", "normal")
        self.attn_type = attn_type

        if attn_type == "relative":
            self.block_stack = nn.ModuleList(
                [EncoderBlock(attention_type="relative", **kwargs) for _ in range(kwargs["n_layers"])])
            self.rel_pos_key = RelativePositionEmbeddings(128, kwargs["d_model"]//kwargs["n_head"])
            self.rel_pos_values = RelativePositionEmbeddings(128, kwargs["d_model"]//kwargs["n_head"])

        elif attn_type == "normal":
            self.block_stack = nn.ModuleList(
                [EncoderBlock(**kwargs) for _ in range(kwargs["n_layers"])]) 
            self.rel_pos_key, self.rel_pos_values = None, None
            
    def _build_global_encoder_layer(self, **kwargs):
        glb_attn_type = kwargs.get("global_encoder_attention_type", "none")
        self.glb_attn_type = glb_attn_type
        if glb_attn_type == "none":
            self.global_encoder_layer = None

        elif glb_attn_type == "normal":
            self.global_encoder_layer = EncoderBlock(**kwargs)

        elif glb_attn_type == "word-relative":
            raise NotImplementedError

        elif glb_attn_type == "segment-relative":
            self.global_encoder_layer = EncoderBlock(
                attention_type="relative", **kwargs)
            self.global_rel_seg_k = RelativeSegmentEmbeddings(
                kwargs["max_encoder_segment_embedding"], kwargs["d_model"]//kwargs["n_head"])
            self.global_rel_seg_v = RelativeSegmentEmbeddings(
                kwargs["max_encoder_segment_embedding"], kwargs["d_model"]//kwargs["n_head"])
        
        if kwargs.get("global_encoder_gate", False):
            self.glb_gate = GatedConnection(kwargs["d_model"])

    # -------------------------- prepares -------------------------------- #
    def _prepare_local_mask(self, segment_ids, enc_mask):
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

    def _prepare_masks(self, enc_mask, segment_ids):
        # enc_mask: padding mask [bsz, src_len]
        global_mask = enc_mask[:, None, :].expand(*enc_mask.size(), enc_mask.size(1))
        
        if self.glb_attn_type != "none": # which means the encoder layers using local encoder
            local_mask = self._prepare_local_mask(segment_ids, global_mask)
            encoder_self_attention_mask = local_mask
            return encoder_self_attention_mask, global_mask

        return global_mask, global_mask

    def _prepare_relative(self, length):
        # relative attention related
        return self.rel_pos_key(length), self.rel_pos_values(length)

    def _prepare_segment_relative(self, segment_ids):
        # segment-level relative attention related
        return self.global_rel_seg_k(segment_ids), self.global_rel_seg_v(segment_ids)

    # -------------------------- forwards -------------------------------- #
    def forward_embedding(self, src_seq, position=None, segment_ids=None):
        emb = self.embeddings(src_seq)
        if self.attn_type == "normal":
            # segment embedding
            if segment_ids is not None and self.segment_embed is not None:
                seg_emb = self.segment_embed(segment_ids)
                emb = emb + seg_emb
            # absolute position embeddings. 
            # reset for each segment or not
            if self.reset_position and position is not None:
                emb = self.pos_emb(emb, pos_seq=position) 
            else:
                emb = self.pos_emb(emb)
            return emb

        elif self.attn_type == "relative":
            # do not use absolute pe when using relative attention
            return emb
        
    def forward_encoder_layers(self, emb, encoder_layer_mask):
        out = emb

        if self.attn_type == "normal":
            for layer in self.block_stack:
                out = layer(
                    out,
                    encoder_layer_mask)

        elif self.attn_type == "relative":
            # relative attention related
            src_len = emb.size(1)
            rel_k, rel_v = self._prepare_relative(src_len)
            for layer in self.block_stack:
                out = layer(
                    out,
                    encoder_layer_mask,
                    rel_attn_kv=[rel_k, rel_v])
        else:
            raise NotImplementedError
        return out

    def forward_global_encoder_layer(self, out, global_encoder_mask, segment_ids):
        if self.glb_attn_type == "none":
            return out

        elif self.glb_attn_type == "normal":
            glb_out = self.global_encoder_layer(out, global_encoder_mask)

        elif self.glb_attn_type == "word-relative":
            raise NotImplementedError

        elif self.glb_attn_type == "segment-relative":
            seg_rel_k, seg_rel_v = self._prepare_segment_relative(segment_ids)
            glb_out = self.global_encoder_layer(out, global_encoder_mask, rel_attn_kv=[seg_rel_k, seg_rel_v])

        if self.glb_gate is not None:
            return self.glb_gate(out, glb_out)
        else:
            return glb_out

    def forward(self, src_seq, position=None, segment_ids=None):
        # Word embedding look up
        enc_mask = src_seq.detach().eq(PAD)

        # embeddings
        emb = self.forward_embedding(
            src_seq=src_seq,
            position=position,
            segment_ids=segment_ids)

        # masks
        encoder_self_attention_mask, global_attention_mask = \
            self._prepare_masks(
                enc_mask=enc_mask,
                segment_ids=segment_ids)

        # main encoder layers
        out = self.forward_encoder_layers(
            emb=emb,
            encoder_layer_mask=encoder_self_attention_mask)

        # global encoder layer, if applicable
        out = self.forward_global_encoder_layer(
            out=out,
            global_encoder_mask=global_attention_mask,
            segment_ids=segment_ids)

        return self.layer_norm(out), enc_mask


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

    def finish_decoder(self):
        for layer in self.decoder.layers:
            layer.ctx_attn.attn_cache = None

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

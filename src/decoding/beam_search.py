# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

from src.data.vocabulary import BOS, EOS, PAD
from src.models.base import NMTModel
from .utils import mask_scores, tensor_gather_helper
import src.context_cache as ctx


def beam_search(nmt_model, beam_size, max_steps, dec_state, alpha=-1.0):
    """

    Args:
        nmt_model (NMTModel):
        beam_size (int):
        max_steps (int):
        src_seqs (torch.Tensor):

    Returns:

    """

    batch_size = dec_state["ctx"].size(0)

    init_dec_states = nmt_model.init_decoder(dec_state, expand_size=beam_size)

    # Prepare for beam searching
    beam_mask = dec_state["ctx"].new(batch_size, beam_size).fill_(1).float()
    final_lengths = dec_state["ctx"].new(batch_size, beam_size).zero_().float()
    beam_scores = dec_state["ctx"].new(batch_size, beam_size).zero_().float()
    final_word_indices = dec_state["ctx"].new(batch_size, beam_size, 1).fill_(BOS).long()

    dec_states = init_dec_states
    
    src_empty_mask = (1 - dec_states["ctx_mask"].float()).sum(-1).eq(0).view(batch_size, beam_size) # [batch, beam] 1 means empty

    if ctx.memory_cache is None:
        ctx.memory_cache = tuple()

    current_memory_cache = ctx.memory_cache

    for t in range(max_steps):
        ctx.memory_cache = current_memory_cache

        next_scores, dec_states = nmt_model.decode(final_word_indices.view(batch_size * beam_size, -1), dec_states)

        next_scores = - next_scores  # convert to negative log_probs
        next_scores = next_scores.view(batch_size, beam_size, -1)
        next_scores = mask_scores(scores=next_scores, beam_mask=beam_mask)

        beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

        vocab_size = beam_scores.size(-1)

        if t == 0 and beam_size > 1:
            # Force to select first beam at step 0
            beam_scores[:, 1:, :] = float('inf')

        # Length penalty
        if alpha > 0.0:
            normed_scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + beam_mask + final_lengths).unsqueeze(2) ** alpha
        else:
            normed_scores = beam_scores.detach().clone()

        normed_scores = normed_scores.view(batch_size, -1)

        # Get topK with beams
        # indices: [batch_size, ]
        _, indices = torch.topk(normed_scores, k=beam_size, dim=-1, largest=False, sorted=False)
        next_beam_ids = torch.div(indices, vocab_size)  # [batch_size, ]
        next_word_ids = indices % vocab_size  # [batch_size, ]

        # Re-arrange by new beam indices
        beam_scores = beam_scores.view(batch_size, -1)
        beam_scores = torch.gather(beam_scores, 1, indices)

        beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                         gather_from=beam_mask,
                                         batch_size=batch_size,
                                         beam_size=beam_size,
                                         gather_shape=[-1])

        final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                  gather_from=final_word_indices,
                                                  batch_size=batch_size,
                                                  beam_size=beam_size,
                                                  gather_shape=[batch_size * beam_size, -1])

        final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=final_lengths,
                                             batch_size=batch_size,
                                             beam_size=beam_size,
                                             gather_shape=[-1])
        ### 20191103 attn_cache的优化之后再说！
        # dec_states = nmt_model.reorder_dec_states(dec_states, new_beam_indices=next_beam_ids, beam_size=beam_size)

        # If next_word_ids is EOS, beam_mask_ should be 0.0
        beam_mask_ = 1.0 - next_word_ids.eq(EOS).float()
        next_word_ids.masked_fill_(((beam_mask_ + beam_mask).eq(0.0) + src_empty_mask).gt(0),
                                   PAD)  # If last step a EOS is already generated, we replace the last token as PAD
        beam_mask = beam_mask * beam_mask_

        # # If an EOS or PAD is encountered, set the beam mask to 0.0
        final_lengths += beam_mask

        final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

        if beam_mask.eq(0.0).all():
            break

    # Length penalty
    if alpha > 0.0:
        scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + final_lengths) ** alpha
    else:
        scores = beam_scores / final_lengths

    _, reranked_ids = torch.sort(scores, dim=-1, descending=False)
    

    final_word_indices = tensor_gather_helper(gather_indices=reranked_ids,
                                gather_from=final_word_indices[:, :, 1:].contiguous(),
                                batch_size=batch_size,
                                beam_size=beam_size,
                                gather_shape=[batch_size * beam_size, -1])

    # @zzx (2019-11-20): leave only best beam
    if ctx.memory_cache:
        # [bsz, beam, len, d] 
        length = final_word_indices.size(-1)
        new_mems = [rerank_tensor(cache.transpose(0, 1).view(batch_size, beam_size, length, -1), reranked_ids) for cache in ctx.memory_cache]
        # [len, bsz*beam, d]
        new_mems_best = [leave_best_beam_and_repeat(mem).view(batch_size*beam_size, length, -1).transpose(0, 1) for mem in new_mems] 
        # [len, bsz*beam, d]
        ctx.memory_cache = new_mems_best 

        # [bsz, beam, len]
        best_word_indices = leave_best_beam_and_repeat(final_word_indices)
        # [len, bsz*beam]
        ctx.memory_mask = best_word_indices.eq(PAD).view(batch_size*beam_size, -1).transpose(0, 1)

    nmt_model.finish_decoder()
    
    return final_word_indices


def rerank_tensor(tensor, reranked_ids):
    # tensor [batch, beam, ...]
    # reranked_ids [batch, beam]
    sizes = tensor.size()
    bsz, beam_size = reranked_ids.size()
    _tensor = tensor.reshape(bsz, beam_size, -1)
    reranked_tensor = torch.gather(
        _tensor, 1, reranked_ids[:, :, None].expand_as(_tensor))
    # return [batch, beam, ...] as tensor
    return reranked_tensor.reshape(*sizes)


def leave_best_beam_and_repeat(tensor):
    # tensor [batch, beam, ...]
    sizes = tensor.size()
    bsz, beam_size = sizes[0], sizes[1]
    _tensor = tensor.reshape(bsz, beam_size, -1)
    # return [batch, beam, ...] as tensor
    return _tensor[:, 0, :].repeat(1, beam_size, 1).reshape(*sizes)


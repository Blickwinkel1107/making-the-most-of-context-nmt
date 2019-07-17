### GLOBAL VARIABLES
### created by yx
import pickle
import torch
from src.utils.common_utils import *
from src.data.vocabulary import Vocabulary


BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
PAD = Vocabulary.PAD

ENABLE_CONTEXT = True

vocab_src = {}
vocab_tgt = {}

GLOBAL_INDEX = 0
sent2idx = dict()
idx2sent = dict()
CONTEXT_SIZE = 3


def get_context(srcSents):
    '''
    :param srcSent: src input tensor
    :param tgtSent: tgt input tensor
    :return: context sentences list

    the function will return the context sentences of input sentences

    example: (assume you are training zh-en task)
        srcSent = [[zh-sent4], [zh-sent5]],
        CONTEXT_SIZE = 3
        when the sentences order in source document is distributed like below:
        [
            ...
            [zh-sent0],
            [zh-sent1],
            [zh-sent2],
            [zh-sent3],
            [zh-sent4],
            [zh-sent5],
            ...
        ]
        then the function return:
        [
            [zh-sent1, zh-sent2, zh-sent3],
            [zh-sent2, zh-sent3, zh-sent4]
        ]
    '''
    srcSents = srcSents.data.numpy().tolist()
    contextSrcSents = []
    maxCtxSentLen = 0
    for srcSent in srcSents:
        # print(originSentPair)    ##PAD = 0   EOS = 1   BOS = 2   UNK = 3
        srcSent = [ x for x in srcSent if x not in [PAD, EOS, BOS] ]
        sent = tuple(srcSent)  ##tuplize in order to match "key" format in "sent2idx" dict
        # binSentPair = str(sentPair)

        binSent = pickle.dumps(sent)
        idx = sent2idx[binSent].pop()


        if sent2idx[binSent].__len__() == 0:
            sent2idx.pop(binSent)   ###every key will only be accessed 1 time, so delete after accessed
        contextSrcSentsForCurrentSent = []
        for i in range( max(0, idx-CONTEXT_SIZE), idx ):    ##access {idx-CONTEXT_SIZE, ..., idx-2, idx-1} or {0, ..., idx-2, idx-1}
            binContextSrcSent= idx2sent[i][1]     ##[0]:ttl, [1]:binSentPair
            contextSrcSent = pickle.loads(binContextSrcSent)
            contextSrcSentsForCurrentSent += contextSrcSent   ###only extend src context!
            idx2sent[i][0] -= 1
            if idx2sent[i][0] == 0: # ttl == 0
                idx2sent.pop(i)
        # if contextSrcSentsForCurrentSent.__len__() != 0:
        #     contextSrcSents.append(contextSrcSentsForCurrentSent)
        contextSrcSents.append(contextSrcSentsForCurrentSent)
        maxCtxSentLen = max(maxCtxSentLen, contextSrcSentsForCurrentSent.__len__())
    ##BOS EOS PAD
    for i in range(contextSrcSents.__len__()):
        contextSrcSents[i] = [BOS] + contextSrcSents[i] + [EOS] + [PAD]*(maxCtxSentLen-contextSrcSents[i].__len__())
    contextSrcSents = torch.tensor(contextSrcSents)
    if GlobalNames.USE_GPU is True:
        contextSrcSents = contextSrcSents.cuda()
    return contextSrcSents

# global global_index, sent2idx, idx2sent




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


def get_context(srcSent, tgtSent):
    '''
    :param srcSent: src input tensor
    :param tgtSent: tgt input tensor
    :return: context sentences list

    the function will return the context sentences of input sentences

    example: (assume you are training zh-en task)
        srcSent = [[zh-sent4], [zh-sent5]],
        tgtSent = [[en-sent4], [en-sent5]],
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
    srcSent = srcSent.data.numpy().tolist()
    tgtSent = tgtSent.data.numpy().tolist()
    contextSrcSents = []
    maxCtxSentLen = 0
    for originSentPair in zip(srcSent, tgtSent):
        # print(originSentPair)    ##PAD = 0   EOS = 1   BOS = 2   UNK = 3
        sentPair = []
        for originSent in originSentPair:
            sent = [ x for x in originSent if x not in (PAD, BOS, EOS) ]
            sentPair.append(sent)
        sentPair = tuple(sentPair)  ##tuplize in order to match "key" format in "sent2idx" dict
        # binSentPair = str(sentPair)

        for k,v in sent2idx.items():
            src = pickle.loads(k)
            src = vocab_src.ids2sent(src[0])
            print(src)

        binSentPair = pickle.dumps(sentPair)
        idx = sent2idx[binSentPair].pop()

        if sent2idx[binSentPair].__len__() == 0:
            sent2idx.pop(binSentPair)   ###every key will only be accessed 1 time, so delete after accessed
        contextSrcSentsForCurrentSent = []
        for i in range( max(0, idx-CONTEXT_SIZE), idx ):    ##access {idx-CONTEXT_SIZE, ..., idx-2, idx-1} or {0, ..., idx-2, idx-1}
            binContextSrcSentPair = idx2sent[i][1]     ##[0]:ttl, [1]:binSentPair
            contextSrcSentPair = pickle.loads(binContextSrcSentPair)
            contextSrcSentsForCurrentSent += contextSrcSentPair[0]   ###only extend src context!
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




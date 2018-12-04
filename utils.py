import torch

def prep_seq(seq, mapping):
    return torch.tensor([mapping[w] for w in seq], dtype=torch.long)

def pairs2vocab(p):
    vocab = set()
    for s1, s2 in p:
        vocab = vocab.union(set(s1.split()+s2.split()))
    return sorted(vocab)

PAD_TOKEN = 0 # padding
SOS_TOKEN = 1 # start of sentence
EOS_TOKEN = 2 # end of sentence

class Vocab:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {
            PAD_TOKEN: "<PAD>",
            SOS_TOKEN: "<SOS>",
            EOS_TOKEN: "<EOS>", 
        }
        self.size = 3

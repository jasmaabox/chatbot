import torch
import torch.nn as nn
import json
import numpy as np
import re
import random

def clean_word(s):
    """ Strips words to alphanumeric """
    return re.sub(r'[^a-zA-Z0-9_ ]+', '', s.lower())

def read_pairs(fname, v, speaker=None):
    """ Reads in conversational pairs from FB message dump """
    pairs = []
    with open(fname, 'r') as f:
        data = json.loads(f.read())

        # set speaker if none
        if speaker == None:
            speaker = data['participants'][0]['name']

        messages = data['messages']
        messages.reverse()
        for i in range(len(messages)):
            if i < len(messages)-1 and messages[i]['sender_name'] == speaker and messages[i+1]['sender_name'] != speaker:
                m1 = clean_word(messages[i]['content'])
                m2 = clean_word(messages[i+1]['content'])
                v.add_sentence(m1)
                v.add_sentence(m2)
                pairs.append( (m1, m2) )
    return pairs

def read_embeds(fname, vocab, embed_dim_size):
    """ Read glove embeddings and generate embedding layer """
    # get embed dict
    embed_map = {}
    with open(fname, 'r', encoding='utf8') as f:
        embeds = f.read().split('\n')

        for l in embeds:
            if len(l) > 0:
                data = l.split()
                embed_map[data[0]] = torch.FloatTensor(list(map(float, data[1:])))

    # convert to embedding layer
    weights_matrix = np.zeros( (vocab.size, embed_dim_size) )

    for word in vocab.word2idx:
        idx = vocab.word2idx[word]
        try:
            weights_matrix[idx] = embed_map[word]
        except KeyError:
            # generate random embedding if not in vocab
            weights_matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim_size,))

    weights_matrix = torch.Tensor(weights_matrix)

    # generate embedding
    embedding = nn.Embedding(vocab.size, embed_dim_size)
    embedding.load_state_dict({'weight':weights_matrix})
    embedding.weight.requires_grad = False
    return embedding



PAD_TOKEN = 0 # padding
SOS_TOKEN = 1 # start of sentence
EOS_TOKEN = 2 # end of sentence

class Vocab:
    """ Maps words to indices """

    def __init__(self):
        self.trimmed = False
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {
            PAD_TOKEN: "<PAD>",
            SOS_TOKEN: "<SOS>",
            EOS_TOKEN: "<EOS>",
        }
        self.size = 3

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.size
            self.word2count[word] = 1
            self.idx2word[self.size] = word

            self.size += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        """ Culls words below a count threshold """
        if self.trimmed:
            return

        keep_idxs = []
        for w in self.word2count:
            if self.word2count[w] > min_count:
                keep_idxs.append(self.word2idx[w])

        old_word2count = self.word2count
        old_idx2word = self.idx2word

        # re-init and add
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {
            PAD_TOKEN: "<PAD>",
            SOS_TOKEN: "<SOS>",
            EOS_TOKEN: "<EOS>",
        }
        self.size = 3

        for i in keep_idxs:
            self.add_word(old_idx2word[i])
            self.word2count[old_idx2word[i]] = old_word2count[old_idx2word[i]]

def sentence2idxs(seq, vocab, max_len):
    """ Converts sentence to index tensor with zero padding """
    res = []
    for w in seq.split():
        try:
            res.append(vocab.word2idx[w])
        except KeyError:
            # generate random for now
            res.append(random.randint(3, vocab.size))

    res += [EOS_TOKEN]
    res = res + [PAD_TOKEN] * (max_len-len(res))

    return torch.tensor(res, dtype=torch.long)

def pairs2batch(pairs, vocab):
    """ Converts pairs of sentences into tensor batch """
    in_mat, out_mat = [], []
    len_vec = []
    bin_mat = []

    # gets max len
    max_len = -1
    for s_in, s_out in pairs:
        max_len = max( len(s_in.split()), len(s_out.split()), max_len )
    max_len += 1

    for s_in, s_out in pairs:

        # process input
        in_vec = sentence2idxs(s_in, vocab, max_len)

        # process output
        out_vec = sentence2idxs(s_out, vocab, max_len)
        bin_vec = torch.tensor(list(map(lambda x: 1 if x != PAD_TOKEN else 0, out_vec)), dtype=torch.uint8)

        bin_mat.append(bin_vec.numpy())
        in_mat.append(in_vec.numpy())
        out_mat.append(out_vec.numpy())
        len_vec.append((in_vec == EOS_TOKEN).nonzero().item() + 1)

    in_mat = np.transpose(torch.LongTensor(in_mat))
    len_vec = torch.IntTensor(len_vec)
    out_mat = np.transpose(torch.LongTensor(out_mat))
    bin_mat = np.transpose(torch.ByteTensor(bin_mat))

    return in_mat, len_vec, out_mat, bin_mat, max_len

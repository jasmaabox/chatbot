import torch
import json
import numpy as np

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
                v.add_sentence(messages[i]['content'])
                v.add_sentence(messages[i+1]['content'])
                pairs.append( (messages[i]['content'], messages[i+1]['content']) )
    return pairs

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
        else:
            self.word2count[word] += 1
            
        self.size += 1

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
    res = [vocab.word2idx[w] for w in seq.split(" ")] + [EOS_TOKEN]
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
        try:
            # process input
            in_vec = sentence2idxs(s_in, vocab, max_len)

            # process output
            out_vec = sentence2idxs(s_out, vocab, max_len)
            bin_vec = torch.tensor(list(map(lambda x: 1 if x != PAD_TOKEN else 0, out_vec)), dtype=torch.uint8)

            bin_mat.append(bin_vec.numpy())
            in_mat.append(in_vec.numpy())
            out_mat.append(out_vec.numpy())
            len_vec.append((in_vec == EOS_TOKEN).nonzero().item() + 1)
            
        except KeyError:
            # pass on sentence with words not in vocab
            continue

    in_mat = np.transpose(torch.LongTensor(in_mat))
    len_vec = torch.IntTensor(len_vec)
    out_mat = np.transpose(torch.LongTensor(out_mat))
    bin_mat = np.transpose(torch.ByteTensor(bin_mat))

    return in_mat, len_vec, out_mat, bin_mat


vocab = Vocab()
pairs = read_pairs('data/message.json', vocab)
in_mat, len_vec, out_mat, bin_mat = pairs2batch(pairs, vocab)

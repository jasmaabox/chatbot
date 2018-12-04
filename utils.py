import torch
import json

def prep_seq(seq, vocab):
    return torch.tensor([vocab.word2idx[w] for w in seq], dtype=torch.long)

def read_pairs(fname, v, speaker=None):
    """Reads in conversational pairs from FB message dump"""
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
    """Maps words to indices"""
    
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
        """Culls words below a count threshold"""
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


v = Vocab()
pairs = read_pairs('data/message.json', v)
        

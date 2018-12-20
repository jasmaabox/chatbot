import torch
import torch.nn as nn

import numpy as np

from models import *
from utils import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def evaluate(encoder, decoder, searcher, vocab, inp, max_length=100):
    # create input and length vecs
    idx_batch = [sentence2idxs(inp, vocab, len(inp.split())+1).numpy()]
    lengths = torch.IntTensor( [len(inp.split())+1] )
    idx_batch = torch.LongTensor(idx_batch).transpose(0, 1)

    input_batch = idx_batch.to(device)
    lengths = lengths.to(device)

    tokens, scores = searcher(input_batch, lengths, max_length)

    decoded_words = [vocab.idx2word[token.item()] for token in tokens]
    return decoded_words


# === MAIN ===

# Load checkpoint
checkpoint = torch.load('checkpoint/4000_checkpoint.tar')

# load vocab
vocab = Vocab()
pairs = read_pairs('data/message_ex.json', vocab)
vocab.trim(3)

# read in embedding
embedding = nn.Embedding(vocab.size, 500)
embedding.load_state_dict(checkpoint['embedding'])
embedding.eval()

# load models
encoder = EncoderRNN(500, embedding, 2, 0.1)
encoder.load_state_dict(checkpoint['en'])
encoder.eval()
decoder = LuongAttnDecoderRNN(DOT_METHOD, embedding, 500, vocab.size, 2, 0.1)
decoder.load_state_dict(checkpoint['de'])
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)

while True:
    sentence = input('> ')
    words = evaluate(encoder, decoder, searcher, vocab, sentence)
    resp = ""
    for w in words:
        if w == '<EOS>':
            break
        resp += w + " "
    resp = resp.strip()

    print('BOT: '+resp)

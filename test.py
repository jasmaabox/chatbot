import torch
import torch.nn as nn

import numpy as np

from models import *
from utils import *

def evaluate(encoder, decoder, searcher, vocab, inp, max_length=100):
    """ Runs input through decoder as batch of size 1 """
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

# set config
HIDDEN_SIZE = 500
n_layers = 2
dropout = 0.1
max_length = 100

try:
    with open('config.json', 'r') as f:
        config = json.loads(f.read())

        convo_path = config['convo_path']
        speaker = config['speaker']
        HIDDEN_SIZE = config['hidden_size']
        n_layers = config['n_layers']
        dropout = config['dropout']
        max_length = config['max_length']
        print("Read config from file.")
except FileNotFoundError:
    print("No config file. Using default config.")

# Load checkpoint
checkpoint = torch.load('checkpoint/me/4000_checkpoint.tar')

# load vocab
vocab = Vocab()
pairs = read_pairs(convo_path, vocab, speaker=speaker if speaker else None)
vocab.trim(3)

# read in embedding
embedding = nn.Embedding(vocab.size, HIDDEN_SIZE)
embedding.load_state_dict(checkpoint['embedding'])
embedding.eval()

# load models
encoder = EncoderRNN(500, embedding, n_layers, dropout)
encoder.load_state_dict(checkpoint['en'])
encoder.eval()
decoder = LuongAttnDecoderRNN(DOT_METHOD, embedding, 500, vocab.size, n_layers, dropout)
decoder.load_state_dict(checkpoint['de'])
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)

# Run chatbot
while True:
    sentence = input('> ')
    words = evaluate(encoder, decoder, searcher, vocab, sentence, max_length=max_length)
    resp = ""
    for w in words:
        if w == '<EOS>':
            break
        resp += w + " "
    resp = resp.strip()

    print("BOT: ""+resp)

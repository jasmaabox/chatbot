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

    print('INPUT')
    print(idx_batch)

    input_batch = idx_batch.to(device)
    lengths = lengths.to(device)

    tokens, scores = searcher(input_batch, lengths, max_length)

    decoded_words = [vocab.idx2word[token.item()] for token in tokens]
    return " ".join(decoded_words)


# === MAIN ===
# load vocab
vocab = Vocab()
pairs = read_pairs('data/message.json', vocab)
vocab.trim(3)

# read in embedding
embedding_model_path = 'data/embedding_model'
embedding = torch.load(embedding_model_path)
embedding.eval()

# load models
encoder = EncoderRNN(25, embedding, 4)
encoder.load_state_dict(torch.load('checkpoint/encoder-100'))
encoder.eval()
decoder = LuongAttnDecoderRNN(DOT_METHOD, embedding, 25, vocab.size)
decoder.load_state_dict(torch.load('checkpoint/decoder-100'))
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)

while True:
    sentence = input('> ')
    resp = evaluate(encoder, decoder, searcher, vocab, sentence)
    print('BOT: '+resp)

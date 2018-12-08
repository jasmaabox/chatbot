import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from models import *
from utils import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def maskNLLLoss(inp, target, mask):
    """ Masked loss calculated from decoder output, target vec, and target mask """
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, n_total.item()


# === MAIN ===
# extract data
vocab = Vocab()
pairs = read_pairs('data/message.json', vocab)
embeds = read_embeds('data/glove.twitter.27B/glove.twitter.27B.25d.txt')

# === TRAIN ===

def train(inputs, lengths, target, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    """ Trains model """

    # zero grad
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # device options
    inputs = inputs.to(device)
    lengths = lengths.to(device)
    target = target.to(device)
    mask = mask.to(device)

    # loss vars
    loss = 0
    print_losses = []
    n_totals = 0

    # forward thru encoder
    encoder_outputs, encoder_hidden = encoder(inputs, lengths)

    # init decoder
    decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # roll teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # feed in target as next input
            decoder_input = target[t].view(1, -1)

            mask_loss, n_total = maskNLLLoss(decoder_output, target[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # feed in prev output as next input
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            mask_loss, n_total = maskNLLLoss(decoder_output, target[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    # backprop
    loss.backward()
    # clip grads
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    # update params
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

# === MAIN ===

MAX_LENGTH = 10
teacher_forcing_ratio = 0.8
BATCH_SIZE = 200
n_iteration = 100
learning_rate = 0.01
decoder_lr_ratio = 5
clip = 50

# random sample for training batch
inputs, lengths, target, mask, max_target_len = pairs2batch([random.choice(pairs) for _ in range(BATCH_SIZE)], vocab)

encoder = EncoderRNN(32, embedding, 4)
decoder = LuongAttnDecoderRNN(attn, embedding, 32, vocab.size)

encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate * decoder_lr_ratio)

print("Initializing...")
start_iteration = 1
print_loss = 0

# load from file
#if out_fname:
#    start_iteration = checkpoint['iteration'] + 1

# training loop
print("Training...")
for iteration in range(start_iteration, n_iteration + 1):

    input_batch = inputs[iteration - 1]
    length_batch = lengths[iteration - 1]
    target_batch = target[iteration - 1]
    mask_batch = mask[iteration - 1]

    loss = train(input_batch, length_batch, target_batch, mask_batch, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, BATCH_SIZE, clip)
    print_loss += loss
    print_loss_avg = print_loss / iteration
    print(f"Iteration {iteration}\tAverage loss{print_loss_avg}")

    # save checkpoint
    # TODO
    torch.save(encoder.state_dict(), f"checkpoint/encoder-{iteration}")
    torch.save(decoder.state_dict(), f"checkpoint/decoder-{iteration}")

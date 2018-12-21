import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import os

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

def train(inputs, lengths, target, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip):
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

# set config
checkpoint_path = "checkpoint"
convo_path = "data/message.json"
speaker = None
teacher_forcing_ratio = 1.0
BATCH_SIZE = 64
n_iteration = 4000
learning_rate = 0.0001
decoder_lr_ratio = 5.0
clip = 50
HIDDEN_SIZE = 500
n_layers = 2
dropout = 0.1
save_every = 500
trim_threshold = 3

try:
    with open('config.json', 'r') as f:
        config = json.loads(f.read())

        checkpoint_path = config['checkpoint_path']
        convo_path = config['convo_path']
        speaker = config['speaker']
        teacher_forcing_ratio = config['teacher_forcing_ratio']
        BATCH_SIZE = config['batch_size']
        n_iteration = config['n_iteration']
        learning_rate = config['learning_rate']
        decoder_lr_ratio = config['decoder_lr_ratio']
        clip = config['clip']
        HIDDEN_SIZE = config['hidden_size']
        n_layers = config['n_layers']
        dropout = config['dropout']
        save_every = config['save_every']
        trim_threshold = config['trim_threshold']
        print("Read config from file.")
except FileNotFoundError:
    print("No config file. Using default config.")

# extract data
vocab = Vocab()
pairs = read_pairs(convo_path, vocab, speaker=speaker if speaker else None)
vocab.trim(trim_threshold)

embedding = nn.Embedding(vocab.size, HIDDEN_SIZE)

# Load current checkpoint
max_iter = 1
checkpoint_files = os.listdir(checkpoint_path)
for i in range(n_iteration):
    if f'{i + 1}_checkpoint.tar' in checkpoint_files:
        max_iter = i + 1
start_iteration = max_iter

encoder = EncoderRNN(HIDDEN_SIZE, embedding, n_layers, dropout)
decoder = LuongAttnDecoderRNN(DOT_METHOD, embedding, HIDDEN_SIZE, vocab.size, n_layers, dropout)

encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate * decoder_lr_ratio)

print_loss = 0

if max_iter > 1:
    print(f"Loading checkpoint from {start_iteration}")
    checkpoint = torch.load(os.path.join(checkpoint_path, f"{start_iteration}_checkpoint.tar"))

    encoder.load_state_dict(checkpoint['en'])
    encoder.eval()
    decoder.load_state_dict(checkpoint['de'])
    decoder.eval()
    encoder_optimizer.load_state_dict(checkpoint['en_opt'])
    decoder_optimizer.load_state_dict(checkpoint['de_opt'])
    print_loss = checkpoint['print_loss']

# print stats
print("====== STATS =====")
print(f"Vocab size:\t{vocab.size}")
print("==================")
print("Start training...")

# random sample training batches
training_batches = [pairs2batch(random.sample(pairs, BATCH_SIZE), vocab) for _ in range(n_iteration)]

for iteration in range(start_iteration, n_iteration + 1):

    training_batch = training_batches[iteration - 1]
    input_batch, lengths_batch, target_batch, mask_batch, max_target_len = training_batch

    # re-sort in descending order
    lengths_batch = lengths_batch.numpy()
    input_temp = torch.LongTensor(input_batch.size())
    target_temp = torch.LongTensor(target_batch.size())
    mask_temp = torch.ByteTensor(mask_batch.size())

    for i, idx in enumerate(lengths_batch.argsort()):
        input_temp[:, len(lengths_batch) - i - 1] = input_batch[:, idx.item()]
        target_temp[:, len(lengths_batch) - i - 1] = target_batch[:, idx.item()]
        mask_temp[:, len(lengths_batch) - i - 1] = mask_batch[:, idx.item()]

    lengths_batch[::-1].sort()
    lengths_batch = torch.from_numpy(lengths_batch)
    input_batch = input_temp
    target_batch = target_temp
    mask_batch = mask_temp

    loss = train(input_batch, lengths_batch, target_batch, mask_batch, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, BATCH_SIZE, clip)
    print_loss += loss
    print_loss_avg = print_loss / iteration
    print(f"Iteration {iteration}\tAverage loss {print_loss_avg}")

    # save checkpoint
    if iteration % save_every == 0:
        torch.save({
            'iteration': iteration,
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': loss,
            'print_loss':print_loss,
            'embedding': embedding.state_dict()
        }, os.path.join(checkpoint_path, f'{iteration}_checkpoint.tar'))

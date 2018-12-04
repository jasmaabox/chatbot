import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import *
from utils import *

# === MAIN ===
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# extract data
with open("test_convo.txt", "r") as f:
    lines = f.read().split("\n")

pairs = []
for i in range(0, len(lines)-2, 2):
    pairs.append( (lines[i].lower(), lines[i+1].lower()) )

# === TRAIN ===
model = Chatbot(VOCAB_SIZE, 32, 32)
loss_f = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(300):
    for sentence, target in pairs:
        # zero grad and clear hidden
        model.zero_grad()
        model.hidden = model.init_hidden()
        # prep data
        inputs = prep_seq(sentence.split(), word2idx)
        targets = prep_seq(target.split(), word2idx)
        # forward pass
        scores = model.forward(inputs)
        # loss
        loss = loss_f(scores, targets)
        loss.backward()
        optimizer.step()

        print(loss)
    print("===")

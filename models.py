import torch
import torch.nn as nn
import torch.nn.functional as F

class Chatbot(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Chatbot, self).__init__()

        # embeds words to vecs
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # lstm
        self.lstm = nn.LSTM(hidden_size, embedding_size) # embedding dim and hidden dim
        self.hidden = self.init_hidden()
        # linear
        self.linear = nn.Linear(embedding_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def init_hidden(self):
        return (torch.zeros(1, 1, hidden_size),
                torch.zeros(1, 1, hidden_size))

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        out = F.relu(self.linear(lstm_out))
        out = self.linear2(out)
        scores = F.log_softmax(out, dim=1)
        return scores

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    """ Encode sentence to thought vec """
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = embedding
        self.n_layers = n_layers

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout), bidirectional=True)

    def forward(self, inputs, inp_lens, hidden=None):
        embedded = self.embedding(inputs)                                       # embed
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, inp_lens)    # pack
        outputs, hidden = self.gru(packed, hidden)                              # forward pass thru GRU
        outputs = torch.nn.utils.rnn.pad_packed_sequence(outputs)               # unpack

        # sum bidir outputs?
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]

        return outputs, hidden

DOT_METHOD = 0
GENERAL_METHOD = 1
CONCAT_METHOD = 2

class Attn(nn.Module):
    """ Luong attention layer """
    def __init__(self, hidden_size, method=DOT_METHOD):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == GENERAL_METHOD:
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == CONCAT_METHOD:
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def score(self, hidden, encoder_outputs):
        if self.method == DOT_METHOD:
            return torch.sum(hidden * encoder_outputs, dim=2)
        elif self.method == GENERAL_METHOD:
            energy = self.attn(encoder_outputs)
            return torch.sum(hidden * energy, dim=2)
        elif self.method == CONCAT_METHOD:
            energy = self.attn(torch.cat(
                (hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs),
                2
            )).tanh()
            return torch.sum(hidden * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()                           # transpose
        return F.log_softmax(atten_energies, dim=1).unsqueeze(1)    # do softmax and return to original axis?
                    

class LuongAttnDecoderRNN(nn.Module):
    """ Luong decoder """

    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """ Forward pass run step-by-step """
        # embed
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # forward pass thru GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # get attention weights
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # get concat context and do luong?
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden

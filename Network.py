import torch
import torch.nn as nn

class LSTMEncoder(torch.nn.Module):
    def __init__(self, batch_size , input_seq_length, enc_hidden_size):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = enc_hidden_size
        self.seq_len = input_seq_length
        self.num_layers = 1
        
        self.lstm_layer = nn.LSTM(1, enc_hidden_size, num_layers = self.num_layers)
        
        self.reset(batch_size)
        
    def forward(self, x):
        x, self.h1 = self.lstm_layer(x, self.h1)
        return self.h1[0]
    
    def reset(self, batch_size):
        self.h1 = (torch.zeros([self.num_layers, batch_size, self.hidden_size]),
                   torch.zeros([self.num_layers, batch_size, self.hidden_size]))

        
class LSTMDecoder(torch.nn.Module):
    def __init__(self, batch_size, out_seq_length, dec_hidden_size, out_size = 1):
        super(LSTMDecoder, self).__init__()
        
        self.hidden_size = dec_hidden_size
        self.seq_len = out_seq_length
        self.num_layers = 1
        
        self.lstm_layer = nn.LSTM(dec_hidden_size, dec_hidden_size, num_layers = self.num_layers)
        self.fc1 = nn.Linear(dec_hidden_size * out_seq_length, out_size)
        
        self.reset(batch_size)
        
        
    def forward(self, x):
        x, self.h1 = self.lstm_layer(x, self.h1)
        x = self.fc1(x.transpose(1,0).flatten(start_dim = 1))
        
        return x
    
    def reset(self, batch_size):
        self.h1 = (torch.zeros([self.num_layers, batch_size, self.hidden_size]),
                   torch.zeros([self.num_layers, batch_size, self.hidden_size]))
        

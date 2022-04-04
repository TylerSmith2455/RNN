import torch
from turtle import forward
from torch import dropout, nn

# LSTM class
class LSTMmodel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, max_len, device):
        super(LSTMmodel, self).__init__()

        # Hyper Parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.size = input_size

        self.embedding = nn.Embedding(num_embeddings = input_size,embedding_dim = input_size)

        # Define the RNN
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout=.2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size,output_size)
        

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_hidden(self, seq_len):
        
        return (torch.zeros(self.num_layers, seq_len, self.hidden_size),
                torch.zeros(self.num_layers, seq_len, self.hidden_size))


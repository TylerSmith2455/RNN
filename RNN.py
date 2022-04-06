import torch
from turtle import forward
from torch import nn

# RNN class
class RNNmodel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, device):
        super(RNNmodel, self).__init__()

        # Hyper Parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Device either CPU or GPU
        self.device = device

        # Define the RNN
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # Create hidden layer
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Obtaining output and hidden state
        output, hidden_state = self.rnn(x, hidden_state)

        # Make the output look the way it needs to
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)

        return output, hidden_state
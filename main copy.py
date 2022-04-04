import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from RNN import RNNmodel

# Create one hots
def create_one_hot(sequence, vocab_size):
    encoding = np.zeros((1,len(sequence), vocab_size), dtype=np.float32) 
    for i in range(len(sequence)): 
        encoding[0 ,i, sequence[i]] = 1 
        
    return encoding

def main():

    # Read in txt data
    with open('tiny-shakespeare.txt') as f:
        lines = f.readlines()
    lines = lines[:100]
    # Create dictionaries for conversion
    characters = set(''.join(lines)) 
    intChar = dict(enumerate(characters))
    charInt = {character: index for index, character in intChar.items()}
    
    #intChar[len(intChar)] = '^'
    #charInt['^'] = len(charInt)

    # Create inputs and targets
    input_sequence = [] 
    target_sequence = [] 
    for i in range(len(lines)): 
        #Remove the last character from the input sequence 
        input_sequence.append(lines[i][:-1]) 
        #Remove the first element from target sequences 
        target_sequence.append(lines[i][1:])

    # Replace all characters with integers
    for i in range(len(lines)): 
        input_sequence[i] = [charInt[character] for character in input_sequence[i]] 
        target_sequence[i] = [charInt[character] for character in target_sequence[i]]

    vocab_size = len(charInt)
    
    # Create one hots and tensors
    #input_sequence = create_one_hot(input_sequence, vocab_size, maxlen-1, len(lines))
    #input_sequence = torch.from_numpy(input_sequence)
    #target_sequence = torch.Tensor(target_sequence)
    
    # Check if GPU is available
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    # Create RNN model
    model = RNNmodel(vocab_size,vocab_size,300,1,device)
    model.to(device)

    # Define Loss 
    loss = nn.CrossEntropyLoss()
    #Use Adam as the optimizer 
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 10

    for epoch in range(100):
        for i in range(len(lines)):
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            x = (torch.from_numpy(create_one_hot(input_sequence[i], vocab_size))).to(device)
            if len(x[0]) == 0: 
                continue
            y = torch.Tensor(target_sequence[i]).to(device)
            output, hidden = model(x)
            
            lossValue = loss(output, y.view(-1).long()) 
            lossValue.backward() 
            optimizer.step()

        print("Loss: {:.4f}".format(lossValue.item()))

    # Predict the next character and return it along with the hidden state
    def predict(model, character):
        characterInput = np.array([charInt[c] for c in character])
        characterInput = create_one_hot(characterInput, vocab_size)
        characterInput = torch.from_numpy(characterInput)
        characterInput = characterInput.to(device)
        out, hidden = model(characterInput)

        prob = nn.functional.softmax(out[-1], dim=0).data
        
        # Taking the highest probability score from the output
        char_ind = torch.topk(prob,2,dim=0,largest=True,sorted=True)[1]
        if char_ind[0].item() == 65:
            return intChar[char_ind[1].item()], hidden
        return intChar[char_ind[0].item()], hidden

    def sample(model, out_len, start):
        model.eval() 

        # First off, run through the starting characters
        output = [ch for ch in start]
        size = out_len - len(output)

        # Now pass in the previous characters and get a new one
        for i in range(size):
            char, h = predict(model, output)
            output.append(char)
        
        return ''.join(output)
    
    print(sample(model, 1000, "First Citizen:\n"))

if __name__ == "__main__":
    main()

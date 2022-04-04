import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from LSTM import LSTMmodel

# Create one hots
def create_one_hot(sequence, vocab_size, seq_len, batch_size):
    encoding = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)

    for i in range(batch_size): 
        for j in range(seq_len): 
            encoding[i, j, sequence[i][j]] = 1 
        
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
    
    intChar[len(intChar)] = '^'
    charInt['^'] = len(charInt)

    # Longest string's length
    maxlen = len(max(lines, key=len))
    
    # Add padding for batch updates
    for i in range(len(lines)):
        while len(lines[i]) <= maxlen:
            lines[i] += '^'
    
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
    input_sequence = torch.LongTensor(input_sequence)
    target_sequence = torch.LongTensor(target_sequence)
    
    # Check if GPU is available
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    # Create RNN model
    model = LSTMmodel(vocab_size,vocab_size,300,3,maxlen,device)
    model.to(device)

    # Define Loss 
    loss = nn.CrossEntropyLoss()
    #Use Adam as the optimizer 
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 1

    for epoch in range(150):
        state_h, state_c = model.init_hidden(maxlen)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        for i in range(1, len(lines)//batch_size):
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            input = input_sequence[(i-1)*batch_size:i*batch_size].to(device)

            output, (state_h, state_c) = model(input, (state_h, state_c))
            lossValue = loss(output[0], target_sequence[(i-1)*batch_size:i*batch_size].to(device).view(-1).long())
            
            state_h = state_h.detach()
            state_c = state_c.detach()

            lossValue.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

        print("Loss: {:.4f}".format(lossValue.item()))

    # Predict the next character and return it along with the hidden state
    def predict(model, character):
        state_h, state_c = model.init_hidden(len(character))
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        characterInput = np.array([[charInt[c] for c in character]])
        
        characterInput = torch.from_numpy(characterInput)
        characterInput = characterInput.to(device)

        out, (state_h, state_c) = model(characterInput, (state_h, state_c))

        prob = nn.functional.softmax(out[0][-1], dim=0).data
        
        # Taking the highest probability score from the output
        char_ind = torch.topk(prob,3,dim=0,largest=True,sorted=True)[1]
        if intChar[char_ind[0].item()] == '^':
            return intChar[char_ind[1].item()]
        return intChar[char_ind[0].item()]

    def sample(model, out_len, start):
        model.eval() 

        # First off, run through the starting characters
        output = [ch for ch in start]
        size = out_len - len(output)

        # Now pass in the previous characters and get a new one
        for i in range(size):
            char = predict(model, output)
            output.append(char)
        
        return ''.join(output)
    
    def predict2(model, text, next_words=100):
        model.eval()

        output = [ch for ch in text]
        for i in range(0, next_words):
            state_h, state_c = model.init_hidden(len(output))
            state_h = state_h.to(device)
            state_c = state_c.to(device)
            x = torch.tensor([[charInt[c] for c in output]])
            x = x.to(device)
            
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).cpu().detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            output.append(intChar[word_index])

        return ''.join(output)

    print(predict2(model,"A"))

if __name__ == "__main__":
    main()

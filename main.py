import numpy as np
import torch
from torch import nn
from RNN import RNNmodel

# Create one hots
def create_one_hot(sequence, vocab_size, seq_len, batch_size):
    encoding = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)

    for i in range(batch_size): 
        for j in range(seq_len): 
            encoding[i, j, sequence[i][j]] = 1 
        
    return encoding

def main():

    # Read in txt data
    text_file = open("tiny-shakespeare.txt", "r")
 
    # Read whole file to a string
    input = text_file.read()
 
    # Close file
    text_file.close()
    
    # Create dictionaries for conversion
    characters = set(''.join(input)) 
    intChar = dict(enumerate(characters))
    charInt = {character: index for index, character in intChar.items()}
    
    # Create inputs and targets
    input_sequence = [] 
    target_sequence = [] 
    for i in range(len(input)//100): 
        #Remove the last character from the input sequence 
        input_sequence.append(input[i*100:((i+1)*100)-1]) 
        #Remove the first element from target sequences 
        target_sequence.append(input[(i*100)+1:((i+1)*100)])

    # Replace all characters with integers
    for i in range(len(input_sequence)): 
        input_sequence[i] = [charInt[character] for character in input_sequence[i]] 
        target_sequence[i] = [charInt[character] for character in target_sequence[i]]
    
    vocab_size = len(charInt)
    
    # Create one hots and tensors
    input_sequence = create_one_hot(input_sequence, vocab_size, 99, len(input_sequence))
    input_sequence = torch.from_numpy(input_sequence)
    target_sequence = torch.Tensor(target_sequence)
    
    # Check if GPU is available
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    # Create RNN model
    model = RNNmodel(vocab_size,vocab_size,300,2,device)
    model.to(device)

    # Define Loss 
    loss = nn.CrossEntropyLoss()

    #Use Adam as the optimizer 
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 10
    num_epochs = 50

    # Train the model
    for epoch in range(num_epochs):
        for i in range(1, len(input_sequence)//batch_size):
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            input = input_sequence[(i-1)*batch_size:i*batch_size]
            
            output, hidden = model(input)
            
            lossValue = loss(output, target_sequence[(i-1)*batch_size:i*batch_size].to(device).view(-1).long())
            lossValue.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

        print(f"Epoch: {epoch+1}", "  Loss: {:.4f}".format(lossValue.item()))

    # Predict the next character and return it along with the hidden state
    def predict(model, character):
        # Create one hot for input
        characterInput = np.array([[charInt[c] for c in character]])
        characterInput = create_one_hot(characterInput, vocab_size, characterInput.shape[1], 1)
        characterInput = torch.from_numpy(characterInput).to(device)
        
        out, hidden = model(characterInput)
        
        # Compute soft max
        prob = nn.functional.softmax(out[-1], dim=0).data
        
        # Taking the highest probability score from the output
        char_ind = torch.topk(prob,2,dim=0,largest=True,sorted=True)[1]
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
    
    print(sample(model, 300, "All:\n"))

if __name__ == "__main__":
    main()

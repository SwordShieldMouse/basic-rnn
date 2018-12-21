import glob
import os
import string
import unicodedata
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

dtype = torch.float

# the rnn for name classification
class Classification_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classification_RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size

        # keep the hidden state
        self.s = torch.zeros(1, hidden_size, dtype = dtype, requires_grad = False)

        # input weights
        self.U = nn.Parameter(torch.randn(input_size, hidden_size, dtype = dtype, requires_grad = True))

        # context weights
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype = dtype, requires_grad = True))

        # output weights
        self.V = nn.Parameter(torch.randn(hidden_size, output_size, dtype = dtype, requires_grad = True))

    def forward(self, x):
        # try leaky relu?
        self.s = F.leaky_relu(x.mm(self.U) + self.s.mm(self.W))

        return F.softmax(self.s.mm(self.V))

    def reset_hidden(self):
        self.s = torch.zeros(1, self.hidden_size, dtype = dtype, requires_grad = False)

class Generative_RNN(nn.Module):
    # generates a name character by character when give a category and a starting character
    def __init__(self, category_size, input_size, hidden_size1, hidden_size2, output_size):
        super(Generative_RNN, self).__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.dropout = nn.Dropout(p = 0.5)

        # hidden layers
        self.hidden1 = nn.Linear(category_size + input_size + hidden_size1, hidden_size1)
        self.hidden2 = nn.Linear(category_size + hidden_size1 + hidden_size2, hidden_size2)
        self.hidden3 = nn.Linear(hidden_size2, output_size)

        # hold the hidden states
        self.s1 = torch.zeros(1, hidden_size1, dtype = dtype)
        self.s2 = torch.zeros(1, hidden_size2, dtype = dtype)


    def forward(self, category, x):
        self.s1 = F.leaky_relu(self.hidden1(torch.cat((category, x, self.s1), 1)))
        self.s1 = self.dropout(self.s1)
        self.s2 = F.leaky_relu(self.hidden2(torch.cat((category, self.s1, self.s2), 1)))
        self.s2 = self.dropout(self.s2)

        output = self.dropout(self.hidden3(self.s2))
        return F.softmax(output)

    def reset_hidden(self):
        self.s1 = torch.zeros(1, self.hidden_size1, dtype = dtype, requires_grad = False)
        self.s2 = torch.zeros(1, self.hidden_size2, dtype = dtype, requires_grad = False)

class Names(Dataset):
    def __init__(self):
        self.data = []
        for filename in glob.glob("data/names/*.txt"):
            category = os.path.basename(filename).split('.')[0]
            with open(filename, encoding = 'utf-8') as file:
                for line in file:
                    #print(line)
                    self.data.append([unicode_to_ascii(line.strip()), category])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix]

def find_files(path):
    return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


category_lines = {}
all_categories = []

def read_lines(filename):
    lines = open(filename, encoding = 'utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

# need to convert to ascii otherwise the one-hot encoding would be ridiculous
# code from http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(name):
    return ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# read the files
for filename in find_files('data/names/*.txt'):
    category = os.path.basename(filename).split('.')[0]
    all_categories.append(category)
    category_lines[category] = read_lines(filename)

n_categories = len(all_categories)

#print(category_lines["French"])

# do one-hot encoding
def letter_to_index(letter):
    return all_letters.find(letter)

# turn a line into an array of one-hot tensors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters) # 1 is just the batch size
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def category_to_tensor(category):
    tensor = torch.zeros(1, n_categories)
    tensor[0][all_categories.index(category)] = 1
    return tensor

# get the top category from the softmax output
def category_from_output(output):
    top_n, top_ix = output.topk(1)
    category_i = top_ix[0].item()
    return all_categories[category_i], category_i

# get a random item from a list
def rand_element(l):
    return l[np.random.randint(len(l))]

# draw a random line, category training example
def rand_training_example():
    category = rand_element(all_categories)
    line = rand_element(category_lines[category])
    return (line_to_tensor(line), torch.LongTensor([all_categories.index(category)]))

#input = line_to_tensor("Alan")

n_hidden = 64
epochs = 2
#n_iters = 5000
learning_rate = 1e-4

rnn = Classification_RNN(n_letters, n_hidden, n_categories)
generator_rnn = Generative_RNN(n_categories, n_letters, n_hidden, n_hidden, n_letters)

#print(rnn(input[0]))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)

generator_optimizer = torch.optim.Adam(generator_rnn.parameters(), lr = learning_rate)


data = Names()

# training, test sets
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

trainloader = DataLoader(train_data, batch_size = 10, shuffle = True)
testloader = DataLoader(test_data, shuffle = True)

# train
for _ in range(epochs):
    for line, y in trainloader:
        rnn.reset_hidden()
        generator_rnn.reset_hidden()

        generator_loss = 0

        y_pred = torch.zeros(len(line), n_categories)

        for ix in range(len(line)):
            x = line_to_tensor(line[ix])
            #print(x)
            rnn.reset_hidden()
            generator_rnn.reset_hidden()
            for char_ix, char in enumerate(x):
                #print(char)
                y_pred[ix] = rnn(char)
                next_letter = generator_rnn(category_to_tensor(y[ix]), char)

                # have to do char.find(1) as crossentropy takes a class index for the second argument
                #print(char[0])
                generator_loss += criterion(next_letter, (char[0] == 1).nonzero()[0])

        #print(category_to_tensor(y))
        # convert categories to tensors
        y = torch.tensor([all_categories.index(category) for category in y])
        #print(y)

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# test it
def evaluate(line_tensor):
    y_pred = torch.zeros(1, n_categories)
    for character in line_tensor:
        y_pred = rnn(character)
    return y_pred

#print(category_from_output(evaluate(line_to_tensor("Davis"))))
hits = 0
for line, y in testloader:
    y_pred = category_from_output(evaluate(line_to_tensor(line)))
    if y_pred[0] == y:
        hits += 1

print("accuracy on test set for classification was: " + str(hits / len(test_data)))


# generate a name

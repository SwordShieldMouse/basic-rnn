import glob
import os
import string
import unicodedata
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

dtype = torch.float

# the rnn
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

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


# Dataset

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

n_hidden = 128
epochs = 10
n_iters = 5000
learning_rate = 1e-4

rnn = RNN(n_letters, n_hidden, n_categories)

#print(rnn(input[0]))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)

for _ in range(n_iters):
    line, y = rand_training_example()
    rnn.reset_hidden()

    # run through the characters in the line
    y_pred = torch.zeros(1, n_categories)
    for character in line:
        #print(character)
        y_pred = rnn(character)

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

print(category_from_output(evaluate(line_to_tensor("Davis"))))

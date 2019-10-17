#!/usr/bin/env python
# coding: utf-8

import torch
import os

class Merlin(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_dim, lstm = False):
        super(Merlin, self).__init__()

        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_dim = output_dim
        self.lstm = lstm

        print("Merlin Model", self.input_size,self.hidden_size, self.output_dim)
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = torch.nn.Linear(self.hidden_size, self.hidden_size)

        if lstm:
            self.layer6 = torch.nn.LSTM(self.hidden_size, self.hidden_size, bidirectional = True)
            self.fc_out = torch.nn.Linear(self.hidden_size * 2, self.output_dim)
        else:
            self.layer6 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.fc_out = torch.nn.Linear(self.hidden_size, self.output_dim)


        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):

        hidden1 = self.relu(self.fc1(x))
        hidden2 = self.relu(self.fc2(hidden1))
        hidden3 = self.relu(self.fc3(hidden2))
        hidden4 = self.relu(self.fc4(hidden3))
        hidden5 = self.relu(self.fc5(hidden4))

        # hidden1 = torch.tanh(self.fc1(x))
        # hidden2 = torch.tanh(self.fc2(hidden1))
        # hidden3 = torch.tanh(self.fc3(hidden2))
        # hidden4 = torch.tanh(self.fc4(hidden3))
        # hidden5 = torch.tanh(self.fc5(hidden4))

        if self.lstm:
            out, hidden6 = self.layer6(hidden5)
        else:
            hidden6 = self.relu(self.layer6(hidden5))

        output = self.fc_out(hidden6)
        #output = self.sigmoid(output)
        return output



    def save(self, path):
        torch.save(self.state_dict(), path)

    def restore(self, path):
        if not os.path.exists(path):
            print('\nNew Training Session...\n')
            self.save(path)
        else:
            print(f'\nLoading Weights: "{path}"\n')
            self.load(path)

    def load(self, path, device='cpu'):
        # because PyTorch places on CPU by default, we follow those semantics by using CPU as default.
        self.load_state_dict(torch.load(path, map_location=device), strict=False)




# In[ ]:

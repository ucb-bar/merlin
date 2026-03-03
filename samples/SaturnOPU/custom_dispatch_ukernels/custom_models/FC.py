import torch.nn as nn

class SimpleFCModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, output_size=10):
        super(SimpleFCModel, self).__init__()
        self.fc = nn.Linear(input_size,   hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc(x))
        return self.softmax(self.fc2(x))
    
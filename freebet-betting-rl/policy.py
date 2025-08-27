import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim=6):  # 6 actions = 5 mises + stop
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)        # scores non normalisés
        probs = F.softmax(logits, dim=-1)  # distribution de probabilité
        return probs

import torch.nn as nn
import torch.nn.functional as F

class MarelleDualHeadNet(nn.Module):
    def __init__(self, input_size=24, hidden_size=128):
        super().__init__()
        # Tronc commun (feature extractor)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Tête 1 : placement
        self.head_place = nn.Linear(hidden_size, input_size)  # 24 positions possibles

        # Tête 2 : retrait
        self.head_remove = nn.Linear(hidden_size, input_size)  # 24 positions possibles

    def forward(self, x):
        # x : (batch_size, 24) représentation plate de l’état du plateau
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits_place = self.head_place(x)   # distribution brute pour placement
        logits_remove = self.head_remove(x) # distribution brute pour retrait

        return logits_place, logits_remove

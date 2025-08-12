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


# -------------------------
# Model 2-head Actor–Critic
# -------------------------
class ActorCriticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Actor heads
        self.actor_place = nn.Linear(64, 24)
        self.actor_remove = nn.Linear(64, 24)
        # Critic
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        logits_place = self.actor_place(x)
        logits_remove = self.actor_remove(x)
        value = self.critic(x)
        return logits_place, logits_remove, value


class ActorCriticModelLarge(nn.Module):
    """
    Actor-Critic avec ~5x le nombre de paramètres de l'ActorCriticModel original.
    Hidden size choisi : 176 (approx. 44k paramètres au total).
    """
    def __init__(self, input_size=24, hidden_size=176):
        super().__init__()
        # Tronc partagé
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Actor heads
        self.actor_place = nn.Linear(hidden_size, input_size)
        self.actor_remove = nn.Linear(hidden_size, input_size)
        # Critic
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        logits_place = self.actor_place(x)
        logits_remove = self.actor_remove(x)
        value = self.critic(x)
        return logits_place, logits_remove, value

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

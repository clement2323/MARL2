from .agents import BaseAgent
from .strategies import ModelStrategy, greedy_placement, SmartPlacement, SmartRemoval, block_opponent, CanonicalModelStrategy
from .modeles import MarelleDualHeadNet, ActorCriticModel,ActorCriticModelLarge

import torch

def create_agent_offensif():
    return BaseAgent(
        player_id=1,
        placement_strategy=greedy_placement,
        removal_strategy=SmartRemoval(1),
        name="offensif"
    )

def create_agent_defensif():
    return BaseAgent(
        player_id=1,
        placement_strategy=block_opponent,
        removal_strategy=None,
        name="defensif"
    )

def create_smart_agent():
    return BaseAgent(
        player_id=1,
        placement_strategy=SmartPlacement(1),
        removal_strategy=SmartRemoval(1),
        name="smart"
    )

def create_ml_agent(model_path="save_models/marelle_model_final.pth", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MarelleDualHeadNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return BaseAgent(
        player_id=1,
        placement_strategy=ModelStrategy(model, 1, mode="placement", device=device),
        removal_strategy=ModelStrategy(model, 1, mode="removal", device=device),
        name="ML Agent "+model_path
    )


  
def create_actor_critic_agent(model_path="save_models/marelle_model_actor_critic_2heads.pth",id=1, device=None):
    """
    Charge un ActorCriticModel entraîné et crée un agent
    utilisant uniquement les deux têtes Actor (placement et removal),
    compatible avec ModelStrategy.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Wrapper qui expose seulement les deux têtes actor
    class ActorOnlyModel(torch.nn.Module):
        def __init__(self, actor_critic_model):
            super().__init__()
            self.shared_layers = actor_critic_model.shared_layers
            self.actor_place = actor_critic_model.actor_place
            self.actor_remove = actor_critic_model.actor_remove
        
        def forward(self, x):
            x = self.shared_layers(x)
            logits_place = self.actor_place(x)
            logits_remove = self.actor_remove(x)
            return logits_place, logits_remove  # PAS de critic

    # Charger le modèle complet
    full_model = ActorCriticModel()
    full_model.load_state_dict(torch.load(model_path, map_location=device))
    full_model.to(device)
    full_model.eval()

    # Créer la version Actor-only
    actor_model = ActorOnlyModel(full_model).to(device)
    actor_model.eval()

    # Retourner un agent BaseAgent avec les deux stratégies ML
    return BaseAgent(
        player_id=id,
        placement_strategy=ModelStrategy(actor_model, id, mode="placement", device=device),
        removal_strategy=ModelStrategy(actor_model, id, mode="removal", device=device),
        name=f"ActorCritic Agent (Actor Only) [{model_path}]"
    )

# --- create_actor_critic_agent_large avec CanonicalModelStrategy intégré ---
def create_actor_critic_agent_large(model_path="save_models/marelle_model_actor_critic_2heads.pth", id =1, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Wrapper qui expose seulement les deux têtes actor
    class ActorOnlyModel(torch.nn.Module):
        def __init__(self, actor_critic_model):
            super().__init__()
            self.shared_layers = actor_critic_model.shared_layers
            self.actor_place = actor_critic_model.actor_place
            self.actor_remove = actor_critic_model.actor_remove
        
        def forward(self, x):
            x = self.shared_layers(x)
            logits_place = self.actor_place(x)
            logits_remove = self.actor_remove(x)
            return logits_place, logits_remove  # PAS de critic

    # Charger le modèle complet
    full_model = ActorCriticModelLarge()
    full_model.load_state_dict(torch.load(model_path, map_location=device))
    full_model.to(device)
    full_model.eval()

    # Créer la version Actor-only
    actor_model = ActorOnlyModel(full_model).to(device)
    actor_model.eval()

    # Utiliser la stratégie canonique (hérite de ModelStrategy mais override __call__)
    placement_strategy = CanonicalModelStrategy(actor_model, player_id=id, device=device, mode="placement")
    removal_strategy = CanonicalModelStrategy(actor_model, player_id=id, device=device, mode="removal")

    return BaseAgent(
        player_id=id,
        placement_strategy=placement_strategy,
        removal_strategy=removal_strategy,
        name=f"ActorCritic Agent (Actor Only + Canonical) [{model_path}]"
    )

# Dictionnaire de tous les agents
AGENTS = {
    "offensif": create_agent_offensif,
    "defensif": create_agent_defensif,
    "smart": create_smart_agent,
    "ml": create_ml_agent,
    "ac": create_actor_critic_agent,
    "ac_large": create_actor_critic_agent_large,
    "random": lambda: BaseAgent(player_id=1, name="random")
} 
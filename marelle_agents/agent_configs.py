from .agents import BaseAgent
from .strategies import ModelStrategy, greedy_placement, SmartPlacement, SmartRemoval, block_opponent
from .modeles import MarelleDualHeadNet
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

def create_ml_agent(model_path="marelle_model_final.pth", device=None):
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
        name="ML Agent"
    )

# Dictionnaire de tous les agents
AGENTS = {
    "offensif": create_agent_offensif,
    "defensif": create_agent_defensif,
    "smart": create_smart_agent,
    "ml": create_ml_agent,
    "random": lambda: BaseAgent(player_id=1, name="random")
} 
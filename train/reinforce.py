import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import sys
import os
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marelle_agents.agents import BaseAgent
from marelle_agents.strategies import ModelStrategy, greedy_placement, SmartRemoval
from marelle_agents.modeles import MarelleDualHeadNet
from environnement.marelle_env import MarelleEnv
from marelle_agents.agent_configs import AGENTS



def train_reinforce(env, model, agent_contre, num_episodes=1000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gamma = 0.99

    loss_history = deque(maxlen=100)
    winner_history = deque(maxlen=100)

    best_loss = float("inf")
    best_model_state = None

    device_selected = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device_selected)

    for episode in range(num_episodes):
        env = MarelleEnv()

        # Agent ML
        placement_strategy = ModelStrategy(model, player_id=1, mode="placement", device=device_selected)
        removal_strategy = ModelStrategy(model, player_id=1, mode="removal", device=device_selected)
        agent_ml = BaseAgent(1, placement_strategy, removal_strategy)

        log_probs, rewards = [], []

        while not env.is_phase1_over():
            current_agent = agent_ml if env.current_player == 1 else agent_contre
            move = current_agent.choose_move(env)

            if env.waiting_for_removal:
                if env.current_player == 1:
                    state = torch.tensor(
                        placement_strategy.encode_state(env),
                        dtype=torch.float32, device=device_selected
                    ).unsqueeze(0)
                    _, logits_remove = model(state)
                    mask = placement_strategy._create_action_mask(env.get_removable_pawns(), 24)
                    probs = F.softmax(logits_remove + mask, dim=-1)
                    log_prob = torch.log(probs[0, move] + 1e-8)
                    log_probs.append(log_prob)
                env.remove_pawn(move)

            else:
                if env.current_player == 1:
                    state = torch.tensor(
                        placement_strategy.encode_state(env),
                        dtype=torch.float32, device=device_selected
                    ).unsqueeze(0)
                    logits_place, _ = model(state)
                    mask = placement_strategy._create_action_mask(env.get_legal_moves(), 24)
                    probs = F.softmax(logits_place + mask, dim=-1)
                    log_prob = torch.log(probs[0, move] + 1e-8)
                    log_probs.append(log_prob)
                env.play_move(move)

        # Reward final basé sur get_winner()
        winner = env.get_winner()
        reward = 1 if winner == 1 else -1 if winner == -1 else 0
        rewards = [reward] * len(log_probs)

        # Calcul des retours (REINFORCE)
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device_selected)

        # Normalisation (centrage-réduction)
        # Loss = -E[log_prob * return]
        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        winner_history.append(winner)

        if episode % 1000 == 0:
            avg_loss = sum(loss_history) / len(loss_history)
            avg_win_rate = sum(1 for w in winner_history if w == 1) / len(winner_history)

            print(f"[Episode {episode}] Avg Loss={avg_loss:.4f} Avg Win Rate={avg_win_rate:.2%}")

    return model

def load_trained_model(model_path="marelle_model_final.pth", device=None):
    """Charge un modèle entraîné depuis un fichier."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MarelleDualHeadNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.train()

    return model

def train_against_multiple_agents_mixed(model, agents_to_train_against, nb_changement_agents = 300,  episodes_per_agent = 200):
    """Entraîne contre tous les agents en mélangeant les épisodes."""
    all_agents = list(agents_to_train_against.values())
    
    for episode in range(episodes_per_agent * nb_changement_agents):
        # Choisir un agent aléatoirement
        agent_contre = random.choice(all_agents)()
        
        # Entraînement normal
        model = load_trained_model("marelle_model_final.pth")
        trained_model = train_reinforce(
            env=MarelleEnv(),
            model=model,
            agent_contre=agent_contre,
            num_episodes=episodes_per_agent
        )
        torch.save(trained_model.state_dict(), "marelle_model_final.pth")
        
        print(f"Entraînement contre {agent_contre.name} terminé")

if __name__ == "__main__":
    # Configuration des agents d'entraînement
    training_agents = {
        "ml": AGENTS["ml"],
        "smart": AGENTS["smart"], 
        "ml_again": AGENTS["ml"],
        "offensif": AGENTS["offensif"],
        "ml_again2": AGENTS["ml"],
        "defensif": AGENTS["defensif"],
    }
    
    # Entraînement séquentiel
    train_against_multiple_agents_mixed(model=None, agents_to_train_against=training_agents)

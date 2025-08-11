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
        
        # Variables pour tracker les moulins
        moulins_j1_avant = 0
        moulins_j2_avant = 0

        # Agent ML
        placement_strategy = ModelStrategy(model, player_id=1, mode="placement", device=device_selected)
        removal_strategy = ModelStrategy(model, player_id=1, mode="removal", device=device_selected)
        agent_ml = BaseAgent(1, placement_strategy, removal_strategy)

        log_probs, rewards = [], []

        while not env.is_phase1_over():
            # Compter les moulins avant le coup
            moulins_j1_avant = sum(1 for mill in env.mills if all(env.G.nodes[n]["state"] == 1 for n in mill))
            moulins_j2_avant = sum(1 for mill in env.mills if all(env.G.nodes[n]["state"] == -1 for n in mill))
            
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
                    # Reward immédiat pour le retrait
                    rewards.append(0.0)  # Ajouter reward même pour retrait
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
                    
                    # Reward immédiat pour le placement
                    env.play_move(move)
                    
                    # Compter les moulins après le coup
                    moulins_j1_apres = sum(1 for mill in env.mills if all(env.G.nodes[n]["state"] == 1 for n in mill))
                    moulins_j2_apres = sum(1 for mill in env.mills if all(env.G.nodes[n]["state"] == -1 for n in mill))
                    
                    # Reward immédiat pour les moulins
                    if moulins_j1_apres > moulins_j1_avant:
                        rewards.append(0.1)
                    elif moulins_j2_apres > moulins_j2_avant:
                        rewards.append(-0.1)
                    else:
                        rewards.append(0.0)
                else:
                    env.play_move(move)
                    # Pas de reward quand l'adversaire joue

        # Reward final basé sur get_winner()
        winner = env.get_winner()
        final_reward = 1 if winner == 1 else -1 if winner == -1 else 0
  
        # Ajouter le reward final à tous les rewards immédiats
        rewards = [r + final_reward for r in rewards]

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

        if episode % 500 == 0 and episode!=0:
            avg_loss = sum(loss_history) / len(loss_history)
            wins = sum(1 for w in winner_history if w == 1)
            losses = sum(1 for w in winner_history if w == -1)
            draws = sum(1 for w in winner_history if w == 0)
            total = len(winner_history)
            
            print(f"[{agent_contre.name}] Loss={avg_loss:.4f} W:{wins}/{total} L:{losses}/{total} D:{draws}/{total}")

    return model

def load_trained_model(model_path="marelle_model_final.pth", device=None):
    """Charge un modèle entraîné depuis un fichier."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_path is None:
        return MarelleDualHeadNet()
    
    model = MarelleDualHeadNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.train()

    return model

def train_against_multiple_agents_mixed(model_path, agents_to_train_against, nb_changement_agents = 30,  episodes_per_agent = 1000):
    """Entraîne contre tous les agents en mélangeant les épisodes."""
    all_agents = list(agents_to_train_against.values())
    
    for episode in range(episodes_per_agent * nb_changement_agents):
        # Choisir un agent aléatoirement
        agent_contre = random.choice(all_agents)()
        
        # Entraînement normal
        model = load_trained_model(model_path)
        trained_model = train_reinforce(
            env=MarelleEnv(),
            model=model,
            agent_contre=agent_contre,
            num_episodes=episodes_per_agent
        )
        torch.save(trained_model.state_dict(), "marelle_model_final.pth")
        
        #print(f"Entraînement contre {agent_contre.name} terminé")

if __name__ == "__main__":
    # Configuration des agents d'entraînement
    training_agents = {
        "ml": lambda: AGENTS["ml"]("marelle_model_final_1.pth"),
        "ml_again": lambda: AGENTS["ml"]("marelle_model_final_2.pth"),
        "ml_again2": lambda: AGENTS["ml"]("marelle_model_final_3.pth"),
    }
    
    
    # Entraînement séquentiel
    train_against_multiple_agents_mixed(model_path="marelle_model_final.pth", agents_to_train_against=training_agents)


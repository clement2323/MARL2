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
from marelle_agents.strategies import ModelStrategy
from marelle_agents.modeles import MarelleDualHeadNet
from environnement.marelle_env import MarelleEnv
from marelle_agents.agent_configs import AGENTS


def calculate_turn_reward(env_before, env_after, player_id):
    reward = 0.0
    moulins_j1_avant = sum(1 for mill in env_before.mills if all(env_before.G.nodes[n]["state"] == 1 for n in mill))
    moulins_j2_avant = sum(1 for mill in env_before.mills if all(env_before.G.nodes[n]["state"] == -1 for n in mill))
    moulins_j1_apres = sum(1 for mill in env_after.mills if all(env_after.G.nodes[n]["state"] == 1 for n in mill))
    moulins_j2_apres = sum(1 for mill in env_after.mills if all(env_after.G.nodes[n]["state"] == -1 for n in mill))

    # formation de moulins
    if player_id == 1:
        if moulins_j1_apres > moulins_j1_avant:
            reward += 0.1 * (moulins_j1_apres - moulins_j1_avant)
        if moulins_j2_apres > moulins_j2_avant:
            reward -= 0.1 * (moulins_j2_apres - moulins_j2_avant)
    else:
        if moulins_j2_apres > moulins_j2_avant:
            reward += 0.1 * (moulins_j2_apres - moulins_j2_avant)
        if moulins_j1_apres > moulins_j1_avant:
            reward -= 0.1 * (moulins_j1_apres - moulins_j1_avant)
    return reward


def save_env_state(env):
    env_copy = MarelleEnv()
    env_copy.G = env.G.copy()
    env_copy.current_player = env.current_player
    env_copy.waiting_for_removal = env.waiting_for_removal
    env_copy.pawns_to_place = env.pawns_to_place.copy()
    return env_copy


def load_trained_model(model_path=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MarelleDualHeadNet()
    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.train()
    return model


def train_against_multiple_agents_mixed_v2(
    model_path="save_models/marelle_model_final.pth",
    agents_to_train_against=None,
    total_episodes=30000,
    lr=1e-3,
    exploration_epsilon=0.1,
    switch_every=1,
    save_every=2000,
    log_every=500,
    device=None,
):
    """
    Entraîne un seul modèle contre plusieurs agents en *continu*.
    - model_path : chemin de sauvegarde / chargement initial
    - agents_to_train_against : dict de factories comme dans ton code (valeurs appelables)
    - total_episodes : nombre total d'épisodes à exécuter
    - switch_every : nombre d'épisodes entre changements d'adversaire (1 -> changer à chaque épisode)
    - save_every : sauvegarde le modèle tous les `save_every` épisodes
    - log_every : intervalle d'affichage des logs
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Préparation modèle + optim (chargé UNE seule fois)
    model = load_trained_model(model_path=model_path, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gamma = 0.99

    # Préparer la liste d'agents
    if agents_to_train_against is None:
        raise ValueError("agents_to_train_against doit être un dict non vide")
    agent_factories = list(agents_to_train_against.values())

    # Statistiques / logs
    rolling_loss = deque(maxlen=100)
    stats_by_agent = {name: {"episodes": 0, "wins": 0, "losses": 0, "draws": 0} for name in agents_to_train_against.keys()}
    overall = {"episodes": 0, "wins": 0, "losses": 0, "draws": 0}

    for ep in range(1, total_episodes + 1):
        # Choix d'adversaire (plus fréquent si switch_every petit)
        if (ep - 1) % switch_every == 0:
            selected_name = random.choice(list(agents_to_train_against.keys()))
            agent_contre = agents_to_train_against[selected_name]()
        # else reuse previous agent_contre

        env = MarelleEnv()
        placement_strategy = ModelStrategy(model, player_id=1, mode="placement", device=device)
        removal_strategy = ModelStrategy(model, player_id=1, mode="removal", device=device)
        agent_ml = BaseAgent(1, placement_strategy, removal_strategy)

        log_probs = []
        rewards = []
        reward_turn = 0.0

        # --- déroulé d'un épisode complet ---
        while not env.is_phase1_over():
            env_before = save_env_state(env)
            current_agent = agent_ml if env.current_player == 1 else agent_contre

            if env.current_player == 1:
                state = torch.tensor(placement_strategy.encode_state(env), dtype=torch.float32, device=device).unsqueeze(0)
                if env.waiting_for_removal:
                    if random.random() < exploration_epsilon:
                        move = random.choice(env.get_removable_pawns())
                    else:
                        move = current_agent.choose_move(env)

                    # suppression
                    _, logits_remove = model(state)
                    mask = placement_strategy._create_action_mask(env.get_removable_pawns(), 24)
                    probs = F.softmax(logits_remove + mask, dim=-1)
                    log_prob = torch.log(probs[0, move] + 1e-8)
                    log_probs.append(log_prob)
                    rewards.append(0.0)  # suppression : reward immédiat 0 par défaut
                    env.remove_pawn(move)
                else:
                    if random.random() < exploration_epsilon:
                        move = random.choice(env.get_legal_moves())
                    else:
                        move = current_agent.choose_move(env)

                    logits_place, _ = model(state)
                    mask = placement_strategy._create_action_mask(env.get_legal_moves(), 24)
                    probs = F.softmax(logits_place + mask, dim=-1)
                    log_prob = torch.log(probs[0, move] + 1e-8)
                    log_probs.append(log_prob)

                    env.play_move(move)
                    reward_turn += calculate_turn_reward(env_before, env, player_id=1)
            else:
                # adversaire joue
                move = current_agent.choose_move(env)
                if env.waiting_for_removal:
                    env.remove_pawn(move)
                else:
                    env.play_move(move)
                    reward_turn += calculate_turn_reward(env_before, env, player_id=1)
                    # on push le reward du tour adversaire (du point de vue joueur 1)
                    rewards.append(reward_turn)
                    reward_turn = 0.0

        # Episode fini -> reward final selon winner
        winner = env.get_winner()
        final_reward = 1.0 if winner == 1 else -1.0 if winner == -1 else 0.0

        # Ajout du final_reward à chaque reward intermédiaire
        rewards = [r + final_reward for r in rewards]

        # Calcul des retours (REINFORCE)
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        if len(returns) == 0:
            # cas où pas de rewards collectés (sécurité)
            returns = [final_reward] * len(log_probs) if len(log_probs) > 0 else [final_reward]

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        # Si plus de log_probs que returns (ou inverse), bout de sécurité
        n = min(len(log_probs), len(returns))
        log_probs_t = torch.stack(log_probs[:n])
        returns_t = returns[:n]

        loss = -torch.sum(log_probs_t * returns_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rolling_loss.append(loss.item())
        overall["episodes"] += 1
        if winner == 1:
            overall["wins"] += 1
            stats_by_agent[selected_name]["wins"] += 1
        elif winner == -1:
            overall["losses"] += 1
            stats_by_agent[selected_name]["losses"] += 1
        else:
            overall["draws"] += 1
            stats_by_agent[selected_name]["draws"] += 1
        stats_by_agent[selected_name]["episodes"] += 1

        # Logging périodique
        if ep % log_every == 0:
            avg_loss = sum(rolling_loss) / len(rolling_loss) if len(rolling_loss) > 0 else 0.0
            total = overall["episodes"]
            w = overall["wins"]
            l = overall["losses"]
            d = overall["draws"]
            
            # Calculer les pourcentages
            win_rate = (w / total) * 100 if total > 0 else 0
            loss_rate = (l / total) * 100 if total > 0 else 0
            draw_rate = (d / total) * 100 if total > 0 else 0
            
            print(f"EP {ep}/{total_episodes} | AvgLoss(100)={avg_loss:.4f} | Overall W:{win_rate:.1f}% L:{loss_rate:.1f}% D:{draw_rate:.1f}%")
            
            # détails par adversaire
            for name, s in stats_by_agent.items():
                e = s["episodes"]
                if e > 0:
                    # Calculer les pourcentages pour chaque agent
                    agent_win_rate = (s['wins'] / e) * 100
                    agent_loss_rate = (s['losses'] / e) * 100
                    agent_draw_rate = (s['draws'] / e) * 100
                    print(f"   - {name}: ep={e} W:{agent_win_rate:.1f}% L:{agent_loss_rate:.1f}% D:{agent_draw_rate:.1f}%")

        # sauvegarde périodique
        if ep % save_every == 0:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"[SAVE] modèle sauvegardé -> {model_path} (ep {ep})")

    # sauvegarde finale
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("[DONE] Entraînement terminé. Modèle sauvegardé.")

    return model, stats_by_agent, overall


if __name__ == "__main__":
    # Exemple d'usage : factories doivent produire des instances d'adversaires
    training_agents = {
      #  "ml": lambda: AGENTS["ml"]("save_models/marelle_model_best.pth"),
        "offensif": AGENTS["offensif"],
    }

    model, stats, overall = train_against_multiple_agents_mixed_v2(
        model_path="save_models/marelle_model_final.pth",
        agents_to_train_against=training_agents,
        total_episodes=30000,
        lr=1e-3,
        exploration_epsilon=0.15,
        switch_every=1,      # change d'adversaire chaque épisode
        save_every=10000,
        log_every=1000,
    )

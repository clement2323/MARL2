import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import copy
from marelle_env import MarelleEnv

class MarelleDualHeadNetwork(nn.Module):
    """Réseau à deux têtes : placement et suppression"""
    def __init__(self, input_size=32, hidden_size=128):
        super().__init__()
        
        # Tronc commun
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Tête 1 : Politique de placement (24 positions)
        self.placement_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 24),
            nn.Softmax(dim=-1)
        )
        
        # Tête 2 : Politique de suppression (24 positions + 1 "pas de suppression")
        self.capture_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 25),  # 24 positions + 1 action "ne rien faire"
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        placement_policy = self.placement_head(shared_features)
        capture_policy = self.capture_head(shared_features)
        return placement_policy, capture_policy

class MarelleDualHeadAgent(BaseAgent):
    """Agent RL avec réseau à deux têtes"""
    def __init__(self, player_id, learning_rate=0.001, epsilon=0.1, lambda_weight=0.5):
        super().__init__(player_id)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.lambda_weight = lambda_weight
        
        # Réseau à deux têtes
        self.network = MarelleDualHeadNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Mémoire d'expériences
        self.memory = deque(maxlen=10000)
        
        # Fonction de récompense
        self.reward_function = self._calculate_reward
    
    def _calculate_reward(self, env, action, next_env):
        """Calcule la récompense basée sur les pions pris/perdus"""
        reward = 0
        
        # Compter les pions avant et après
        pions_before = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == self.player_id)
        pions_after = sum(1 for n in next_env.G.nodes() if next_env.G.nodes[n]["state"] == self.player_id)
        
        # Récompense pour les pions pris
        pions_taken = pions_after - pions_before
        reward += pions_taken * 2  # Bonus plus important pour les prises
        
        # Pénalité pour les pions perdus
        opponent_before = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == -self.player_id)
        opponent_after = sum(1 for n in next_env.G.nodes() if next_env.G.nodes[n]["state"] == -self.player_id)
        pions_lost = opponent_before - opponent_after
        reward -= pions_lost * 2
        
        # Bonus pour formation de moulin
        if self._forms_mill_after_action(env, action):
            reward += 5
        
        return reward
    
    def _forms_mill_after_action(self, env, action):
        """Vérifie si une action forme un moulin"""
        if env.waiting_for_removal:
            return False  # Pas de moulin lors d'une suppression
        
        # Simuler le placement
        original_state = env.G.nodes[action]["state"]
        env.G.nodes[action]["state"] = self.player_id
        forms_mill = env._forms_mill(action, self.player_id)
        env.G.nodes[action]["state"] = original_state
        return forms_mill
    
    def get_state_representation(self, env):
        """État enrichi pour le réseau"""
        state = []
        
        # État du plateau (24 positions)
        for node in sorted(env.G.nodes()):
            node_state = env.G.nodes[node]["state"]
            if node_state == self.player_id:
                state.append(1)
            elif node_state == -self.player_id:
                state.append(-1)
            else:
                state.append(0)
        
        # Features stratégiques
        state.extend([
            env.pawns_to_place[self.player_id] / 9.0,
            env.pawns_to_place[-self.player_id] / 9.0,
            1.0 if env.waiting_for_removal else 0.0,
            len(env.get_legal_moves()) / 24.0,
            self._count_our_mills(env) / 8.0,
            self._count_opponent_mills(env) / 8.0,
            self._count_mill_threats(env) / 8.0,
            self._evaluate_board_control(env)
        ])
        
        return torch.FloatTensor(state)
    
    def choose_move(self, env):
        """Choisit un coup en utilisant la tête appropriée"""
        if env.waiting_for_removal:
            return self._choose_capture(env)
        else:
            return self._choose_placement(env)
    
    def _choose_placement(self, env):
        """Utilise la tête de placement"""
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        state = self.get_state_representation(env)
        with torch.no_grad():
            placement_policy, _ = self.network(state)
        
        # Filtrer les coups légaux
        legal_probs = []
        for move in legal_moves:
            legal_probs.append((move, placement_policy[move].item()))
        
        # Choisir selon la politique
        total_prob = sum(prob for _, prob in legal_probs)
        if total_prob > 0:
            # Normaliser les probabilités
            legal_probs = [(move, prob/total_prob) for move, prob in legal_probs]
            moves, probs = zip(*legal_probs)
            chosen_move = np.random.choice(moves, p=probs)
        else:
            chosen_move = random.choice(legal_moves)
        
        return chosen_move
    
    def _choose_capture(self, env):
        """Utilise la tête de capture"""
        removable = env.get_removable_pawns()
        if not removable:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(removable)
        
        state = self.get_state_representation(env)
        with torch.no_grad():
            _, capture_policy = self.network(state)
        
        # Filtrer les pions supprimables
        legal_probs = []
        for pawn in removable:
            legal_probs.append((pawn, capture_policy[pawn].item()))
        
        # Choisir selon la politique
        total_prob = sum(prob for _, prob in legal_probs)
        if total_prob > 0:
            legal_probs = [(pawn, prob/total_prob) for pawn, prob in legal_probs]
            pawns, probs = zip(*legal_probs)
            chosen_pawn = np.random.choice(pawns, p=probs)
        else:
            chosen_pawn = random.choice(removable)
        
        return chosen_pawn
    
    def store_experience(self, state, action, reward, next_state, done, action_type):
        """Stocke une expérience avec le type d'action"""
        self.memory.append((state, action, reward, next_state, done, action_type))
    
    def train(self, batch_size=32):
        """Entraînement avec loss combinée"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, action_types = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)
        
        # Forward pass
        placement_policy, capture_policy = self.network(states)
        next_placement_policy, next_capture_policy = self.network(next_states)
        
        # Calculer les losses
        placement_loss = 0
        capture_loss = 0
        
        for i in range(len(actions)):
            if action_types[i] == "placement":
                # Loss pour la tête de placement
                target_value = rewards[i] + 0.9 * torch.max(next_placement_policy[i]) if not dones[i] else rewards[i]
                placement_loss += F.mse_loss(placement_policy[i][actions[i]], target_value)
            
            elif action_types[i] == "capture":
                # Loss pour la tête de capture
                target_value = rewards[i] + 0.9 * torch.max(next_capture_policy[i]) if not dones[i] else rewards[i]
                capture_loss += F.mse_loss(capture_policy[i][actions[i]], target_value)
        
        # Loss combinée
        total_loss = placement_loss + self.lambda_weight * capture_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def _count_our_mills(self, env):
        """Compte nos moulins actuels"""
        count = 0
        for mill in env.mills:
            if all(env.G.nodes[n]["state"] == self.player_id for n in mill):
                count += 1
        return count
    
    def _count_opponent_mills(self, env):
        """Compte les moulins adverses"""
        count = 0
        opponent = -self.player_id
        for mill in env.mills:
            if all(env.G.nodes[n]["state"] == opponent for n in mill):
                count += 1
        return count
    
    def _count_mill_threats(self, env):
        """Compte les menaces de moulins"""
        threats = 0
        for mill in env.mills:
            our_pawns = sum(1 for n in mill if env.G.nodes[n]["state"] == self.player_id)
            empty_spots = sum(1 for n in mill if env.G.nodes[n]["state"] == 0)
            if our_pawns == 2 and empty_spots == 1:
                threats += 1
        return threats
    
    def _evaluate_board_control(self, env):
        """Évalue le contrôle du plateau"""
        our_central = sum(1 for n in [1,3,5,7,9,11,13,15,17,19,21,23] 
                         if env.G.nodes[n]["state"] == self.player_id)
        opponent_central = sum(1 for n in [1,3,5,7,9,11,13,15,17,19,21,23] 
                              if env.G.nodes[n]["state"] == -self.player_id)
        return (our_central - opponent_central) / 12.0

def train_dual_head_agent(agent, opponent, episodes=1000):
    """Entraîne l'agent à deux têtes"""
    for episode in range(episodes):
        env = MarelleEnv()
        
        while not env.is_phase1_over():
            # Tour de l'agent RL
            if env.current_player == agent.player_id:
                state = agent.get_state_representation(env)
                
                if env.waiting_for_removal:
                    action = agent.choose_move(env)
                    action_type = "capture"
                else:
                    action = agent.choose_move(env)
                    action_type = "placement"
                
                if action is not None:
                    # Sauvegarder l'état avant
                    old_env = copy.deepcopy(env)
                    
                    # Jouer le coup
                    if action_type == "capture":
                        env.remove_pawn(action)
                    else:
                        env.play_move(action)
                    
                    # Calculer la récompense
                    reward = agent.reward_function(old_env, action, env)
                    
                    # Stocker l'expérience
                    next_state = agent.get_state_representation(env)
                    done = env.is_phase1_over()
                    agent.store_experience(state, action, reward, next_state, done, action_type)
            
            # Tour de l'adversaire
            else:
                opponent_move = opponent.choose_move(env)
                if opponent_move is not None:
                    env.play_move(opponent_move)
        
        # Entraîner l'agent
        agent.train()
        
        # Diminuer epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995) 
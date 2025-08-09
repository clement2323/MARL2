import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from marelle_env import MarelleEnv

class BaseAgent:
    """Classe de base pour tous les agents"""
    def __init__(self, player_id):
        self.player_id = player_id
    
    def choose_move(self, env):
        """Méthode à implémenter par chaque agent"""
        raise NotImplementedError

class RandomAgent(BaseAgent):
    """Agent qui joue aléatoirement"""
    def choose_move(self, env):
        legal_moves = env.get_legal_moves()
        if legal_moves:
            return random.choice(legal_moves)
        return None

class GreedyAgent(BaseAgent):
    """Agent qui essaie de former des moulins"""
    def choose_move(self, env):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None
        
        # Essayer de former un moulin
        for move in legal_moves:
            # Simuler le coup
            original_state = env.G.nodes[move]["state"]
            env.G.nodes[move]["state"] = self.player_id
            
            # Vérifier si ça forme un moulin
            if env._forms_mill(move, self.player_id):
                env.G.nodes[move]["state"] = original_state
                return move
            
            # Restaurer l'état
            env.G.nodes[move]["state"] = original_state
        
        # Sinon, jouer aléatoirement
        return random.choice(legal_moves)

class DefensiveAgent(BaseAgent):
    """Agent qui bloque les moulins adverses"""
    def choose_move(self, env):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None
        
        opponent = -self.player_id
        
        # Bloquer les moulins adverses
        for move in legal_moves:
            # Simuler le coup adverse
            original_state = env.G.nodes[move]["state"]
            env.G.nodes[move]["state"] = opponent
            
            # Vérifier si ça formerait un moulin adverse
            if env._forms_mill(move, opponent):
                env.G.nodes[move]["state"] = original_state
                return move  # Bloquer ce coup
            
            # Restaurer l'état
            env.G.nodes[move]["state"] = original_state
        
        # Sinon, utiliser l'agent greedy
        greedy = GreedyAgent(self.player_id)
        return greedy.choose_move(env)

class SmartAgent(BaseAgent):
    """Agent plus intelligent qui gère les moulins et la suppression"""
    def choose_move(self, env):
        # Si on doit supprimer un pion, le faire automatiquement
        if env.waiting_for_removal:
            removable = env.get_removable_pawns()
            if removable:
                return random.choice(removable)
        
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None
        
        # Essayer de former un moulin
        for move in legal_moves:
            # Simuler le coup
            original_state = env.G.nodes[move]["state"]
            env.G.nodes[move]["state"] = self.player_id
            
            # Vérifier si ça forme un moulin
            if env._forms_mill(move, self.player_id):
                env.G.nodes[move]["state"] = original_state
                return move
            
            # Restaurer l'état
            env.G.nodes[move]["state"] = original_state
        
        # Bloquer les moulins adverses
        opponent = -self.player_id
        for move in legal_moves:
            original_state = env.G.nodes[move]["state"]
            env.G.nodes[move]["state"] = opponent
            
            if env._forms_mill(move, opponent):
                env.G.nodes[move]["state"] = original_state
                return move
            
            env.G.nodes[move]["state"] = original_state
        
        # Sinon, jouer aléatoirement
        return random.choice(legal_moves) 

class MarelleRLAgent(BaseAgent):
    """Agent RL pour la marelle"""
    def __init__(self, player_id, learning_rate=0.001, epsilon=0.1):
        super().__init__(player_id)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        # Réseaux de neurones
        self.placement_network = MarelleNetwork()
        self.removal_network = MarelleNetwork()
        
        # Optimiseurs
        self.placement_optimizer = optim.Adam(self.placement_network.parameters(), lr=learning_rate)
        self.removal_optimizer = optim.Adam(self.removal_network.parameters(), lr=learning_rate)
        
        # Mémoire d'expérience
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
        reward += pions_taken
        
        # Pénalité pour les pions perdus
        opponent_before = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == -self.player_id)
        opponent_after = sum(1 for n in next_env.G.nodes() if next_env.G.nodes[n]["state"] == -self.player_id)
        pions_lost = opponent_before - opponent_after
        reward -= pions_lost
        
        return reward
    
    def get_state_representation(self, env):
        """Convertit l'état du jeu en vecteur pour le réseau"""
        state = []
        for node in sorted(env.G.nodes()):
            node_state = env.G.nodes[node]["state"]
            if node_state == self.player_id:
                state.append(1)
            elif node_state == -self.player_id:
                state.append(-1)
            else:
                state.append(0)
        
        # Ajouter des features supplémentaires
        state.extend([
            env.pawns_to_place[self.player_id] / 9.0,  # Jetons restants normalisés
            env.pawns_to_place[-self.player_id] / 9.0,
            1.0 if env.waiting_for_removal else 0.0,  # En attente de suppression
            len(env.get_legal_moves()) / 24.0  # Coups légaux normalisés
        ])
        
        return torch.FloatTensor(state)
    
    def choose_move(self, env):
        """Choisit un coup (placement ou suppression)"""
        if env.waiting_for_removal:
            return self._choose_removal(env)
        else:
            return self._choose_placement(env)
    
    def _choose_placement(self, env):
        """Choisit où placer un pion"""
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None
        
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Exploitation avec le réseau
        state = self.get_state_representation(env)
        with torch.no_grad():
            q_values = self.placement_network(state)
        
        # Filtrer les coups légaux
        legal_q_values = []
        for move in legal_moves:
            legal_q_values.append((move, q_values[move].item()))
        
        # Choisir le meilleur coup légal
        best_move = max(legal_q_values, key=lambda x: x[1])[0]
        return best_move
    
    def _choose_removal(self, env):
        """Choisit quel pion supprimer"""
        removable = env.get_removable_pawns()
        if not removable:
            return None
        
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            return random.choice(removable)
        
        # Exploitation avec le réseau de suppression
        state = self.get_state_representation(env)
        with torch.no_grad():
            q_values = self.removal_network(state)
        
        # Filtrer les pions supprimables
        legal_q_values = []
        for pawn in removable:
            legal_q_values.append((pawn, q_values[pawn].item()))
        
        # Choisir le meilleur pion à supprimer
        best_removal = max(legal_q_values, key=lambda x: x[1])[0]
        return best_removal
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stocke une expérience pour l'apprentissage"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=32):
        """Entraîne les réseaux sur un batch d'expériences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Entraîner le réseau de placement
        self._train_placement_network(states, actions, rewards, next_states, dones)
        
        # Entraîner le réseau de suppression
        self._train_removal_network(states, actions, rewards, next_states, dones)
    
    def _train_placement_network(self, states, actions, rewards, next_states, dones):
        """Entraîne le réseau de placement"""
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.placement_network(states)
        next_q_values = self.placement_network(next_states)
        
        # Calculer les Q-values cibles
        target_q_values = current_q_values.clone()
        for i in range(len(actions)):
            if not dones[i]:
                target_q_values[i][actions[i]] = rewards[i] + 0.9 * torch.max(next_q_values[i])
            else:
                target_q_values[i][actions[i]] = rewards[i]
        
        # Calculer la loss et faire la backpropagation
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.placement_optimizer.zero_grad()
        loss.backward()
        self.placement_optimizer.step()
    
    def _train_removal_network(self, states, actions, rewards, next_states, dones):
        """Entraîne le réseau de suppression"""
        # Similaire à _train_placement_network mais pour la suppression
        pass 

class MarelleNetwork(nn.Module):
    """Réseau de neurones pour la marelle"""
    def __init__(self, input_size=28, hidden_size=64, output_size=24):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MarelleValueNetwork(nn.Module):
    """Réseau qui évalue la valeur d'une position"""
    def __init__(self, input_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Une seule valeur de sortie
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class StrategicMarelleRLAgent(BaseAgent):
    """Agent RL qui planifie ses actions sur plusieurs coups"""
    def __init__(self, player_id, learning_rate=0.001, epsilon=0.1):
        super().__init__(player_id)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        # Un seul réseau qui évalue les états
        self.value_network = MarelleValueNetwork()
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Mémoire d'expériences
        self.memory = deque(maxlen=10000)
    
    def choose_move(self, env):
        """Choisit le meilleur coup en considérant les conséquences futures"""
        if env.waiting_for_removal:
            return self._choose_removal_strategic(env)
        else:
            return self._choose_placement_strategic(env)
    
    def _choose_placement_strategic(self, env):
        """Placement en considérant les moulins futurs"""
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        best_move = None
        best_value = float('-inf')
        
        for move in legal_moves:
            # Simuler le coup
            future_env = self._simulate_move(env, move)
            
            # Évaluer la valeur de cette position future
            value = self._evaluate_position(future_env)
            
            # Bonus pour formation de moulin immédiat
            if self._forms_mill_after_move(env, move):
                value += 10  # Bonus important pour moulin
            
            # Bonus pour bloquer les moulins adverses
            if self._blocks_opponent_mill(env, move):
                value += 5
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    def _choose_removal_strategic(self, env):
        """Suppression en considérant l'impact stratégique"""
        removable = env.get_removable_pawns()
        if not removable:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(removable)
        
        best_removal = None
        best_value = float('-inf')
        
        for pawn in removable:
            # Simuler la suppression
            future_env = self._simulate_removal(env, pawn)
            
            # Évaluer l'impact de cette suppression
            value = self._evaluate_removal_impact(env, pawn)
            
            if value > best_value:
                best_value = value
                best_removal = pawn
        
        return best_removal
    
    def _evaluate_position(self, env):
        """Évalue la valeur d'une position"""
        state = self.get_state_representation(env)
        with torch.no_grad():
            return self.value_network(state).item()
    
    def _evaluate_removal_impact(self, env, pawn_to_remove):
        """Évalue l'impact stratégique d'une suppression"""
        opponent = -self.player_id
        
        # Pénalité pour supprimer un pion dans un moulin (plus difficile à recréer)
        if self._is_pawn_in_mill(env, pawn_to_remove, opponent):
            return -5
        
        # Bonus pour supprimer un pion qui bloque nos moulins
        if self._is_blocking_our_mills(env, pawn_to_remove):
            return 8
        
        # Bonus pour supprimer un pion central (plus stratégique)
        central_positions = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        if pawn_to_remove in central_positions:
            return 3
        
        return 1  # Valeur de base
    
    def _simulate_move(self, env, move):
        """Simule un coup et retourne l'environnement résultant"""
        import copy
        future_env = copy.deepcopy(env)
        future_env.play_move(move)
        return future_env
    
    def _simulate_removal(self, env, pawn):
        """Simule une suppression"""
        import copy
        future_env = copy.deepcopy(env)
        future_env.remove_pawn(pawn)
        return future_env
    
    def _forms_mill_after_move(self, env, move):
        """Vérifie si un coup forme un moulin"""
        original_state = env.G.nodes[move]["state"]
        env.G.nodes[move]["state"] = self.player_id
        forms_mill = env._forms_mill(move, self.player_id)
        env.G.nodes[move]["state"] = original_state
        return forms_mill
    
    def _blocks_opponent_mill(self, env, move):
        """Vérifie si un coup bloque un moulin adverse"""
        opponent = -self.player_id
        original_state = env.G.nodes[move]["state"]
        env.G.nodes[move]["state"] = opponent
        blocks_mill = env._forms_mill(move, opponent)
        env.G.nodes[move]["state"] = original_state
        return blocks_mill
    
    def _is_pawn_in_mill(self, env, pawn, player):
        """Vérifie si un pion fait partie d'un moulin"""
        return env._forms_mill(pawn, player)
    
    def _is_blocking_our_mills(self, env, pawn):
        """Vérifie si un pion bloque nos moulins potentiels"""
        # Logique pour détecter si la suppression de ce pion nous aide
        return False  # À implémenter selon vos besoins 

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
        """Entraînement avec loss combinée sur GPU"""
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
import copy
from environnement.marelle_env import MarelleEnv
from marelle_agents.rewards import RewardSystem

class TrainingEnv(MarelleEnv):
    """Environnement spécialisé pour l'entraînement avec rewards."""
    
    def __init__(self):
        super().__init__()
        self.reward_system = RewardSystem()
        self.episode_history = []
        
    def reset(self):
        super().reset()
        self.episode_history = []
        return self._get_state()
    
    def step(self, action, player_id):
        """Exécute une action et retourne (state, reward, done, info)."""
        old_state = copy.deepcopy(self._get_state())
        
        try:
            if self.waiting_for_removal:
                self.remove_pawn(action)
                reward = self.reward_system.calculate_reward(
                    self, action, player_id, 
                    game_over=False
                )
            else:
                self.play_move(action)
                reward = self.reward_system.calculate_reward(
                    self, action, player_id, 
                    game_over=False
                )
                
        except ValueError:
            reward = self.reward_system.INVALID_MOVE_PENALTY
            
        # Vérifier si la partie est terminée
        done = self._is_game_over()
        winner = self._get_winner() if done else None
        
        if done:
            episode_reward = self.reward_system.calculate_episode_reward(self, player_id, winner)
            reward += episode_reward
            
        info = {
            'winner': winner,
            'moves_count': len(self.moves),
            'pawns_remaining': self.pawns_to_place.copy()
        }
        
        new_state = self._get_state()
        self.episode_history.append((old_state, action, reward, new_state, done))
        
        return new_state, reward, done, info
    
    def _get_state(self):
        """Retourne l'état actuel encodé."""
        return [self.G.nodes[i]["state"] for i in range(24)]
    
    def _is_game_over(self):
        """Vérifie si la partie est terminée."""
        # Vérifier si un joueur n'a plus que 2 pions
        for player in [1, -1]:
            remaining_pawns = sum(1 for _, data in self.G.nodes(data=True) if data["state"] == player)
            if remaining_pawns < 3:
                return True
        return False
    
    def _get_winner(self):
        """Détermine le gagnant."""
        for player in [1, -1]:
            remaining_pawns = sum(1 for _, data in self.G.nodes(data=True) if data["state"] == player)
            if remaining_pawns < 3:
                return -player
        return None 
import random
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
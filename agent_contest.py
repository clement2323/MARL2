# tournament.py
import itertools
from environnement.marelle_env import MarelleEnv
from marelle_agents.agent_configs import AGENTS


def play_match(agent1, agent2, num_games=1000):
    """Joue num_games parties entre deux agents et renvoie (victoires_agent1, victoires_agent2, nuls)"""
    results = {1: 0, -1: 0, 0: 0}  # 1 = agent1 gagne, -1 = agent2 gagne, 0 = nul

    for g in range(num_games):
        env = MarelleEnv()

        # Alternance des rôles (moitié du temps agent1 est joueur 1, moitié du temps joueur -1)
        if g < num_games // 2:
            agents = {1: agent1, -1: agent2}
        else:
            agents = {1: agent2, -1: agent1}

        env.reset()

        while not env.is_phase1_over():
            move = agents[env.current_player].choose_move(env)

            if env.waiting_for_removal:    
                env.remove_pawn(move)
            else:
                env.play_move(move)

        # Comptage des pions restants pour désigner un vainqueur simple
        count_p1 = sum(1 for _, d in env.G.nodes(data=True) if d["state"] == 1)
        count_p2 = sum(1 for _, d in env.G.nodes(data=True) if d["state"] == -1)

        if count_p1 > count_p2:
            if g < num_games // 2:
                results[1] += 1
            else:
                results[-1] += 1
        elif count_p2 > count_p1:
            if g < num_games // 2:
                results[-1] += 1
            else:
                results[1] += 1
        else:
            results[0] += 1

    return results[1], results[-1], results[0]


def tournament(agent_list, num_games=100):
    """Fait un round robin et affiche les résultats"""
    scores = {agent.name: 0 for agent in agent_list}

    for a1, a2 in itertools.combinations(agent_list, 2):
        v1, v2, draws = play_match(a1, a2, num_games=num_games)
        scores[a1.name] += v1
        scores[a2.name] += v2
        print(f"{a1.name} vs {a2.name} : {v1} - {v2} ({draws} nuls)")

    print("\nClassement final :")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{name} : {score} points")


if __name__ == "__main__":

    # Agent totalement random
    agent_random = AGENTS["random"]()
    agent_offensif = AGENTS["offensif"]()
    agent_defensif = AGENTS["defensif"]()
    smart_agent = AGENTS["smart"]()
    agent_ml= AGENTS["ml"](model_path="save_models/marelle_model_final.pth") 
    agent_ac= AGENTS["ac"]()
    agent_ac_large= AGENTS["ac_large"]()
    

    agent_list = [agent_offensif,agent_ac,agent_ac_large]
    tournament(agent_list, num_games=600)

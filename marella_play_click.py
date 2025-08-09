# marelle_play_click.py
import matplotlib.pyplot as plt
import networkx as nx
from marelle_env import MarelleEnv
from visualisation import plot_marelle

class MarelleClickGame:
    def __init__(self):
        self.env = MarelleEnv()
        self.pos = nx.get_node_attributes(self.env.G, "pos")
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.refresh()

    def refresh(self):
        self.ax.clear()
        plot_marelle(self.env.G, self.env.moves, ax=self.ax)  # version modifiée pour accepter un axe existant
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Trouver le nœud le plus proche du clic
        clicked_node = None
        min_dist = float("inf")
        for n, (x, y) in self.pos.items():
            dx = event.xdata - x
            dy = event.ydata - y
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                clicked_node = n

        if clicked_node is None:
            return

        # Vérifier que c'est une case libre
        if self.env.G.nodes[clicked_node]["state"] != 0:
            print(f"❌ Case {clicked_node} déjà occupée")
            return

        # Jouer le coup
        self.env.play_move(clicked_node)
        print(f"✅ Joueur {self.env.current_player * -1} a joué sur la case {clicked_node}")

        self.refresh()

        if self.env.is_phase1_over():
            print("✅ Phase 1 terminée.")
            plt.close(self.fig)

def play_with_clicks():
    MarelleClickGame()
    plt.show()

play_with_clicks()

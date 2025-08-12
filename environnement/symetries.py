import random
from environnement.marelle_layout import marelle_layout

def generate_symmetry_mappings():
    layout = marelle_layout()
    coords_to_node = {pos: node for node, pos in layout.items()}

    def transform_coords(pos, sym):
        x, y = pos
        if sym == "identity": return (x, y)
        if sym == "rot90": return (y, -x)
        if sym == "rot180": return (-x, -y)
        if sym == "rot270": return (-y, x)
        if sym == "flip_h": return (-x, y)
        if sym == "flip_v": return (x, -y)
        if sym == "flip_d1": return (y, x)
        if sym == "flip_d2": return (-y, -x)

    mappings = {}
    for sym in ["identity","rot90","rot180","rot270","flip_h","flip_v","flip_d1","flip_d2"]:
        perm = []
        for node in range(24):
            new_coords = transform_coords(layout[node], sym)
            perm.append(coords_to_node[new_coords])
        mappings[sym] = perm
    return mappings


# Définition des permutations de nœuds pour chaque symétrie
# Chaque liste donne, pour un index original, le nouvel index après transformation
# Ces mappings doivent être calculés une fois à partir de ton layout (marelle_layout)
SYMMETRY_MAPPINGS = generate_symmetry_mappings()
# Assure-toi d'avoir ces imports en haut du fichier

# --- Helpers pour la canonicalisation (sémantique : mapping[new_index] = orig_index) ---
def to_canonical(state_vec):
    """
    state_vec : séquence de longueur 24 (list/tuple/numpy/torch)
    Retourne (canon_state_list, sym_name)
    On suppose SYMMETRY_MAPPINGS[sym] est une liste length 24 telle que
        state_sym[new_index] = state_orig[ SYMMETRY_MAPPINGS[sym][new_index] ]
    """
    s = list(state_vec)
    best = None
    best_sym = None
    for sym_name, mapping in SYMMETRY_MAPPINGS.items():
        # build tuple for lexicographic compare
        transformed = tuple(s[mapping[i]] for i in range(24))
        if (best is None) or (transformed < best):
            best = transformed
            best_sym = sym_name
    # return as list for downstream code expecting list-like
    return list(best), best_sym

def map_moves_to_canonical(legal_moves, sym_name):
    """
    legal_moves: list of original indices
    returns list of canonical indices
    """
    mapping = SYMMETRY_MAPPINGS[sym_name]           # mapping[new] = orig
    inverse = {orig: new for new, orig in enumerate(mapping)}  # orig -> new
    return [inverse[m] for m in legal_moves]

def map_move_from_canonical(canon_move_idx, sym_name):
    """
    canon_move_idx: index in canonical coordinate system (new index)
    returns original index = mapping[new]
    """
    mapping = SYMMETRY_MAPPINGS[sym_name]  # mapping[new] = orig
    return mapping[canon_move_idx]

from itertools import pairwise

from .tracking import generate_graph

def build_graph_v2(data, max_score, score_func, tracked):
    return generate_graph(data, max_score, score_func, tracked)

def generate_positions_list(df, tracked):
    data = []

    for _, dft in df.groupby('tframe'):
        positions = []
        for _, row in dft.iterrows():
            if tracked:
                positions.append((row['y'], row['x'], row['displ_y'], row['displ_x'], row['track_no']))
            else:
                positions.append((row['y'], row['x'], row['displ_y'], row['displ_x']))

        data.append(positions)

    # shared_state.NUM_DET_PER_FRAME = [len(sublist) for sublist in shared_state.DATA]
    return data

def track_to_posarray(G, track, data):
    if track is None:
        track = None
    else:
        track_pos = [-1] * len(data)
        for n in track:
            if G.nodes[n]["type"]=="D":
                track_pos[G.nodes[n]["time_point"]] = G.nodes[n]["idx"]
        return track_pos

def track_to_lines(G, track):
    lines = []
    for n1, n2 in pairwise(track):
        if (G.nodes[n1]["type"] == "D") and (G.nodes[n2]["type"] == "D"):
            lines.append([[G.nodes[n1]["y"], G.nodes[n1]["x"]], [G.nodes[n2]["y"], G.nodes[n2]["x"]]])
    return lines

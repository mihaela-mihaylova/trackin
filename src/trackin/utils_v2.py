from .tracking import generate_graph

def build_graph(data, max_score, score_func, tracked):
    return generate_graph(data, max_score, score_func, tracked)

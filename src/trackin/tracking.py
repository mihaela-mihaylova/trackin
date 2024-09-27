import sys
import networkx as nx
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heapq
from datetime import datetime

# a basic score function to be used when generating edges
def calculate_score(x1,y1,x2,y2,score):
    if score == 'linear':
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)
    elif score == 'squared':
        return ((x1-x2)**2 + (y1-y2)**2)
    elif score == 'constant':
        return 1
    else:
        return None  
    
# this function generates a graph, where nodes correspond to detections in frames (time points)
def generate_graph(points, max_score, score_func, tracked):
    # assign value of coefficient in case of edges between tracked positions
    if tracked:
        coeff = 1/max_score
    # this is the number of frames (time points)
    t_max = len(points)

    # to save list of nodes per frame, used for adding STXY-edges
    nodes_per_frame = [[] for _ in range(t_max)]

    # if the points list is empty, return an empty list
    if t_max == 0:
        return []
    # if max_score is 0, then an empty list is returned
    if max_score == 0:
        return []
    # if an invalid value is assigned to score_func, then an empty list is returned
    if score_func not in ['linear', 'squared']:
        return []
    # declare a directed graph
    G = nx.DiGraph()  
   
    # iterate over frames(time points) and detections contained in those and generate nodes
    for t, pt in enumerate(points):
        # add the detection nodes contained in frame t
        for i, pti in enumerate(pt):
            if not tracked:
                G.add_node(f"D_{t}_{i}", time_point = t, idx = i, y=pti[0], x=pti[1], displ_y=int(pti[2]), displ_x=int(pti[3]), type = 'D')
            else:
                G.add_node(f"D_{t}_{i}", time_point = t, idx = i, y=pti[0], x=pti[1], displ_y=int(pti[2]), displ_x=int(pti[3]), track_no = int(pti[4]), type = 'D')
            
            # add nodes to list of nodes per frame, used for adding STXY-edges later
            nodes_per_frame[t].append(f"D_{t}_{i}")

    # add D-D edges
    for t in range(t_max-1):
        for i in range(len(points[t])):
            # add edges between D_t_i's and D_t+1_j's (detections in two consecutive frames)
            for j in range(len(points[t+1])):
                # calculate distance between nodes and only assign edge if score <= max_score
                # coordinates of D_t_i
                x1 = G.nodes[f"D_{t}_{i}"]['displ_x']+G.nodes[f"D_{t}_{i}"]['x']
                y1 = G.nodes[f"D_{t}_{i}"]['displ_y']+G.nodes[f"D_{t}_{i}"]['y']

                # coordinates of D_t+1_j
                x2 = G.nodes[f"D_{t+1}_{j}"]['x']
                y2 = G.nodes[f"D_{t+1}_{j}"]['y']

                # calculate score, using the score function
                score = calculate_score(x1, y1, x2, y2, score_func)
                
                # check if both nodes are in the graph
                if f"D_{t}_{i}" in G.nodes() and f"D_{t+1}_{j}" in G.nodes():
                    # if detections come from a tracked file and are part of the same track, multiply score by coeff
                    if tracked and G.nodes[f"D_{t}_{i}"]['track_no'] == G.nodes[f"D_{t+1}_{j}"]['track_no']:
                        score = score*coeff
                        G.add_edge(f"D_{t}_{i}", f"D_{t+1}_{j}", weight = score)
                    else:
                        if score <= max_score:
                            G.add_edge(f"D_{t}_{i}", f"D_{t+1}_{j}", weight = score)

    G = add_stxy_nodes_and_edges(G, num_frames=len(points), max_score=max_score, nodes_per_frame=nodes_per_frame)
 
    return G


# adds the stxy-nodes and -edges
def add_stxy_nodes_and_edges(G, num_frames, max_score, nodes_per_frame):

    # add source and target node
    # these are also required for the tracked_graph, as that would make correction procedure easier to implement
    G.add_node('S', type='S')
    G.add_node('T', type='T')
   
    # Y nodes for when a tracks tarts later than 0th frame 
    # X nodes for when a track ends earlier than last frame
    for i in range(num_frames):
        if i < num_frames - 1:
            G.add_node(f'Y_{i}', type='Y')

        if i > 0:
            G.add_node(f'X_{i}', type='X')     

    # add edges between S,Y,X,T-nodes
    for i in range(num_frames-1):
        if i < num_frames - 2:
            G.add_edge(f'Y_{i}', f'Y_{i+1}', weight=max_score)

        if i > 0:
            G.add_edge(f'X_{i}', f'X_{i+1}', weight=max_score)
        

    G.add_edge('S', f'Y_{0}', weight=max_score)
    G.add_edge(f'X_{num_frames-1}', 'T', weight=max_score)

    for k in range(num_frames):
        if k==0:
            for _, node in enumerate(nodes_per_frame[0]):
                # adges to and from nodes in 0th frame
                G.add_edge('S', node, weight=max_score)
                G.add_edge(node,'X_1', weight=max_score)
        else:
            for _, node in enumerate(nodes_per_frame[k]):
                # add edges to nodes in all other frames
                G.add_edge(f'Y_{k-1}', node, weight=max_score)
                if k < num_frames-1:
                    G.add_edge(node, f'X_{k+1}', weight=max_score)
                else:
                    G.add_edge(node, 'T', weight=max_score)

    return G

# takes a networkx graph G as an input and generates a track using Dijkstra's shortest path
def generate_track(G):
    graph=G.copy()
    shortest_path = None
    try:
        shortest_path = nx.shortest_path(graph, 'S', 'T', 'weight')
    except nx.exception.NetworkXNoPath:
       pass
          
    if shortest_path != None:
        if all(graph.nodes[n]['type'] != 'D' for n in shortest_path[1:-1]):
            return None
        else:
            return shortest_path[1:-1]
    else:
        return None    

def generate_tracks(graph, detections, max_num=math.inf, debug = False):
    if graph == nx.null_graph:
        return []
    elif not('S' in graph.nodes()) or not('T' in graph.nodes()):    
        return []
    else:
        tracks = []
        for d in detections:
            tracks.append([None]*len(d))
        current_track = 1   
    # iterate while there are still D nodes in the list of nodes
    # check if the graph has D nodes - if not, return an empty list
        if len([n for n in graph.nodes.data('type') if n[1] == 'D']) == 0:
            return []
    # else generate shortest_paths until the shortest path doesn't contain any D nodes
        else:
            # initialize first shortest path (which is guaranteed to contain a D, as getting to a D node is less expensive than Y's and X's)
            while current_track <= max_num:
                path = generate_track(graph)
                if path is None:
                    break
                if debug:
                    print(path)
                for p in path:
                    if graph.nodes[p]['type'] == 'D':
                        t = graph.nodes[p]["time_point"]
                        i = graph.nodes[p]["idx"]
                        tracks[t][i] = current_track
                current_track = current_track+1
                graph.remove_nodes_from([n for n in path if graph.nodes[n]['type'] == 'D'])
            return tracks
        

def find_node_by_attributes(G, **attrs):
    """
    Find nodes in graph G with specific attribute values.

    Parameters:
    - G (networkx.Graph): The graph to search.
    - **attrs: Key-value pairs of attributes to match.

    Returns:
    - list: A list of nodes that match the given attributes.
    """
    matching_nodes = []
    for node, node_attrs in G.nodes(data=True):
        if all(node_attrs.get(attr) == value for attr, value in attrs.items()):
            matching_nodes.append(node)
    
    return matching_nodes

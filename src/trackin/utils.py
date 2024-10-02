import sys
import networkx as nx
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heapq
from datetime import datetime
from .tracking import generate_graph, generate_track, generate_tracks, remove_all_successors
from ._widget import csv_loaded_event, remove_track_node_event, accept_track_event, delete_segment_event, save_segment_event, delete_all_connections_event
from .shared_state import shared_state
from napari.utils.notifications import show_info
from napari.utils.events import EventEmitter
import os


'''def safeindex(l,i):
	try:
		return l.index(i)
	except ValueError: 
		return -1'''

def track_to_posarray( trackp ):
	data = shared_state.DATA
	shared_state.track = [-1] * len( data )
	for n in trackp:
		if shared_state.G.nodes[n]["type"]=="D":
			shared_state.track[shared_state.G.nodes[n]["time_point"]] = shared_state.G.nodes[n]["idx"]
	return shared_state.track

def send_track():
	# load up-to-date values of variables
	data = shared_state.DATA
	graph = shared_state.G
	shared_state.track = track_to_posarray( generate_track( graph ) )
	shared_state.track_dict = dict(track=shared_state.track,
		npos=shared_state.G.number_of_nodes()-2-2*(len(data)-1),
		nconn=shared_state.G.number_of_edges()-2*(len(data)))

#built_graph_event.connect(send_track)

'''def update_counts():
	return dict(npos=G.number_of_nodes()-2-2*(len(DATA)-1),
		nconn=G.number_of_edges()-2*(len(DATA)))'''

def accept_track():
    shared_state.N_TRACKS += 1
    track = shared_state.track
    track_with_nodes = []
    session_file_path = os.path.join(shared_state.csv_folder_to_save, shared_state.SESSION_FILE)
    
    # Determine if header should be added based on file existence and size
    add_header = not os.path.exists(session_file_path) or os.path.getsize(session_file_path) == 0
    
    with open(session_file_path, "a") as f:
        if add_header:
            f.write("tframe,y,x,displ_y,displ_x,track_no\n")  # Write the header row
        for i, p in enumerate(track):
            if p >= 0:
                f.write(f"{i},{shared_state.DATA[i][p][0]},{shared_state.DATA[i][p][1]}, {shared_state.DATA[i][p][2]},{shared_state.DATA[i][p][3]},{shared_state.N_TRACKS}\n")
                # Assign nonsense values so that it's clear that the points from the accepted track have been deleted
                shared_state.DATA[i][p] = (-1000000, -1000000)
                # remove nodes from track
                shared_state.G.remove_node(f'D_{i}_{p}')
                # add node names to list
                track_with_nodes.append(f'D_{i}_{p}')
    # this is used to remove the nodes used for the track
    print(f'Accepted track {shared_state.N_TRACKS}.')

    # Save the up-to-date version of DATA
    updated_data_file_path = os.path.join(shared_state.csv_folder_to_save, shared_state.UPDATED_DATA_FILE)
    with open(updated_data_file_path, "w") as g:
        g.write('tframe,y,x,displ_y,displ_x\n')
        for i, _ in enumerate(shared_state.DATA):
            for j, _ in enumerate(shared_state.DATA[i]):
                # Ensure that the detections included in the accepted track are removed from the saved version
                if shared_state.DATA[i][j][0] != -1000000 and shared_state.DATA[i][j][1] != -1000000:
                    g.write(f"{i},{shared_state.DATA[i][j][0]},{shared_state.DATA[i][j][1]},{shared_state.DATA[i][j][2]},{shared_state.DATA[i][j][3]}\n")
    
    return send_track()

accept_track_event.connect(accept_track)

# remove all the obsolete edges connected to nodes in a track segment
def save_segment():
    """Save the currently defined segment to the graph."""
    s = shared_state.track[0:shared_state.current_index]
    if len(s) > 1:
        for i in range(len(s) - 1):
            if s[i] == -1 or s[i + 1] == -1:
                continue

            # Define node names based on current segment positions
            n1 = f"D_{i}_{s[i]}"
            n2 = f"D_{i + 1}_{s[i + 1]}"

            #print(f"Processing nodes: {n1}, {n2}")  # Debug print statement
            #print(list(shared_state.G.successors(n1)))

            # Only remove D-D edges, not D-X edges
            nb = [(n1, ss) for ss in shared_state.G.successors(n1) if (shared_state.G.nodes[ss]["type"] in ["D"])]
            shared_state.G.remove_edges_from(nb)
            shared_state.G.add_edge(n1, n2, weight=0)

            print(f'Removing edges: {nb}.')
            print(f"Successor nodes of {n1}: {list(shared_state.G.successors(n1))}")

    return send_track()


save_segment_event.connect(save_segment)

def delete_segment():
    segment_to_delete = []
    updated_data_file_path = os.path.join(shared_state.csv_folder_to_save, shared_state.UPDATED_DATA_FILE)
    for i, p in enumerate(shared_state.track[0:shared_state.current_index+1]):
        if p >= 0:
            # Assign nonsense values so that it's clear that the points from the accepted track have been deleted
            shared_state.DATA[i][p] = (-1000000, -1000000)
            # remove nodes from track
            shared_state.G.remove_node(f'D_{i}_{p}')
            # add node names to list
            segment_to_delete.append(f'D_{i}_{p}')
    print(f'Deleted segment: {segment_to_delete}')

    with open(updated_data_file_path, "w") as g:
        g.write('tframe,y,x,displ_y,displ_x\n')
        for i, _ in enumerate(shared_state.DATA):
            for j, _ in enumerate(shared_state.DATA[i]):
                # Ensure that the detections included in the accepted track are removed from the saved version
                if shared_state.DATA[i][j][0] != -1000000 and shared_state.DATA[i][j][1] != -1000000:
                    g.write(f"{i},{shared_state.DATA[i][j][0]},{shared_state.DATA[i][j][1]},{shared_state.DATA[i][j][2]},{shared_state.DATA[i][j][3]}\n")
    
    return send_track()
     
delete_segment_event.connect(delete_segment)

# deletes all edges from a node, apart from an X- or T-edge, 
# as well as all obsolete edges to nodes in the segment up until the current detection
def delete_all_connections():
    t1 = shared_state.current_index
    i1 = shared_state.track[shared_state.current_index]
    node_name = f'D_{t1}_{i1}'

    # Remove all edges coming out of the node, except for the one leading to an X-node
    if t1 != len(shared_state.DATA)-1:
        remove_all_successors(shared_state.G, node_name, f'X_{t1+1}')
        print(f'Removed all edges from {node_name}, apart from X_{t1+1}.')
        save_segment()
    else:
        print('No obsolete edges to remove. This is the very last frame. No incorrect edges will be deleted between the nodes in segment up to that point.')

delete_all_connections_event.connect(delete_all_connections)

def build_graph():
    # ensure new values are loaded (values not updated by default)
    data = shared_state.DATA
    tracked = shared_state.TRACKED
    max_score = shared_state.MAX_SCORE
    score_func = shared_state.SCORE_FUNC

    print(f'value of shared.TRACKED in utils:{shared_state.TRACKED}')
    # Generate the graph using the updated global variables
    #G = generate_graph(data, max_score, score_func, tracked)
    shared_state.G = generate_graph(data, max_score, score_func, tracked)
    
def track():
    # If the graph isn't built yet, build it
    if shared_state.G is None:
        show_info("Building graph since it hasn't been created yet...")
        build_graph()

    # Ensure the graph was actually built
    send_track()
	
csv_loaded_event.connect(track)
remove_track_node_event.connect(track)

'''@app.route('/frame/<path:path>', methods=['GET'])
def frame(path):
    return send_from_directory( FRAMES_DIR, path )'''

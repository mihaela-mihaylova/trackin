import sys
import networkx as nx
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heapq
from datetime import datetime
from .tracking import generate_graph, generate_track, generate_tracks
from ._widget import data_updated_event, csv_loaded_event,  DATA, TRACKED, MAX_SCORE, SCORE_FUNC  # Import variables and event
from napari.utils.notifications import show_info


'''def safeindex(l,i):
	try:
		return l.index(i)
	except ValueError: 
		return -1

def track_to_posarray( trackp ):
	track = [-1] * len( DATA )
	for n in trackp:
		if G.nodes[n]["type"]=="D":
			track[G.nodes[n]["time_point"]] = G.nodes[n]["idx"]
	return track

def send_track():
	track = track_to_posarray( generate_track( G ) )
	return dict(track=track,
		npos=G.number_of_nodes()-2-2*(len(DATA)-1),
		nconn=G.number_of_edges()-2*(len(DATA)))

def update_counts():
	return dict(npos=G.number_of_nodes()-2-2*(len(DATA)-1),
		nconn=G.number_of_edges()-2*(len(DATA)))

@app.route('/accept-track/', methods=['POST'])
def accept_track():
	global N_TRACKS
	N_TRACKS += 1
	track = track_to_posarray( generate_track( G ) )
	f = open("session.csv", "a")
	for i,p in enumerate(track):
		if p >= 0:
			f.write(f"{i},{DATA[i][p][0]},{DATA[i][p][1]},{N_TRACKS}\n")
			# assign nonsense values so that it's clear that the points from the accepted track have been deleted
			DATA[i][p] = (-1000000, -1000000)
	f.close()
	# this is used to remove the nodes used for the track
	generate_tracks( G, DATA, max_num=1, debug=False )

	# save up-to-date version of DATA
	g = open(UPDATED_DATA_FILE, "w")
	g.write('tframe,y,x,displ_y,displ_x\n')
	for i,_ in enumerate(DATA):
		for j,_ in enumerate(DATA[i]):
			# ensure that the detections included in the accepted track are removed from saved version
			if DATA[i][j][0]!=-1000000 and DATA[i][j][1]!=-1000000:
				g.write(f"{i},{DATA[i][j][0]},{DATA[i][j][1]},{DATA[i][j][2]},{DATA[i][j][3]}\n")
	g.close()
	return send_track()

@app.route('/delete-detection/', methods=['POST'])
def delete_detection():
	content = request.json
	t = content["frame"]
	i = content["index"]
	n = f"D_{t}_{i}"
	print( f"deleting detection {n}")
	G.remove_node( n )
	# assign nonsense values so that it's clear that the point has been deleted
	DATA[t][i] = (-1000000, -1000000)
	print(DATA[t][i])
	# save up-to-date version of DATA
	f = open(UPDATED_DATA_FILE, "w")
	f.write('tframe,y,x,displ_y,displ_x\n')
	for i,_ in enumerate(DATA):
		for j,_ in enumerate(DATA[i]):
			# ensure that the detection that is deleted is not saved
			if (DATA[i][j][0]!=-1000000 and DATA[i][j][1]!=-1000000):
				f.write(f"{i},{DATA[i][j][0]},{DATA[i][j][1]},{DATA[i][j][2]},{DATA[i][j][3]}\n")
	f.close()

	return send_track()
	

@app.route('/click-delete-detection/', methods=['POST'])
def click_delete_detection():
	content = request.json
	t = content["frame"]
	idx_del = content["idx_del"]
	is_in_track = content["is_in_track"]
	n = f"D_{t}_{idx_del}"
	print( f"deleting detection {n}")
	print(DATA[t][idx_del])
	G.remove_node( n )
	# assign nonsense values so that it's clear that the point has been deleted
	DATA[t][idx_del] = (-1000000, -1000000)
	# save up-to-date version of DATA
	f = open(UPDATED_DATA_FILE, "w")
	f.write('tframe,y,x,displ_y,displ_x\n')
	for i,_ in enumerate(DATA):
		for j,_ in enumerate(DATA[i]):
			# ensure that the detection that is deleted is not saved
			if (DATA[i][j][0]!=-1000000 and DATA[i][j][1]!=-1000000):
				f.write(f"{i},{DATA[i][j][0]},{DATA[i][j][1]},{DATA[i][j][2]},{DATA[i][j][3]}\n")
	f.close()
    # check if the deleted node is in the current track - if not, then don't go back to the first frame
	if is_in_track:
		return send_track()
	else:
		return update_counts()


@app.route('/add-detection/', methods=['POST'])
def add_detection():
	content = request.json
	t = content["frame"]
	new_detection = content["detection"]
	idx_new_det = content["detectionindex"]
	n =  f"D_{t}_{idx_new_det}"
	print( f"adding detection D_{t}_{idx_new_det}")
	# displ_x and displ_y are assigned to 0 for convenience
	G.add_node(n, x = new_detection[1], y=new_detection[0], displ_x=0, displ_y=0, time_point = t, idx = idx_new_det, type = 'D')
	# add edges to the added node
	add_edge_to_and_from_nodes(n, G, DATA, MAX_SCORE)
	print(new_detection[1],new_detection[0])
	# add the new detection to the DATA subarray corresponding to the current frame
	DATA[t].append([new_detection[1], new_detection[0],0,0])
	# save up-to-date version of DATA
	f = open(UPDATED_DATA_FILE, "w")
	f.write('tframe,y,x,displ_y,displ_x\n')
	for i,_ in enumerate(DATA):
		for j,_ in enumerate(DATA[i]):
			if DATA[i][j][0]!= -1000000 and DATA[i][j][1]!=-1000000:
				f.write(f"{i},{DATA[i][j][0]},{DATA[i][j][1]},{DATA[i][j][2]},{DATA[i][j][3]}\n")
	f.close()
	return update_counts()

# push current segment(not the suggested one, accepted by pressing A, but the one with registered changes)
@app.route('/push-segment/', methods=['POST'])
def push_segment():
	global N_TRACKS
	N_TRACKS += 1
	content = request.json
	s = content["segment"]
	nodes_of_segment=[]
	print(s)
	if len(s)>1:
		for i in range(len(s)-1):
			if s[i] == -1 : continue
			if s[i+1] == -1 : continue
			n1 = f"D_{i}_{s[i]}"
			n2 = f"D_{i+1}_{s[i+1]}"
			# add node names to list
			if n1 not in nodes_of_segment:
				nodes_of_segment.append(n1)
			if n2 not in nodes_of_segment:
				nodes_of_segment.append(n2)					
			#print( f"{n1},{n2} before" )
			#print( list(G.successors( n1 )) )
			#nb = [(n1,ss) for ss in G.successors(n1) if (G.nodes[ss]["type"] in ["D","X"])]
			#G.remove_edges_from(nb)
			#G.add_edge(n1, n2, weight = 0)
			#print( "after" )
			#print( list(G.successors( n1 )) )
	print(nodes_of_segment)		
	track = track_to_posarray(nodes_of_segment)
	print(track)
	f = open("session.csv", "a")
	for i,p in enumerate(track):
		if p >= 0:
			f.write(f"{i},{DATA[i][p][0]},{DATA[i][p][1]},{N_TRACKS}\n")
			# assign nonsense values so that it's clear that the points from the accepted track have been deleted
			DATA[i][p] = (-1000000, -1000000)
	f.close()
	# save up-to-date version of DATA
	g = open(UPDATED_DATA_FILE, "w")
	g.write('tframe,y,x,displ_y,displ_x\n')
	for i,_ in enumerate(DATA):
		for j,_ in enumerate(DATA[i]):
			# ensure that the detections included in the accepted track are removed from saved version
			if DATA[i][j][0]!=-1000000 and DATA[i][j][1]!=-1000000:
				g.write(f"{i},{DATA[i][j][0]},{DATA[i][j][1]},{DATA[i][j][2]},{DATA[i][j][3]}\n")
	g.close()
	for node in nodes_of_segment:
		G.remove_node(node)
	return send_track()


@app.route('/delete-connection/', methods=['POST'])
def delete_connection():
	content = request.json
	t1 = content["frame1"]
	i1 = content["index1"]
	t2 = content["frame2"]
	i2 = content["index2"]
	n1 = f"D_{t1}_{i1}"
	n2 = f"D_{t2}_{i2}"
	print( f"deleting connection ({n1},{n2})")
	G.remove_edge( n1, n2 )
	return send_track()


@app.route('/delete-connection-prev-conns/', methods=['POST'])
def delete_connection_prev_conns():
	content = request.json
	t1 = content["frame1"]
	i1 = content["index1"]
	t2 = content["frame2"]
	i2 = content["index2"]
	n1 = f"D_{t1}_{i1}"
	n2 = f"D_{t2}_{i2}"
	print(n2)
	print(n2 in G.nodes())
	print(n2 in G.successors(n1))
	print((n1,n2) in G.edges())
	print( f"deleting connection ({n1},{n2})")
	G.remove_edge( n1, n2 )
	segment_list=[]
	s = content["segment"]
	if len(s)>1:
		for i in range(len(s)-1):
			if s[i] == -1 : continue
			if s[i+1] == -1 : continue
			n1 = f"D_{i}_{s[i]}"
			n2 = f"D_{i+1}_{s[i+1]}"
			# only remove D-D edges, not D-X edges (if D node accidentally deleted, then first part of track won't be suggested anymore)
			nb = [(n1,ss) for ss in G.successors(n1) if (G.nodes[ss]["type"] in ["D"])]
			G.remove_edges_from(nb)
			print(f'Remove edges from {n1}:{nb}')
			G.add_edge(n1, n2, weight = 0)
			segment_list.append(n1)
			last_element_idx = i		
		segment_list.append(f"D_{last_element_idx+1}_{s[last_element_idx+1]}")	
	else:
		segment_list=[]
	print(f'Correct segment saved: {segment_list}')
	return send_track()


@app.route('/delete-all-connections-from-node/', methods=['POST'])
def delete_all_connections():
	content = request.json
	t1 = content["frame1"]
	i1 = content["index1"]
	node_name = f'D_{t1}_{i1}'
	# remove all edges coming out of node, except for the one leading to an X-node
	print(f'Removed all edges from {node_name}, apart from ({node_name}, X_{t1+1}):')
	remove_edges_start_with(node_name, f'X_{t1+1}', G)

	return send_track()


@app.route('/accept-segment/', methods=['POST'])
def accept_segment():
	content = request.json
	s = content["segment"]
	if len(s)>1:
		for i in range(len(s)-1):
			if s[i] == -1 : continue
			if s[i+1] == -1 : continue
			n1 = f"D_{i}_{s[i]}"
			n2 = f"D_{i+1}_{s[i+1]}"
			print( f"{n1},{n2} before" )
			print( list(G.successors( n1 )) )
			# only remove D-D edges, not D-X edges (if D node accidentally deleted, then first part of track won't be suggested anymore)
			nb = [(n1,ss) for ss in G.successors(n1) if (G.nodes[ss]["type"] in ["D"])]
			G.remove_edges_from(nb)
			G.add_edge(n1, n2, weight = 0)
			print( "after" )
			print( list(G.successors( n1 )) )
	return send_track()

@app.route('/track/')
def track0():
	return redirect(url_for('track',i=0))'''


def build_graph():
    global DATA, TRACKED, MAX_SCORE, SCORE_FUNC  # Access global variables

    data = DATA
    print(type(data))
    tracked = TRACKED
    max_score = MAX_SCORE
    score_func = SCORE_FUNC

    # Generate the graph using the updated global variables
    G = generate_graph(data, max_score=max_score, score_func=score_func, tracked=tracked)
    print(f"Graph generated: {G.nodes}")

data_updated_event.connect(build_graph)
csv_loaded_event.connect(build_graph)


'''@app.route('/track/<int:i>')
def track(i=0):
	#if i > 0:
	#	generate_tracks(g, data, max_num=i, debug=False)
	#tracks = list(map( lambda l: safeindex(l,1), 
	#	generate_tracks(g, data, max_num=1, debug=False) ))
	if G is None or i>0:
		build_graph()
	generate_tracks( G, DATA, max_num=i, debug=False )
	return send_track()

@app.route('/frame/<path:path>', methods=['GET'])
def frame(path):
    return send_from_directory( FRAMES_DIR, path )'''

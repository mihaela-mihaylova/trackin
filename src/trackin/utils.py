import networkx as nx
import math
import numpy as np
import pandas as pd
from datetime import datetime
from .core.tracking import generate_graph, generate_track, generate_tracks, remove_all_successors, add_node_with_dummy_edges
from .core.detection import Detection
from .shared_state import shared_state, NO_VALUE
import os

# These functions don't show anything themselves: they return the messages
# for the user (e.g. 'No detections left.'), which the plugin's interface
# shows. So they don't need napari.


def track_to_posarray( trackp ):
    if trackp is None:
        shared_state.track = None
    else:
        data = shared_state.DATA
        shared_state.track = [-1] * len( data )
        for n in trackp:
            if shared_state.G.nodes[n]["type"]=="D":
                shared_state.track[shared_state.G.nodes[n]["time_point"]] = shared_state.G.nodes[n]["idx"]
        return shared_state.track

def send_track():
    """Propose the next track. Returns the messages for the user: a note
    when there is no track left."""
    # load up-to-date values of variables
    data = shared_state.DATA
    graph = shared_state.G
    shared_state.track = track_to_posarray( generate_track( graph ) )
    return ['No detections left.'] if shared_state.track is None else []

def accept_track(track):
    """Accept `track`: write it to the session file (and the combined file),
    and mark its detections as no longer available. Shift-Q."""
    # Also catches track == [] (e.g. right after images are reloaded, before
    # a new CSV is loaded) -- not just None -- so this doesn't silently open
    # the *previous* dataset's SESSION_FILE and write a header into it.
    if not track:
        return ['No detections left.']
    shared_state.N_TRACKS += 1
    track_with_nodes = []
    session_file_path = os.path.join(shared_state.csv_folder_to_save, shared_state.SESSION_FILE)
    if shared_state.MAX_TRACK_ID is not None:
        upd_tracks_file_path = os.path.join(shared_state.csv_folder_to_save, shared_state.UPD_TRACK_FILE)
    
    # Determine if header should be added based on file existence and size
    add_header = not os.path.exists(session_file_path) or os.path.getsize(session_file_path) == 0
    
    # add to file with both old and new tracks if a track file is loaded
    if shared_state.MAX_TRACK_ID is not None:
        with open(upd_tracks_file_path, "a") as f:
            updated_track_id = shared_state.MAX_TRACK_ID + shared_state.N_TRACKS
            for i, p in enumerate(track):
                if p >= 0:
                    d = shared_state.DATA[i][p]
                    f.write(f"{i},{d.y},{d.x},{d.displ_y},{d.displ_x},{updated_track_id}{shared_state.combined_extra_values(i, p)}\n")

    with open(session_file_path, "a") as f:
        if add_header:
            f.write(f"tframe,y,x,displ_y,displ_x,track_no{shared_state.extra_header()}\n")  # Write the header row
        for i, p in enumerate(track):
            if p >= 0:
                d = shared_state.DATA[i][p]
                f.write(f"{i},{d.y},{d.x},{d.displ_y},{d.displ_x},{shared_state.N_TRACKS}{shared_state.extra_values(i, p)}\n")
                # the points of the accepted track are no longer available
                deactivate(i, p)
                # remove nodes from track
                shared_state.G.remove_node(f'D_{i}_{p}')
                # add node names to list
                track_with_nodes.append(f'D_{i}_{p}')
   
    # Save the up-to-date version of DATA
    updated_data_file_path = os.path.join(shared_state.csv_folder_to_save, shared_state.UPDATED_DATA_FILE)
    with open(updated_data_file_path, "w") as g:
        g.write(f'tframe,y,x,displ_y,displ_x{shared_state.extra_header()}\n')
        for i, _ in enumerate(shared_state.DATA):
            for j, _ in enumerate(shared_state.DATA[i]):
                # Ensure that the detections included in the accepted track are removed from the saved version
                if shared_state.DATA[i][j].active:
                    d = shared_state.DATA[i][j]
                    g.write(f"{i},{d.y},{d.x},{d.displ_y},{d.displ_x}{shared_state.extra_values(i, j)}\n")


    # this is used to remove the nodes used for the track
    return ['Accepted track.'] + send_track()

# remove all the obsolete edges connected to nodes in a track segment
def save_segment(track, upto):
    """Fix the links of `track` from frame 0 up to frame `upto`: each of its
    detections may then only continue to the next one in the track. W."""
    # e.g. right after images are reloaded, before a new CSV is loaded --
    # without this, the segment-building part below is a harmless no-op
    # (empty slice), but the send_track() call at the end still runs and
    # crashes on shared_state.G being None.
    if not track:
        return ['No detections left.']
    s = track[0:upto+1]
    if len(s) > 1:
        for i in range(len(s) - 1):
            if s[i] == -1 or s[i + 1] == -1:
                continue

            # Define node names based on current segment positions
            n1 = f"D_{i}_{s[i]}"
            n2 = f"D_{i + 1}_{s[i + 1]}"

            # Only remove D-D edges, not D-X edges
            nb = [(n1, ss) for ss in shared_state.G.successors(n1) if (shared_state.G.nodes[ss]["type"] in ["D"])]
            shared_state.G.remove_edges_from(nb)
            shared_state.G.add_edge(n1, n2, weight=0)

            print(f'Removing edges: {nb}.')
            print(f"Successor nodes of {n1}: {list(shared_state.G.successors(n1))}")

    return send_track()


def save_segment_without_update(track, upto):
    """Like save_segment, but without proposing a new track."""
    s = track[0:upto+1]
    if len(s) > 1:
        for i in range(len(s) - 1):
            if s[i] == -1 or s[i + 1] == -1:
                continue

            # Define node names based on current segment positions
            n1 = f"D_{i}_{s[i]}"
            n2 = f"D_{i + 1}_{s[i + 1]}"

            # Only remove D-D edges, not D-X edges
            nb = [(n1, ss) for ss in shared_state.G.successors(n1) if (shared_state.G.nodes[ss]["type"] in ["D"])]
            shared_state.G.remove_edges_from(nb)
            shared_state.G.add_edge(n1, n2, weight=0)

            print(f'Removing edges: {nb}.')
            print(f"Successor nodes of {n1}: {list(shared_state.G.successors(n1))}")


def delete_segment(track, upto):
    """Delete the detections of `track` from frame 0 up to frame `upto`. Shift-Z."""
    segment_to_delete = []
    updated_data_file_path = os.path.join(shared_state.csv_folder_to_save, shared_state.UPDATED_DATA_FILE)
    # No track: e.g. right after images are reloaded, before a new CSV is
    # loaded. Stopping here keeps the previous dataset's leftover file intact.
    if not track:
        return ['No detections left.']
    for i, p in enumerate(track[0:upto+1]):
        if p >= 0:
            # the points of the deleted segment are no longer available
            deactivate(i, p)
            # remove nodes from track
            shared_state.G.remove_node(f'D_{i}_{p}')
            # add node names to list
            segment_to_delete.append(f'D_{i}_{p}')
    print(f'Deleted segment: {segment_to_delete}')

    with open(updated_data_file_path, "w") as g:
        g.write(f'tframe,y,x,displ_y,displ_x{shared_state.extra_header()}\n')
        for i, _ in enumerate(shared_state.DATA):
            for j, _ in enumerate(shared_state.DATA[i]):
                # Ensure that the detections included in the accepted track are removed from the saved version
                if shared_state.DATA[i][j].active:
                    d = shared_state.DATA[i][j]
                    g.write(f"{i},{d.y},{d.x},{d.displ_y},{d.displ_x}{shared_state.extra_values(i, j)}\n")

    return send_track()

# deletes all edges from a node, apart from an X- or T-edge, 
# as well as all obsolete edges to nodes in the segment up until the current detection
def delete_all_connections(track, t):
    """Remove all links out of `track`'s detection in frame `t` (apart from
    ending the track there), and end `track` after frame t. X."""
    # e.g. right after images are reloaded, before a new CSV is loaded
    if not track:
        return ['No detections left.']
    t1 = t
    i1 = track[t]
    node_name = f'D_{t1}_{i1}'

    # Remove all edges coming out of the node, except for the one leading to an X-node
    if t1 != len(shared_state.DATA)-1:
        remove_all_successors(shared_state.G, node_name, f'X_{t1+1}')
        print(f'Removed all edges from {node_name}, apart from X_{t1+1}.')
        # assign -1 to all consecutive positions in the track, because all further edges become irrelevant
        for i in range(t1+1,len(track)):
             track[i] = -1
        save_segment_without_update(track, t)
    else:
        print('No obsolete edges to remove. This is the very last frame. No incorrect edges will be deleted between the nodes in segment up to that point.')
    return []

def build_graph():
    # ensure new values are loaded (values not updated by default)
    data = shared_state.DATA
    tracked = shared_state.TRACKED
    max_score = shared_state.MAX_SCORE
    score_func = shared_state.SCORE_FUNC

    print(f'value of shared.TRACKED in utils:{shared_state.TRACKED}')
    # Generate the graph using the updated global variables
    shared_state.G = generate_graph(data, max_score, score_func, tracked)
    
def track():
    # If the graph isn't built yet, build it
    if shared_state.G is None:
        build_graph()

    # Ensure the graph was actually built
    return send_track()

def stats():
    """The number of detections and connections in the graph, as shown in
    the panel ("Detections: …", "Connections: …"). The graph's start, end
    and per-frame helper nodes, and their links, are not counted."""
    n_frames = len(shared_state.DATA)
    return {
        "detections": shared_state.G.number_of_nodes() - 2 - 2 * (n_frames - 1),
        "connections": shared_state.G.number_of_edges() - 2 * (n_frames - 1),
    }


def deactivate(frame, index):
    """Mark detection `index` in `frame` as no longer available: deleted,
    or used in an accepted track. Its position is kept."""
    shared_state.DATA[frame][index] = shared_state.DATA[frame][index]._replace(active=False)


def delete_detection(frame, index):
    """Delete detection `index` in `frame`: mark it as deleted in DATA and
    remove it from the graph. Used by the D key and by right-click."""
    deactivate(frame, index)
    shared_state.G.remove_node(f'D_{frame}_{index}')


def add_detection(frame, y, x):
    """Add a detection by hand at (y, x) in `frame`, to DATA and to the
    graph. It gets NO_VALUE in the extra columns, as it didn't come from
    the detections file. Returns the new detection's index in the frame."""
    index = shared_state.NUM_DET_PER_FRAME[frame]
    shared_state.DATA[frame].append(Detection(y, x))
    shared_state.EXTRA[frame].append(tuple([NO_VALUE] * len(shared_state.EXTRA_COLUMNS)))
    add_node_with_dummy_edges(node=f'D_{frame}_{index}', time_point=frame, idx=index,
                              y=y, x=x, displ_y=0, displ_x=0, G=shared_state.G,
                              highest_frame_id=len(shared_state.DATA)-1,
                              max_score=shared_state.MAX_SCORE)
    print(f'Added node D_{frame}_{index}.')
    shared_state.NUM_DET_PER_FRAME[frame] += 1
    return index

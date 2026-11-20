"""A tracking session: the detections of one dataset, the graph built from
them, and the actions that edit it (accept a track, save or delete a
segment, delete or add a detection). Uses neither napari nor Qt."""

import math
import os

from .detection import Detection
from .tracking import generate_graph, generate_track, remove_all_successors, add_node_with_dummy_edges

# value written in the extra columns for a detection added by hand, which
# didn't come from the detections file and so has no values of its own there
NO_VALUE = -1


class Session:
    """The actions don't show anything themselves: they return the messages
    for the user (e.g. 'No detections left.'), which the plugin shows."""

    def __init__(self):
        self.DATA = []
        # columns of the detections file that the tracking doesn't use (e.g.
        # frame, label, area), in their original order; they are written back
        # out after the standard columns in every output file
        self.EXTRA_COLUMNS = []
        # extra columns of the combined track file written by Add Track File:
        # those of the loaded track file and of the detections file
        self.COMBINED_EXTRA_COLUMNS = []
        self.TRACKED = False
        self.G = None
        # the proposed track: the detection's index in each frame, or -1
        self.track = []
        self.MAX_SCORE = 1600
        self.SCORE_FUNC = 'squared'
        # the accepted tracks, in the order they were accepted: track number
        # k (in the session file) is accepted[k-1]. Each is the detection's
        # index in each frame, or -1; the detections themselves stay in DATA
        # (inactive), with their positions and extra values.
        self.accepted = []
        self.MAX_TRACK_ID = None
        self.csv_folder_to_save = ''
        self.SESSION_FILE = ''
        self.UPDATED_DATA_FILE = ''
        self.UPD_TRACK_FILE = ''

    def extra_header(self):
        """The extra columns' names for a CSV header row, each preceded by a
        comma, or "" if there are none."""
        return "".join("," + csv_value(c) for c in self.EXTRA_COLUMNS)

    def extra_values(self, t, i):
        """The extra columns' values of detection i in frame t for a CSV row,
        each preceded by a comma, or "" if there are none."""
        if not self.EXTRA_COLUMNS:
            return ""
        return "".join("," + csv_value(v) for v in self.DATA[t][i].extra)

    def combined_extra_values(self, t, i):
        """Like extra_values, but in the order of the combined track file's
        extra columns. A column only the loaded track file has gets NO_VALUE,
        as this detection has no value there."""
        values = dict(zip(self.EXTRA_COLUMNS, self.DATA[t][i].extra))
        return "".join("," + csv_value(values.get(c, NO_VALUE)) for c in self.COMBINED_EXTRA_COLUMNS)

    def track_to_posarray(self, trackp ):
        if trackp is None:
            self.track = None
        else:
            data = self.DATA
            self.track = [-1] * len( data )
            for n in trackp:
                if self.G.nodes[n]["type"]=="D":
                    self.track[self.G.nodes[n]["time_point"]] = self.G.nodes[n]["idx"]
            return self.track

    def send_track(self):
        """Propose the next track. Returns the messages for the user: a note
        when there is no track left."""
        # load up-to-date values of variables
        data = self.DATA
        graph = self.G
        self.track = self.track_to_posarray( generate_track( graph ) )
        return ['No detections left.'] if self.track is None else []

    def accept_track(self, track):
        """Accept `track`: write it to the session file (and the combined file),
        and mark its detections as no longer available. Shift-Q."""
        # Also catches track == [] (e.g. right after images are reloaded, before
        # a new CSV is loaded) -- not just None -- so this doesn't silently open
        # the *previous* dataset's SESSION_FILE and write a header into it.
        if not track:
            return ['No detections left.']
        self.accepted.append(list(track))
        track_no = len(self.accepted)
        track_with_nodes = []
        session_file_path = os.path.join(self.csv_folder_to_save, self.SESSION_FILE)
        if self.MAX_TRACK_ID is not None:
            upd_tracks_file_path = os.path.join(self.csv_folder_to_save, self.UPD_TRACK_FILE)

        # Determine if header should be added based on file existence and size
        add_header = not os.path.exists(session_file_path) or os.path.getsize(session_file_path) == 0

        # add to file with both old and new tracks if a track file is loaded
        if self.MAX_TRACK_ID is not None:
            with open(upd_tracks_file_path, "a") as f:
                updated_track_id = self.MAX_TRACK_ID + track_no
                for i, p in enumerate(track):
                    if p >= 0:
                        d = self.DATA[i][p]
                        f.write(f"{i},{d.y},{d.x},{d.displ_y},{d.displ_x},{updated_track_id}{self.combined_extra_values(i, p)}\n")

        with open(session_file_path, "a") as f:
            if add_header:
                f.write(f"tframe,y,x,displ_y,displ_x,track_no{self.extra_header()}\n")  # Write the header row
            for i, p in enumerate(track):
                if p >= 0:
                    d = self.DATA[i][p]
                    f.write(f"{i},{d.y},{d.x},{d.displ_y},{d.displ_x},{track_no}{self.extra_values(i, p)}\n")
                    # the points of the accepted track are no longer available
                    self.deactivate(i, p)
                    # remove nodes from track
                    self.G.remove_node(f'D_{i}_{p}')
                    # add node names to list
                    track_with_nodes.append(f'D_{i}_{p}')

        # Save the up-to-date version of DATA
        updated_data_file_path = os.path.join(self.csv_folder_to_save, self.UPDATED_DATA_FILE)
        with open(updated_data_file_path, "w") as g:
            g.write(f'tframe,y,x,displ_y,displ_x{self.extra_header()}\n')
            for i, _ in enumerate(self.DATA):
                for j, _ in enumerate(self.DATA[i]):
                    # Ensure that the detections included in the accepted track are removed from the saved version
                    if self.DATA[i][j].active:
                        d = self.DATA[i][j]
                        g.write(f"{i},{d.y},{d.x},{d.displ_y},{d.displ_x}{self.extra_values(i, j)}\n")


        # this is used to remove the nodes used for the track
        return ['Accepted track.'] + self.send_track()

    # remove all the obsolete edges connected to nodes in a track segment
    def save_segment(self, track, upto):
        """Fix the links of `track` from frame 0 up to frame `upto`: each of its
        detections may then only continue to the next one in the track. W."""
        # e.g. right after images are reloaded, before a new CSV is loaded --
        # without this, the segment-building part below is a harmless no-op
        # (empty slice), but the self.send_track() call at the end still runs and
        # crashes on self.G being None.
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
                nb = [(n1, ss) for ss in self.G.successors(n1) if (self.G.nodes[ss]["type"] in ["D"])]
                self.G.remove_edges_from(nb)
                self.G.add_edge(n1, n2, weight=0)

                print(f'Removing edges: {nb}.')
                print(f"Successor nodes of {n1}: {list(self.G.successors(n1))}")

        return self.send_track()


    def save_segment_without_update(self, track, upto):
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
                nb = [(n1, ss) for ss in self.G.successors(n1) if (self.G.nodes[ss]["type"] in ["D"])]
                self.G.remove_edges_from(nb)
                self.G.add_edge(n1, n2, weight=0)

                print(f'Removing edges: {nb}.')
                print(f"Successor nodes of {n1}: {list(self.G.successors(n1))}")


    def delete_segment(self, track, upto):
        """Delete the detections of `track` from frame 0 up to frame `upto`. Shift-Z."""
        segment_to_delete = []
        updated_data_file_path = os.path.join(self.csv_folder_to_save, self.UPDATED_DATA_FILE)
        # No track: e.g. right after images are reloaded, before a new CSV is
        # loaded. Stopping here keeps the previous dataset's leftover file intact.
        if not track:
            return ['No detections left.']
        for i, p in enumerate(track[0:upto+1]):
            if p >= 0:
                # the points of the deleted segment are no longer available
                self.deactivate(i, p)
                # remove nodes from track
                self.G.remove_node(f'D_{i}_{p}')
                # add node names to list
                segment_to_delete.append(f'D_{i}_{p}')
        print(f'Deleted segment: {segment_to_delete}')

        with open(updated_data_file_path, "w") as g:
            g.write(f'tframe,y,x,displ_y,displ_x{self.extra_header()}\n')
            for i, _ in enumerate(self.DATA):
                for j, _ in enumerate(self.DATA[i]):
                    # Ensure that the detections included in the accepted track are removed from the saved version
                    if self.DATA[i][j].active:
                        d = self.DATA[i][j]
                        g.write(f"{i},{d.y},{d.x},{d.displ_y},{d.displ_x}{self.extra_values(i, j)}\n")

        return self.send_track()

    # deletes all edges from a node, apart from an X- or T-edge, 
    # as well as all obsolete edges to nodes in the segment up until the current detection
    def delete_all_connections(self, track, t):
        """Remove all links out of `track`'s detection in frame `t` (apart from
        ending the track there), and end `track` after frame t. X."""
        # e.g. right after images are reloaded, before a new CSV is loaded
        if not track:
            return ['No detections left.']
        t1 = t
        i1 = track[t]
        node_name = f'D_{t1}_{i1}'

        # Remove all edges coming out of the node, except for the one leading to an X-node
        if t1 != len(self.DATA)-1:
            remove_all_successors(self.G, node_name, f'X_{t1+1}')
            print(f'Removed all edges from {node_name}, apart from X_{t1+1}.')
            # assign -1 to all consecutive positions in the track, because all further edges become irrelevant
            for i in range(t1+1,len(track)):
                 track[i] = -1
            self.save_segment_without_update(track, t)
        else:
            print('No obsolete edges to remove. This is the very last frame. No incorrect edges will be deleted between the nodes in segment up to that point.')
        return []

    def build_graph(self):
        # ensure new values are loaded (values not updated by default)
        data = self.DATA
        tracked = self.TRACKED
        max_score = self.MAX_SCORE
        score_func = self.SCORE_FUNC

        print(f'value of TRACKED in session:{self.TRACKED}')
        # Generate the graph using the updated global variables
        self.G = generate_graph(data, max_score, score_func, tracked)

    def propose(self):
        # If the graph isn't built yet, build it
        if self.G is None:
            self.build_graph()

        # Ensure the graph was actually built
        return self.send_track()

    def stats(self):
        """The number of detections and connections in the graph, as shown in
        the panel ("Detections: …", "Connections: …"). The graph's start, end
        and per-frame helper nodes, and their links, are not counted."""
        n_frames = len(self.DATA)
        return {
            "detections": self.G.number_of_nodes() - 2 - 2 * (n_frames - 1),
            "connections": self.G.number_of_edges() - 2 * (n_frames - 1),
        }


    def deactivate(self, frame, index):
        """Mark detection `index` in `frame` as no longer available: deleted,
        or used in an accepted track. Its position is kept."""
        self.DATA[frame][index] = self.DATA[frame][index]._replace(active=False)


    def delete_detection(self, frame, index):
        """Delete detection `index` in `frame`: mark it as deleted in DATA and
        remove it from the graph. Used by the D key and by right-click."""
        self.deactivate(frame, index)
        self.G.remove_node(f'D_{frame}_{index}')


    def add_detection(self, frame, y, x):
        """Add a detection by hand at (y, x) in `frame`, to DATA and to the
        graph. It gets NO_VALUE in the extra columns, as it didn't come from
        the detections file. Returns the new detection's index in the frame."""
        # numbered after all detections the frame has had, deleted ones
        # included, as those stay in the list
        index = len(self.DATA[frame])
        self.DATA[frame].append(
            Detection(y, x, extra=(NO_VALUE,) * len(self.EXTRA_COLUMNS), added=True)
        )
        add_node_with_dummy_edges(node=f'D_{frame}_{index}', time_point=frame, idx=index,
                                  y=y, x=x, displ_y=0, displ_x=0, G=self.G,
                                  highest_frame_id=len(self.DATA)-1,
                                  max_score=self.MAX_SCORE)
        print(f'Added node D_{frame}_{index}.')
        return index


def csv_value(value):
    """A value as written in a CSV cell: empty for a missing value, and in
    quotes if it contains a comma, quote or line break, so it stays one cell."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    if any(char in text for char in ',"\n'):
        text = '"' + text.replace('"', '""') + '"'
    return text

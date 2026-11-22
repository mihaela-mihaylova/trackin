"""A tracking session: the detections of one dataset, the graph built from
them, and the actions that edit it (accept a track, save or delete a
segment, delete or add a detection). Uses neither napari nor Qt."""

import math

from .detection import Detection
from .io import NO_VALUE, TRACKING_COLUMNS
from .tracking import generate_graph, generate_track, remove_all_successors, add_node_with_dummy_edges



class Session:
    """The actions don't show anything themselves: they return the messages
    for the user (e.g. 'No detections left.'), which the plugin shows."""

    def __init__(self):
        self.detections = []
        # columns of the detections file that the tracking doesn't use (e.g.
        # frame, label, area), in their original order; they are written back
        # out after the standard columns in every output file
        self.extra_columns = []
        self.has_prior_tracks = False
        self.graph = None
        # the proposed track: the detection's index in each frame, or -1
        self.proposal = []
        self.max_score = 1600
        self.score_func = 'squared'
        # the accepted tracks, in the order they were accepted: track number
        # k (in the session file) is accepted[k-1]. Each is the detection's
        # index in each frame, or -1; the detections themselves stay in self.detections
        # (inactive), with their positions and extra values.
        self.accepted = []

    @classmethod
    def from_dataframe(cls, df, n_frames=0):
        """A session with the detections of a detections table: columns
        tframe, y, x, displ_y and displ_x, optionally track_no (tracks from an
        earlier run), and any extra columns, kept in each detection's `extra`.

        Frames are indexed by their frame number (tframe), so a frame without
        detections stays empty instead of shifting all later frames onto the
        wrong images. There are at least n_frames frames (the number of
        loaded images), so frames after the last detection exist too."""
        session = cls()
        session.has_prior_tracks = 'track_no' in df.columns
        session.extra_columns = [c for c in df.columns if c not in TRACKING_COLUMNS]
        num_frames = max(n_frames, int(df['tframe'].max()) + 1 if len(df) else 0)
        session.detections = [[] for _ in range(num_frames)]
        # Read column by column rather than row by row, so each column keeps its
        # own type: a row mixes whole numbers with e.g. text or decimals in the
        # extra columns, and would otherwise turn all of them into one type
        columns = ['y', 'x', 'displ_y', 'displ_x'] + (['track_no'] if session.has_prior_tracks else [])
        positions = zip(*(df[c].tolist() for c in columns))
        extras = zip(*(df[c].tolist() for c in session.extra_columns)) if session.extra_columns else ([()] * len(df))
        for t, position, extra in zip(df['tframe'].tolist(), positions, extras):
            session.detections[int(t)].append(Detection(*position, extra=tuple(extra)))
        return session

    def path_to_proposal(self, trackp ):
        if trackp is None:
            self.proposal = None
        else:
            data = self.detections
            self.proposal = [-1] * len( data )
            for n in trackp:
                if self.graph.nodes[n]["type"]=="D":
                    self.proposal[self.graph.nodes[n]["time_point"]] = self.graph.nodes[n]["idx"]
            return self.proposal

    def propose_next(self):
        """Propose the next track. Returns the messages for the user: a note
        when there is no track left."""
        # load up-to-date values of variables
        data = self.detections
        graph = self.graph
        self.proposal = self.path_to_proposal( generate_track( graph ) )
        return ['No detections left.'] if self.proposal is None else []

    def accept_track(self, track):
        """Accept `track`: record it in `accepted`, and mark its detections
        as no longer available. Shift-Q. Writing it to the session file
        (see core/io.py) is up to the caller."""
        # Also catches track == [] (e.g. right after images are reloaded, before
        # a new CSV is loaded) -- not just None
        if not track:
            return ['No detections left.']
        self.accepted.append(list(track))
        track_with_nodes = []
        for i, p in enumerate(track):
            if p >= 0:
                # the points of the accepted track are no longer available
                self.deactivate(i, p)
                # remove nodes from track
                self.graph.remove_node(f'D_{i}_{p}')
                # add node names to list
                track_with_nodes.append(f'D_{i}_{p}')

        return ['Accepted track.'] + self.propose_next()

    # remove all the obsolete edges connected to nodes in a track segment
    def save_segment(self, track, upto):
        """Fix the links of `track` from frame 0 up to frame `upto`: each of its
        detections may then only continue to the next one in the track. W."""
        # e.g. right after images are reloaded, before a new CSV is loaded --
        # without this, the segment-building part below is a harmless no-op
        # (empty slice), but the self.propose_next() call at the end still runs and
        # crashes on self.graph being None.
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
                nb = [(n1, ss) for ss in self.graph.successors(n1) if (self.graph.nodes[ss]["type"] in ["D"])]
                self.graph.remove_edges_from(nb)
                self.graph.add_edge(n1, n2, weight=0)

                print(f'Removing edges: {nb}.')
                print(f"Successor nodes of {n1}: {list(self.graph.successors(n1))}")

        return self.propose_next()


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
                nb = [(n1, ss) for ss in self.graph.successors(n1) if (self.graph.nodes[ss]["type"] in ["D"])]
                self.graph.remove_edges_from(nb)
                self.graph.add_edge(n1, n2, weight=0)

                print(f'Removing edges: {nb}.')
                print(f"Successor nodes of {n1}: {list(self.graph.successors(n1))}")


    def delete_segment(self, track, upto):
        """Delete the detections of `track` from frame 0 up to frame `upto`. Shift-Z."""
        segment_to_delete = []
        # No track: e.g. right after images are reloaded, before a new CSV is
        # loaded. There is nothing to delete then.
        if not track:
            return ['No detections left.']
        for i, p in enumerate(track[0:upto+1]):
            if p >= 0:
                # the points of the deleted segment are no longer available
                self.deactivate(i, p)
                # remove nodes from track
                self.graph.remove_node(f'D_{i}_{p}')
                # add node names to list
                segment_to_delete.append(f'D_{i}_{p}')
        print(f'Deleted segment: {segment_to_delete}')

        return self.propose_next()

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
        if t1 != len(self.detections)-1:
            remove_all_successors(self.graph, node_name, f'X_{t1+1}')
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
        data = self.detections
        tracked = self.has_prior_tracks
        max_score = self.max_score
        score_func = self.score_func

        print(f'has_prior_tracks: {self.has_prior_tracks}')
        # Generate the graph using the updated global variables
        self.graph = generate_graph(data, max_score, score_func, tracked)

    def propose(self):
        # If the graph isn't built yet, build it
        if self.graph is None:
            self.build_graph()

        # Ensure the graph was actually built
        return self.propose_next()

    def stats(self):
        """The number of detections and connections in the graph, as shown in
        the panel ("Detections: …", "Connections: …"). The graph's start, end
        and per-frame helper nodes, and their links, are not counted."""
        n_frames = len(self.detections)
        return {
            "detections": self.graph.number_of_nodes() - 2 - 2 * (n_frames - 1),
            "connections": self.graph.number_of_edges() - 2 * (n_frames - 1),
        }


    def deactivate(self, frame, index):
        """Mark detection `index` in `frame` as no longer available: deleted,
        or used in an accepted track. Its position is kept."""
        self.detections[frame][index] = self.detections[frame][index]._replace(active=False)


    def delete_detection(self, frame, index):
        """Delete detection `index` in `frame`: mark it as deleted in detections and
        remove it from the graph. Used by the D key and by right-click."""
        self.deactivate(frame, index)
        self.graph.remove_node(f'D_{frame}_{index}')


    def add_detection(self, frame, y, x):
        """Add a detection by hand at (y, x) in `frame`, to detections and to the
        graph. It gets NO_VALUE in the extra columns, as it didn't come from
        the detections file. Returns the new detection's index in the frame."""
        # numbered after all detections the frame has had, deleted ones
        # included, as those stay in the list
        index = len(self.detections[frame])
        self.detections[frame].append(
            Detection(y, x, extra=(NO_VALUE,) * len(self.extra_columns), added=True)
        )
        add_node_with_dummy_edges(node=f'D_{frame}_{index}', time_point=frame, idx=index,
                                  y=y, x=x, displ_y=0, displ_x=0, G=self.graph,
                                  highest_frame_id=len(self.detections)-1,
                                  max_score=self.max_score)
        print(f'Added node D_{frame}_{index}.')
        return index

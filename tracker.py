import numpy as np
from scipy.optimize import linear_sum_assignment

def bb_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class Track:
    def __init__(self, box, conf, track_id, frame_id, max_age=8):
        self.id = track_id
        self.box = [float(b) for b in box] # Keep box coordinates as floats
        self.conf = conf
        self.age = 0         # Frames since last matched detection
        self.max_age = max_age
        self.hits = 1
        self.velocity = [0.0, 0.0, 0.0, 0.0] # Pixels per video frame
        self.last_frame_id = frame_id
        self.last_matched_box = [float(b) for b in box]
        self.last_matched_frame_id = frame_id
        
        # Attributes for persistent armed state
        self.weapon_counter = 0.0
        self.is_armed = False

    def predict(self, frame_id):
        elapsed = frame_id - self.last_frame_id
        if elapsed > 0:
            # Apply velocity to project the box forward
            self.box = [b + v * elapsed for b, v in zip(self.box, self.velocity)]
            # Decay the velocity during occlusion so the box slows down and stops moving blindly
            decay = 0.85 ** elapsed
            self.velocity = [v * decay for v in self.velocity]
            self.last_frame_id = frame_id
        self.age += 1

    def update(self, new_box, conf, frame_id):
        elapsed = frame_id - self.last_matched_frame_id
        if elapsed <= 0:
            elapsed = 1
            
        # Calculate velocity in pixels per video frame using actual detections to avoid feedback loop
        new_vel = [(n - o) / elapsed for n, o in zip(new_box, self.last_matched_box)]
        
        alpha_vel = 0.35 # Smoother EMA for velocity to prevent jitter/overshoot
        self.velocity = [alpha_vel * nv + (1 - alpha_vel) * v for nv, v in zip(new_vel, self.velocity)]
        
        # Calculate center velocity to adjust smoothing dynamically
        vc_x = (self.velocity[0] + self.velocity[2]) / 2.0
        vc_y = (self.velocity[1] + self.velocity[3]) / 2.0
        speed = np.sqrt(vc_x**2 + vc_y**2)
        
        # Dynamic alpha: smaller alpha (0.25) at low speeds to prevent jitter/wobble,
        # larger alpha (0.7) at high speeds to keep up without lagging
        alpha = 0.25 + 0.45 * min(1.0, speed / 6.0)
        
        # Smooth box update
        self.box = [alpha * n + (1 - alpha) * o for n, o in zip(new_box, self.box)]
        self.last_matched_box = [float(b) for b in new_box]
        self.last_matched_frame_id = frame_id
        
        self.conf = conf
        self.age = 0
        self.hits += 1
        self.last_frame_id = frame_id

class TrackManager:
    def __init__(self, iou_threshold=0.35, max_age=8):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1

    def update(self, detections, frame_id):
        """
        detections: list of dicts {"box": [x1,y1,x2,y2], "conf": float}
        Returns active tracks formatted as Track objects
        """
        # 1. Predict all existing tracks using their motion model
        for track in self.tracks:
            track.predict(frame_id)

        # 2. Match detections with existing tracks using the Hungarian Algorithm
        if len(self.tracks) > 0 and len(detections) > 0:
            # Build cost matrix
            cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                track_w = track.box[2] - track.box[0]
                track_h = track.box[3] - track.box[1]
                track_diag = np.sqrt(track_w**2 + track_h**2)
                
                track_cx = (track.box[0] + track.box[2]) / 2.0
                track_cy = (track.box[1] + track.box[3]) / 2.0
                
                track_vc_x = (track.velocity[0] + track.velocity[2]) / 2.0
                track_vc_y = (track.velocity[1] + track.velocity[3]) / 2.0
                
                for d_idx, det in enumerate(detections):
                    det_box = det["box"]
                    det_w = det_box[2] - det_box[0]
                    det_h = det_box[3] - det_box[1]
                    det_cx = (det_box[0] + det_box[2]) / 2.0
                    det_cy = (det_box[1] + det_box[3]) / 2.0
                    
                    iou = bb_iou(track.box, det_box)
                    
                    # Centroid distance cost
                    dx = det_cx - track_cx
                    dy = det_cy - track_cy
                    dist = np.sqrt(dx**2 + dy**2)
                    norm_dist = dist / max(1.0, track_diag)
                    
                    # Velocity direction consistency check (prevents swap between crossing targets)
                    elapsed = frame_id - track.last_matched_frame_id
                    if elapsed <= 0:
                        elapsed = 1
                        
                    implied_vc_x = (det_cx - ((track.last_matched_box[0] + track.last_matched_box[2]) / 2.0)) / elapsed
                    implied_vc_y = (det_cy - ((track.last_matched_box[1] + track.last_matched_box[3]) / 2.0)) / elapsed
                    
                    dir_penalty = 0.0
                    # Penalty for sudden high-speed direction reversal in X
                    if abs(track_vc_x) > 1.5 and (track_vc_x * implied_vc_x) < 0:
                        dir_penalty += 1.5 * min(3.0, abs(track_vc_x))
                    # Penalty for sudden high-speed direction reversal in Y
                    if abs(track_vc_y) > 1.5 and (track_vc_y * implied_vc_y) < 0:
                        dir_penalty += 1.5 * min(3.0, abs(track_vc_y))
                        
                    # Size change penalty (tall vs short, wide vs narrow)
                    size_change = abs(det_w - track_w) / max(1.0, track_w) + abs(det_h - track_h) / max(1.0, track_h)
                    
                    # Velocity change penalty (smooth acceleration check)
                    vel_change_x = implied_vc_x - track_vc_x
                    vel_change_y = implied_vc_y - track_vc_y
                    norm_vel_change = np.sqrt(vel_change_x**2 + vel_change_y**2) / max(1.0, track_h)
                    
                    # Total combined matching cost
                    cost = (1.0 - iou) + 0.4 * norm_dist + 0.3 * norm_vel_change + 0.2 * size_change + dir_penalty
                    
                    # Gating rules: physically impossible transitions
                    is_gated = False
                    if iou < 0.05 and norm_dist > 0.5:
                        is_gated = True
                    if size_change > 0.6: # Size changes by more than 60%
                        is_gated = True
                    if norm_vel_change > 0.7: # Acceleration is physically impossible
                        is_gated = True
                    if dir_penalty > 2.0: # Reversal of established velocity
                        is_gated = True
                        
                    if is_gated:
                        cost_matrix[t_idx, d_idx] = 10.0 # High cost prevents match
                    else:
                        cost_matrix[t_idx, d_idx] = cost

            # Solve linear sum assignment (Hungarian assignment)
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            matched_track_indices = set()
            matched_det_indices = set()
            
            for t_idx, d_idx in zip(track_indices, det_indices):
                cost = cost_matrix[t_idx, d_idx]
                if cost < 1.3:
                    self.tracks[t_idx].update(detections[d_idx]["box"], detections[d_idx]["conf"], frame_id)
                    matched_track_indices.add(t_idx)
                    matched_det_indices.add(d_idx)
                    
            # Handle unmatched detections (spawn new tracks)
            for d_idx, det in enumerate(detections):
                if d_idx not in matched_det_indices:
                    new_track = Track(det["box"], det["conf"], self.next_id, frame_id, max_age=self.max_age)
                    self.tracks.append(new_track)
                    self.next_id += 1
        else:
            # If no tracks or no detections, handle simple cases
            for det in detections:
                new_track = Track(det["box"], det["conf"], self.next_id, frame_id, max_age=self.max_age)
                self.tracks.append(new_track)
                self.next_id += 1

        # 3. Clean up dead tracks (exceeding max_age)
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # 4. Format outputs as Track objects and return them
        return self.tracks

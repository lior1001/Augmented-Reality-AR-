import cv2
import numpy as np
import pickle
from collections import deque

class ParticleFlowAR:
    """
    Augmented Reality system with bidirectional particle flow between multiple tracked planes.
    """
    
    def __init__(self, calibration_file='camera_calibration.pkl'):
        """
        Initialize the AR system.
        
        Args:
            calibration_file: Path to camera calibration file
        """
        # Load camera calibration
        self.load_calibration(calibration_file)
        
        # Feature detector (SIFT or ORB)
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Reference images for 3 planes
        self.reference_images = []
        self.reference_keypoints = []
        self.reference_descriptors = []
        self.reference_sizes = []
        
        # Particles system
        self.particles = []
        self.num_particles = 200
        self.plane_colors = [
            (178, 12, 249),  # Pink-ish
            (146, 223, 57),  # Mint Green
            (239, 46, 32)    # Blue
        ]
        
        # Tracking state - seperate visible vs tracked status
        self.plane_centers_3d = {}  # World coordinates of plane centers (only VISIBLE planes)
        self.plane_centers_2d = {}  # Projected 2D coordinates (only VISIBLE planes)
        self.plane_visible = {0: False, 1: False, 2: False} # Currently visible in frame
        
        # Sphere pulsing animation
        self.sphere_pulse = {0: 0.0, 1: 0.0, 2: 0.0}
        
        # Motion trails
        self.particle_trails = {}  # Dictionary of deques for each particle
        self.trail_length = 15

        # Tracking Memory for Safety Buffer
        self.missing_frames = {0: 0, 1: 0, 2: 0}
        self.last_valid_centers = {} 

        # IMPROVED Smoothing for stable tracking (reduced for fast response)
        self.rvec_history = {0: [], 1: [], 2: []}       # List for rolling average
        self.tvec_history = {0: [], 1: [], 2: []}
        self.center_2d_history = {0: [], 1: [], 2: []}
        self.history_length = 3     # Reduced from 5 to 3 for faster response
        self.smoothing_alpha = 0.6  # Not used in rolling average but kept for reference
        
    def load_calibration(self, filename):
        """Load camera calibration parameters."""
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                calib_data = pickle.load(f)
            self.mtx = calib_data['mtx']
            self.dist = calib_data['dist']
        else:  # .npz
            data = np.load(filename)
            self.mtx = data['mtx']
            self.dist = data['dist']
        
        print("✓ Calibration loaded")
        print(f"Camera Matrix (K):\n{self.mtx}\n")
    
    def add_reference_image(self, image_path, plane_id):
        """
        Add a reference image for tracking.
        
        Args:
            image_path: Path to reference image
            plane_id: ID of the plane (0, 1, or 2)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        kp, desc = self.detector.detectAndCompute(gray, None)
        
        if desc is None or len(kp) < 10:
            raise ValueError(f"Not enough features in {image_path}")
        
        self.reference_images.append(img)
        self.reference_keypoints.append(kp)
        self.reference_descriptors.append(desc)
        self.reference_sizes.append((img.shape[1], img.shape[0]))  # (width, height)
        
        print(f"✓ Plane {plane_id}: Loaded with {len(kp)} features from {image_path}")
    
    def smooth_tracking_data(self, plane_id, center_2d, rvec, tvec):
        """
        Apply rolling average smoothing to tracking data.
        IMPORTANT: Uses .copy() to avoid reference issues.
        
        Args:
            plane_id: ID of the plane
            center_2d, rvec, tvec: Current tracking data
            
        Returns:
            Smoothed center_2d, rvec, tvec
        """
        # CRITICAL: .copy() prevents numpy reference issues
        self.center_2d_history[plane_id].append(center_2d.copy())
        self.rvec_history[plane_id].append(rvec.copy())
        self.tvec_history[plane_id].append(tvec.copy())
        
        # Keep only last N frames
        if len(self.center_2d_history[plane_id]) > self.history_length:
            self.center_2d_history[plane_id].pop(0)
            self.rvec_history[plane_id].pop(0)
            self.tvec_history[plane_id].pop(0)
        
        # Calculate rolling average
        smoothed_center = np.mean(self.center_2d_history[plane_id], axis=0)
        smoothed_rvec = np.mean(self.rvec_history[plane_id], axis=0)
        smoothed_tvec = np.mean(self.tvec_history[plane_id], axis=0)
        
        return smoothed_center, smoothed_rvec, smoothed_tvec

    def track_plane(self, frame, plane_id):
        """
        ROBUST: Track using ALL matched points, not just 4 corners.
        POSITION-AWARE: Relaxed thresholds for Plane 1 (upper-right position).
        
        Returns:
            success (bool), center_3d (np.array), center_2d (np.array), rvec, tvec
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in current frame
        kp_frame, desc_frame = self.detector.detectAndCompute(gray, None)
        
        if desc_frame is None or len(kp_frame) < 10:
            return False, None, None, None, None
        
        # Match features
        matches = self.matcher.knnMatch(self.reference_descriptors[plane_id], desc_frame, k=2)
        
        # MINIMAL FIX: Position-aware thresholds
        # Plane 2 (upper-right) needs slightly relaxed thresholds to prevent flickering
        if plane_id == 2:
            ratio_threshold = 0.75      # Slightly more lenient (was 0.7)
            min_matches = 10            # Lower (was 15)
            min_inliers = 8            # Lower (was 12)
            min_inlier_ratio = 0.4     # Lower (was 0.5)
        else:
            # Other planes keep current thresholds
            ratio_threshold = 0.7
            min_matches = 15
            min_inliers = 12
            min_inlier_ratio = 0.5
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < min_matches:
            return False, None, None, None, None
        
        # Get matching points
        ref_pts = np.float32([self.reference_keypoints[plane_id][m.queryIdx].pt 
                              for m in good_matches])
        frame_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches])
        
        # Find homography with RANSAC
        H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)
        
        if H is None or mask is None:
            return False, None, None, None, None
        
        # Check homography quality - require good inlier ratio
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(good_matches)
        
        if inliers < min_inliers or inlier_ratio < min_inlier_ratio:
            return False, None, None, None, None
        
        # ========== KEY FIX: Use ALL inlier points, not just 4 corners ==========
        inlier_indices = np.where(mask.ravel() == 1)[0]
        ref_pts_inliers = ref_pts[inlier_indices]
        frame_pts_inliers = frame_pts[inlier_indices]
        
        # Convert 2D reference points to 3D (on Z=0 plane)
        w, h = self.reference_sizes[plane_id]
        object_points = np.float32([[pt[0], pt[1], 0] for pt in ref_pts_inliers])
        image_points = frame_pts_inliers
        # ========================================================================
        
        # Ensure contiguous arrays
        object_points = np.ascontiguousarray(object_points)
        image_points = np.ascontiguousarray(image_points)
        camera_matrix = np.ascontiguousarray(self.mtx)
        dist_coeffs = np.ascontiguousarray(self.dist)
        
        try:
            # ========== KEY FIX: Use solvePnPRansac with many points ==========
            success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                iterationsCount=100,
                reprojectionError=3.0,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success or inliers_pnp is None or len(inliers_pnp) < 10:
                return False, None, None, None, None
            
            # Refine pose using only PnP inliers
            refined_object_points = object_points[inliers_pnp.ravel()]
            refined_image_points = image_points[inliers_pnp.ravel()]
            
            success, rvec, tvec = cv2.solvePnP(
                refined_object_points,
                refined_image_points,
                camera_matrix,
                dist_coeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            # ==================================================================
            
            if not success:
                return False, None, None, None, None
            
            # Calculate center in 3D (world coordinates)
            center_3d = np.array([w/2, h/2, 0], dtype=np.float32)
            
            # Project center to 2D
            center_2d, _ = cv2.projectPoints(
                center_3d.reshape(1, 1, 3),
                rvec, tvec, camera_matrix, dist_coeffs
            )
            center_2d = center_2d.reshape(2)
            
            # Validate
            if not np.isfinite(center_2d).all():
                return False, None, None, None, None
            
            return True, center_3d, center_2d, rvec, tvec
            
        except cv2.error:
            return False, None, None, None, None
    
    def bezier_curve(self, p0, p1, p2, t):
        """
        Calculate point on quadratic Bézier curve.
        
        Args:
            p0, p1, p2: Control points
            t: Parameter [0, 1]
        """
        return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
    
    def initialize_particles(self):
        """
        Initialize particle system with bidirectional flow.
        Creates particles flowing in BOTH directions between each pair of planes.
        This creates 6 flows total: 0→1, 1→0, 1→2, 2→1, 2→0, 0→2
        """
        self.particles = []
        
        # Define all 6 directional flows (bidirectional between each pair)
        flows = [
            (0, 1),  # Plane 0 to Plane 1
            (1, 0),  # Plane 1 to Plane 0
            (1, 2),  # Plane 1 to Plane 2
            (2, 1),  # Plane 2 to Plane 1
            (2, 0),  # Plane 2 to Plane 0
            (0, 2),  # Plane 0 to Plane 2
        ]
        
        particles_per_flow = self.num_particles // len(flows)
        
        for flow_idx, (start_plane, end_plane) in enumerate(flows):
            for i in range(particles_per_flow):
                particle = {
                    'start_plane': start_plane,
                    'end_plane': end_plane,
                    'color': self.plane_colors[start_plane],
                    't': np.random.random(),  # Position along curve [0, 1]
                    'speed': 0.005 + np.random.random() * 0.01,
                    'id': len(self.particles),
                    'flow_direction': (start_plane, end_plane),  # Remember the flow direction
                    'active': False     # Track if particle should be drawn
                }
                self.particles.append(particle)
                self.particle_trails[particle['id']] = deque(maxlen=self.trail_length)
    
    def update_particles(self):
        """Update particle positions and manage their activity."""
        # Get currently visible planes
        visible_planes = list(self.plane_centers_2d.keys())

        for particle in self.particles:
            start = particle['start_plane']
            end = particle['end_plane']

            # Check if both start and end planes are currently VISIBLE
            particle['active'] = (start in visible_planes and end in visible_planes)
            
            # Only update active particles
            if not particle['active']:
                # Clear trail when particles become inactice
                self.particle_trails[particle['id']].clear()
                continue
            
            particle['t'] += particle['speed']
            
            # If particle reached destination
            if particle['t'] >= 1.0:
                # Trigger pulse at destination
                self.sphere_pulse['end'] = 1.0
                
                # Reset particle to start of the SAME path (maintain flow direction)
                particle['t'] = 0.0
                
                # Trigger pulse at start
                self.sphere_pulse['start'] = 1.0
    
    def get_bezier_control_point(self, p0, p2, centroid, curve_type="in"):
        """
        Calculates P1 based on the Triangle Center.
        """
        p0 = np.array(p0, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        mid_point = (p0 + p2) / 2
        
        # If we don't have a centroid (only 2 points tracked), fallback to simple offset
        if centroid is None:
            # Fallback: simple perpendicular offset
            direction = p2 - p0
            normal = np.array([-direction[1], direction[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            result = mid_point + (normal * 50)

        else:
            # CENTROID LOGIC
            # Vector from Center --> to --> Midpoint
            vec_center_to_mid = mid_point - centroid
            
            # Normalize it (make length 1)
            dist = np.linalg.norm(vec_center_to_mid)
            if dist == 0: 
                return mid_point
            unit_vec = vec_center_to_mid / dist
            
            curvature_strength = 60.0   # Adjust this to make curves "deeper"
            
            if curve_type == "out":
                # Push AWAY from center
                # P1 = Midpoint + (Vector pointing away)
                result = mid_point + (unit_vec * curvature_strength)
            else:
                # Pull TOWARDS center
                # P1 = Midpoint - (Vector pointing away)
                result = mid_point - (unit_vec * curvature_strength)
            
        # Validate the result before returning
        if not np.isfinite(result).all():
            return mid_point  # Fallback to midpoint if invalid
        
        return result
    
    def draw_particles(self, frame, rvecs, tvecs):
        """Draw particles and their trails - ONLY for active particles."""
        
        for particle in self.particles:
            # CRITICAL: Only draw active particles
            if not particle['active']:
                continue
            
            start = particle['start_plane']
            end = particle['end_plane']
            
            # Double-check: both planes must be in plane_centers_2d
            if start not in self.plane_centers_2d or end not in self.plane_centers_2d:
                continue
            
            # Get 2D positions of start and end
            p0_2d = self.plane_centers_2d[start]
            p2_2d = self.plane_centers_2d[end]

            # Calculate centroid if we have 3 visible planes
            visible_planes = list(self.plane_centers_2d.keys())
            centroid = None
            if len(visible_planes) == 3:
                pts = np.array(list(self.plane_centers_2d.values()))
                centroid = np.mean(pts, axis=0)
            
            # Determine Curve Type
            inward_pairs = [(0, 1), (1, 2), (2, 0)]
            curve_type = "in" if (start, end) in inward_pairs else "out"
            
            # Calculate Control Point
            p1_2d = self.get_bezier_control_point(p0_2d, p2_2d, centroid, curve_type)
            
            # Calculate particle position on Bézier curve
            pos_2d = self.bezier_curve(p0_2d, p1_2d, p2_2d, particle['t'])
            
            # Validate the position
            if not np.isfinite(pos_2d).all() or np.abs(pos_2d[0]) > 10000:
                continue
            
            # Update trail
            trail = self.particle_trails[particle['id']]
            
            # Check if we should clear the trail (if position jumped too much)
            if len(trail) > 0 and np.linalg.norm(pos_2d - trail[-1]) > 100:
                trail.clear()
            
            trail.append(pos_2d)

            # Draw trail
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                thickness = max(1, int(3 * alpha))
                pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                pt2 = (int(trail[i][0]), int(trail[i][1]))
                cv2.line(frame, pt1, pt2, particle['color'], thickness)
            
            # Draw particle head
            center_pt = (int(pos_2d[0]), int(pos_2d[1]))
            cv2.circle(frame, center_pt, 5, particle['color'], -1)
            cv2.circle(frame, center_pt, 6, (255, 255, 255), 1)
        
    def draw_bezier_curves(self, frame):
        """
        Draws Bézier curves between visible planes only.
        """
        # Get list of currently visible planes
        visible_planes = list(self.plane_centers_2d.keys())

        # check if at least 2 planes are visible
        if len(visible_planes) < 2:
            return

        # Calculate Triangle Centroid (if we have 3 points)
        centroid = None
        if len(visible_planes) == 3:
            # --- 3-PLANE MODE (STAR) ---
            pts = np.array([self.plane_centers_2d[pid] for pid in visible_planes])
            centroid = np.mean(pts, axis=0) # The math average (Center of triangle)

            # Define which pairs go IN and which go OUT
            # All 6 flows
            flows = [
                (0, 1), (1, 2), (2, 0), # Inward
                (1, 0), (2, 1), (0, 2)  # Outward
            ]
        
        elif len(visible_planes) == 2:
            # --- 2-PLANE MODE (DIRECT ARC) ---
            # No centroid logic. Just connection.            
            p1_id = visible_planes[0]
            p2_id = visible_planes[1]
            # Only draw the flow between these two specific planes
            flows = [(p1_id, p2_id), (p2_id, p1_id)]
        else:
            return  # Should not happen given check above

        # Draw selected curves
        inward_pairs = [(0, 1), (1, 2), (2, 0)]

        for start, end in flows:
            # Only draw if both planes are visible
            if start not in self.plane_centers_2d or end not in self.plane_centers_2d:
                continue

            p0 = self.plane_centers_2d[start]
            p2 = self.plane_centers_2d[end]

            # Determine if this curve should go IN or OUT
            curve_type = "in" if (start, end) in inward_pairs else "out"
         
            # Calculate Control Point relative to Centroid
            p1 = self.get_bezier_control_point(p0, p2, centroid, curve_type)
            
            # Draw curve
            points = []
            valid_curve = True

            for t in np.linspace(0, 1, 50):
                pt = self.bezier_curve(p0, p1, p2, t)

                # Check for NaN/Inf/Overflow before casting
                if not np.isfinite(pt).all() or abs(pt[0]) > 10000:
                    valid_curve = False
                    break
                points.append(pt.astype(int))
            
            if valid_curve and len(points) > 0:
                points = np.array(points)
            
                # Use different colors for IN vs OUT to debug
                #color = (0, 255, 0) if curve_type == "in" else (0, 0, 255)
                #cv2.polylines(frame, [points], False, color, 2)

    def draw_glowing_spheres(self, frame):
        """Draw pulsing spheres at plane centers."""
         # Only iterate over planes that are ACTUALLY in plane_centers_2d
        for plane_id in self.plane_centers_2d.keys():

            raw_center = self.plane_centers_2d[plane_id]

            # Safety check for infinity/NaN
            if not np.isfinite(raw_center).all():
                continue

            # SANITY CHECK: Ignore coordinates that are way off-screen
            # This prevents the "Wrong Type" error caused by huge numbers (overflow)
            if abs(raw_center[0]) > 10000 or abs(raw_center[1]) > 10000:
                continue

            # Convert numpy types to pure python ints
            cx = int(raw_center[0]) 
            cy = int(raw_center[1])
            center = (cx, cy)

            color = self.plane_colors[plane_id]
            
            # Update pulse
            if self.sphere_pulse[plane_id] > 0:
                self.sphere_pulse[plane_id] -= 0.05
            
            # Base radius + pulse
            base_radius = 10
            pulse_factor = 1 + self.sphere_pulse[plane_id] * 0.6
            
            num_layers = 15      # More layers = smoother fade
            max_glow_dist = 60   # How far the glow spreads out

            # Draw multiple circles for glow effect
            for i in range(num_layers,-1, -1):
                # 'progress' goes from 1.0 (outer edge) down to 0.0 (center)
                progress = i / num_layers

                # Calculate Radius: Center is small, Edge is large
                current_radius = int((base_radius + max_glow_dist * progress) * pulse_factor)
                
                # Calculate Alpha:
                # The math (1 - progress)**2 makes the alpha drop off quickly
                # causing a bright center and very faint edges.
                alpha = 0.12 * ((1.0 - progress) ** 2)
                
                # Skip invisible layers
                if alpha < 0.005: continue

                # Draw semi-transparent layer
                overlay = frame.copy()
                cv2.circle(overlay, center, current_radius, color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # (Optional) Tiny white hot-spot in the very center for "energy" look
            # kept very small and semi-transparent to blend in
            overlay = frame.copy()
            cv2.circle(overlay, center, int(base_radius * 0.8), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    def process_video(self, video_path, output_path='output_part5.mp4'):
        """
        Process video with Anti-Flicker Buffer.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.initialize_particles()
        frame_count = 0
        
        # --- CONFIG: How many frames to remember a lost plane ---
        # 8 frames is ~0.3 seconds at 30fps. Enough to bridge the gap of a flicker.
        BUFFER_SIZE = 6  

        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Reset per-frame data
            self.plane_centers_3d.clear()
            self.plane_centers_2d.clear()
            rvecs = {}
            tvecs = {}
            
            for plane_id in range(3):
                # 1. Try to track
                success, center_3d, center_2d, rvec, tvec = self.track_plane(frame, plane_id)
                
                # 2. Check validity (Must be tracked AND on screen)
                is_valid = False
                if success and center_2d is not None:
                    cx, cy = center_2d
                    if 0 <= cx <= width and 0 <= cy <= height:
                        is_valid = True

                if is_valid:        
                    # --- CASE A: FOUND ---
                    self.missing_frames[plane_id] = 0  # Reset counter
                    
                    # Apply smoothing
                    smoothed_center, smoothed_rvec, smoothed_tvec = self.smooth_tracking_data(
                        plane_id, center_2d, rvec, tvec
                    )
                    
                    # Store data
                    self.plane_centers_3d[plane_id] = center_3d
                    self.plane_centers_2d[plane_id] = smoothed_center
                    rvecs[plane_id] = smoothed_rvec
                    tvecs[plane_id] = smoothed_tvec
                    
                else:
                    # --- CASE B: LOST (Potential Flicker) ---
                    self.missing_frames[plane_id] += 1
                    
                    # Check if we can use the "Memory Buffer"
                    # We need history to exist AND be within the time limit
                    if self.missing_frames[plane_id] < BUFFER_SIZE and len(self.center_2d_history[plane_id]) > 0:
                        
                        # Use LAST KNOWN position
                        last_center = self.center_2d_history[plane_id][-1]
                        
                        # Only use it if the last known position was actually on screen
                        if 0 <= last_center[0] <= width and 0 <= last_center[1] <= height:
                            self.plane_centers_2d[plane_id] = last_center
                            
                            # Also recover rotation/translation for particles
                            if len(self.rvec_history[plane_id]) > 0:
                                rvecs[plane_id] = self.rvec_history[plane_id][-1]
                                tvecs[plane_id] = self.tvec_history[plane_id][-1]
                    else:
                        # Time is up - actually delete the plane
                        self.center_2d_history[plane_id] = []
                        self.rvec_history[plane_id] = []
                        self.tvec_history[plane_id] = []
                        
            # 3. Drawing loop
            num_visible = len(self.plane_centers_2d)
            
            if num_visible >= 2:
                self.draw_bezier_curves(frame)
                self.update_particles()
                self.draw_particles(frame, rvecs, tvecs)
                self.draw_glowing_spheres(frame)
            
            # 4. Status Display
            cv2.putText(frame, f"Planes tracked: {num_visible}/3", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for i in range(3):
                if i in self.plane_centers_2d:
                    if self.missing_frames[i] > 0:
                        status = "BUFFER"     # Orange = Saved by code
                        color = (0, 165, 255) 
                    else:
                        status = "VISIBLE"    # Green = Solid tracking
                        color = self.plane_colors[i]
                else:
                    status = "LOST"           # Grey = Gone
                    color = (100, 100, 100)
                    
                cv2.putText(frame, f"Plane {i}: {status}", (10, 60 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)

            if frame_count % 30 == 0 or frame_count == total_frames:
                print(f"Processed {frame_count}/{total_frames} frames...", end='\r')
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Processed {frame_count} frames")
        print(f"✓ Output saved: {output_path}")

def main():
    """Main function to run Part 5."""
    
    print("=" * 50)
    print("Part 5: Particle Flow Visualization (Odd Variant)")
    print("=" * 50)
    
    print()
    
    # Initialize AR system
    ar = ParticleFlowAR(calibration_file='camera_calibration.pkl')
    
    # Add reference images for the 3 planes
    ar.add_reference_image('part5_1.jpg', 1)
    ar.add_reference_image('print10.jpg', 0)
    ar.add_reference_image('part5_3.jpg', 2)
    
    # Process video
    ar.process_video('part_5_video2.mp4', 'output_part5.mp4')
    
    print("\n" + "=" * 50)
    print("✓ Finished!")
    print("=" * 50)


if __name__ == "__main__":
    main()
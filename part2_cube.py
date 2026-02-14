import cv2
import numpy as np
import os

class CubeRenderer:
    """
    Class for rendering a 3D cube on a warped image.
    
    Process:
    1. Detects reference image in frame (using SIFT)
    2. Uses solvePnP to find camera position
    3. Projects 3D cube points to image
    4. Draws the cube
    """
    
    def __init__(self, reference_image_path, camera_matrix, dist_coeffs):
        """
        Initialize the renderer.
        
        Args:
            reference_image_path: path to reference image (the image we track)
            camera_matrix: K matrix from calibration
            dist_coeffs: distortion coefficients from calibration
        """

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # 1. Load reference image and SIFT setup
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError("Reference image not found: {reference_image_path}")
        
        self.ref_img = cv2.imread(reference_image_path)
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        self.ref_kp, self.ref_desc = self.sift.detectAndCompute(self.ref_gray, None)

        # Initialize Matcher (FLANN)
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 2. Prepare 3D points of the reference image corners (Z=0 plane)
        h, w = self.ref_gray.shape
        # we define the reference image as a flat rectangle on the ground (Z=0)
        # order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        self.ref_3d_points = np.float32([
            [0, 0, 0],  # Top-Left
            [w, 0, 0],  # Top-Right
            [w, h, 0],  # Bottom-Right
            [0, h, 0]   # Bottom-Left
        ])

        # cube size relative to image width (20% of width)
        self.cube_size = w * 0.2

        # 3. Smoothing variables
        self.rvec_history = None
        self.tvec_history = None
        self.alpha = 1.0        #Smoothing factor (0.1 = very smooth/slow, 0.9 = fast/jittery)

        
    def find_pose(self, frame):
        """
        Find camera position relative to reference image using All-Points PnP (for stability).
        
        Args:
            frame: frame from video
            
        Returns:
            success: whether we found the position
            rvec: rotation vector
            tvec: translation vector
            matches: good matches
        """

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in frame
        frame_kp, frame_desc = self.sift.detectAndCompute(frame_gray, None)
        
        if frame_desc is None or len(frame_kp) < 4:
            return False, None, None, []
        
        # Match features
        matches = self.flann.knnMatch(self.ref_desc, frame_desc, k=2)
        
        # Filter matches - Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 8:  # Need at least 8 good matches for PnP
            return False, None, None, good_matches
        
        # Prepare point for Homography and PnP
        # Step 1: Get 3D coordinates of the features on the Reference Image
        obj_points = []
        img_points = []
        for m in good_matches:
            # Get the (x, y) of the keypoints on the original reference image
            ref_idx = m.queryIdx
            px, py = self.ref_kp[ref_idx].pt
            obj_points.append([px, py, 0])  # Z=0 plane

            # Get the (x, y) of the match in the current video frame
            frame_idx = m.trainIdx
            img_points.append(frame_kp[frame_idx].pt)

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # Step 2: Use solvePnP with RANSAC (RANSAC automatically ignores "bad" matches that don't fit the plane)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points,
            img_points,
            self.camera_matrix,
            self.dist_coeffs,
            iterationsCount=1000,
            reprojectionError=3.0,
            flags=cv2.SOLVEPNP_IPPE
        )

        #ref_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        #frame_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Step 2: Compute homography to find the corners
        #H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)
        
        #if H is None:
        #    return False, None, None, good_matches
        
        # Find the position of 4 corners of the image in the current frame
        #h, w = self.ref_gray.shape

        # Get corners of the original reference image
        #corners_ref = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Project these corners into the current frame using homography
        #corners_frame = cv2.perspectiveTransform(corners_ref, H)
        
        # Now we match: 
        # we have 4 2D points in frame (corners_frame) and 4 3D points in world (self.ref_3d_points)
        #object_points = self.ref_3d_points
        #image_points = corners_frame.reshape(-1, 2)
        
        # Use solvePnP
        #success, rvec, tvec = cv2.solvePnP(
        #    object_points,      # 3D points (4,3)
        #    image_points,       # 2D points (4,2)
        #    self.camera_matrix,
        #    self.dist_coeffs,
        #    flags=cv2.SOLVEPNP_IPPE
        #)

        if success:

            current_alpha = 1.0     # if jitters - lower to 0.8 etc.
            # exponential moving average for smoothing
            if self.rvec_history is None:
                self.rvec_history = rvec
                self.tvec_history = tvec
            else:
                # Apply smoothing: New = (Alpha * Current) + ((1 - Alpha) * History)
                self.rvec_history = (current_alpha * rvec) + ((1 - current_alpha) * self.rvec_history)
                self.tvec_history = (current_alpha * tvec) + ((1 - current_alpha) * self.tvec_history)
        
            # Return the raw values
            return success, self.rvec_history, self.tvec_history, good_matches

        return False, None, None, good_matches
    
    def draw_cube(self, frame, rvec, tvec, color=(0, 255, 0), thickness=3):
        """
        Draw cube on frame with colored axes.
        
        🟡 Yellow - X axis (Length)
        🔵 Blue - Y axis (Width)
        🔴 Red - Z axis (Height)
        
        Args:
            frame: frame to draw on
            rvec: rotation vector
            tvec: translation vector
            color: cube color (not used, kept for compatibility)
            thickness: line thickness
        """
        
        cube_size = self.cube_size
        
        # Define 8 corners of the cube (Z is negative for "up" in OpenCV)
        axis = np.float32([
            [0, 0, 0],                      # corner 0: bottom left-front
            [cube_size, 0, 0],              # corner 1: bottom right-front
            [cube_size, cube_size, 0],      # corner 2: bottom right-back
            [0, cube_size, 0],              # corner 3: bottom left-back
            [0, 0, -cube_size],             # corner 4: top left-front
            [cube_size, 0, -cube_size],     # corner 5: top right-front
            [cube_size, cube_size, -cube_size],  # corner 6: top right-back
            [0, cube_size, -cube_size]      # corner 7: top left-back
        ])
        
        # Project 3D points to 2D image 
        imgpts, _ = cv2.projectPoints(
            axis, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        imgpts = imgpts.reshape(-1, 2).astype(int)
        
        # Define colors
        YELLOW = (0, 255, 255)    # 🟡 Yellow - X axis (Length)
        BLUE = (255, 0, 0)        # 🔵 Blue - Y axis (Width)
        RED = (0, 0, 255)         # 🔴 Red - Z axis (Height)
        
        # ===================================
        # 🟡 Lines in X direction (Length) - Yellow
        # ===================================
        x_lines = [
            (0, 1),  # bottom front
            (3, 2),  # bottom back
            (4, 5),  # top front
            (7, 6)   # top back
        ]
        
        for p1, p2 in x_lines:
            cv2.line(frame, tuple(imgpts[p1]), tuple(imgpts[p2]), YELLOW, thickness)
        
        # ===================================
        # 🔵 Lines in Y direction (Width) - Blue
        # ===================================
        y_lines = [
            (0, 3),  # bottom left
            (1, 2),  # bottom right
            (4, 7),  # top left
            (5, 6)   # top right
        ]
        
        for p1, p2 in y_lines:
            cv2.line(frame, tuple(imgpts[p1]), tuple(imgpts[p2]), BLUE, thickness)
        
        # ===================================
        # 🔴 Lines in Z direction (Height) - Red
        # ===================================
        z_lines = [
            (0, 4),  # pillar left-front
            (1, 5),  # pillar right-front
            (2, 6),  # pillar right-back
            (3, 7)   # pillar left-back
        ]
        
        for p1, p2 in z_lines:
            cv2.line(frame, tuple(imgpts[p1]), tuple(imgpts[p2]), RED, thickness)
        
        return frame
    
    def draw_axes(self, frame, rvec, tvec, length=None):
        """
        Draw coordinate axes (X, Y, Z) for debugging.
        
        Red = X
        Green = Y
        Blue = Z
        """

        if length is None:
            length = self.cube_size
        
        # Define 3 axes points
        axes = np.float32([
            [0, 0, 0],       # origin
            [length, 0, 0],  # X axis (red)
            [0, length, 0],  # Y axis (green)
            [0, 0, -length]  # Z axis (blue) - negative because up
        ])
        
        # Projection
        imgpts, _ = cv2.projectPoints(
            axes, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        imgpts = imgpts.reshape(-1, 2).astype(int)
        
        origin = tuple(imgpts[0])
        
        # Draw lines
        frame = cv2.line(frame, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X - red
        frame = cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y - green
        frame = cv2.line(frame, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z - blue
        
        return frame
    
    def process_video(self, video_path, output_path, show_axes=False):
        """
        Process video and add cube.
        
        Args:
            video_path: path to input video
            output_path: path to save output
            show_axes: whether to display coordinate axes
        """

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video parameters
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        frame_count = 0
        successful_tracks = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Find position
            success, rvec, tvec, matches = self.find_pose(frame)
            
            if success:
                # Draw cube
                frame = self.draw_cube(frame, rvec, tvec)
                
                if show_axes:
                    frame = self.draw_axes(frame, rvec, tvec)
                
                cv2.putText(frame, f"Tracking: {len(matches)} matches",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                successful_tracks += 1
            else:
                cv2.putText(frame, "Tracking Lost",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame to video
            out.write(frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames... ({successful_tracks/frame_count*100:.1f}% tracking)")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Finished!")
        print(f"Total frames: {frame_count}")
        print(f"Tracking successful: {successful_tracks} ({successful_tracks/frame_count*100:.1f}%)")
        print(f"Saved to: {output_path}")


def main():
    """
    Example usage.
    """

    print("=" * 50)
    print("Part 2: Cube with Colored Axes")
    print("🟡 Yellow - X axis (Length)")
    print("🔵 Blue - Y axis (Width)")
    print("🔴 Red - Z axis (Height)")
    print("=" * 50)
    
    # 1. Load calibration parameters
    print("\n1. Loading calibration parameters...")
    try:
        calib_data = np.load('camera_calibration.npz')
        camera_matrix = calib_data['mtx']
        dist_coeffs = calib_data['dist']
        print("✓ Calibration loaded successfully")
        print(f"Camera Matrix:\n{camera_matrix}")
    except Exception as e:
        print(f"✗ Calibration file not found: {e}")
        print("Please run part2_calibration.py first")
        return
    
    # 2. Create renderer
    print("\n2. Initializing renderer...")

    reference_image = "reference_image.jpg"  # update path
    input_video = "old_reference.mp4"          # update path
    output_video = "output_cube_colored.mp4"  # with colored axes!

    try:
        renderer = CubeRenderer(reference_image, camera_matrix, dist_coeffs)
    except Exception as e:
        print(f"✗ Error initializing: {e}")
        print("Make sure the reference image path is correct")
        return
    
    # 3. Process video
    print("\n3. Processing video...")
    
    try:
        renderer.process_video(input_video, output_video, show_axes=True)
    except Exception as e:
        print(f"✗ Error processing: {e}")
        return
    
    print("\n" + "=" * 50)
    print("✓ Part 2 completed!")
    print("🟡 Yellow = X (Length)")
    print("🔵 Blue = Y (Width)")
    print("🔴 Red = Z (Height)")
    print("=" * 50)


if __name__ == "__main__":
    main()
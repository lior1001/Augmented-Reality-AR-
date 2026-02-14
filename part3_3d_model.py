import cv2
import numpy as np
import trimesh
import open3d as o3d

class Model3DRenderer:
    """
    Class for loading and rendering 3D models (.obj, .ply, .stl)
    on a warped image using AR.
    """
    
    def __init__(self, reference_image_path, model_path, camera_matrix, dist_coeffs, use_texture=True):
        """
        Initialize the renderer.
        
        Args:
            reference_image_path: path to reference image
            model_path: path to 3D model (.obj, .ply, .stl)
            camera_matrix: K matrix from calibration
            dist_coeffs: distortion coefficients
            use_texture: whether to use textures (if available)
        """
        # Load reference image
        self.ref_img = cv2.imread(reference_image_path)
        if self.ref_img is None:
            raise ValueError(f"Failed to load image: {reference_image_path}")
        
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        
        # Camera parameters
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Initialize SIFT
        self.sift = cv2.SIFT_create()
        self.ref_kp, self.ref_desc = self.sift.detectAndCompute(self.ref_gray, None)
        
        # Initialize FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Define 3D reference points
        h, w = self.ref_img.shape[:2]
        self.base_size = w
        self.ref_3d_points = np.float32([
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0]
        ])

        # Load model
        print(f"Loading model: {model_path}")
        self.model_path = model_path
        self.use_texture = use_texture
        self.load_model(model_path)

        self.rvec_history = None
        self.tvec_history = None

        
    def load_model(self, model_path):
        """
        Load 3D model.
        Supports: .obj, .ply, .stl
        """
        try:
            # Try with trimesh
            self.mesh = trimesh.load(model_path)
            print(f"✓ Model loaded successfully")
            print(f"  Vertices: {len(self.mesh.vertices)}")
            print(f"  Faces: {len(self.mesh.faces)}")
            
            # Normalize model - fit to image size
            self.normalize_model()
            
            # Save texture (if available)
            self.has_texture = hasattr(self.mesh.visual, 'material') and self.use_texture
            if self.has_texture:
                print("✓ Model includes texture")
            
        except Exception as e:
            print(f"Error loading model with trimesh: {e}")
            print("Trying with open3d...")
            
            try:
                # Try with open3d
                self.o3d_mesh = o3d.io.read_triangle_mesh(model_path)
                
                # Convert to trimesh
                vertices = np.asarray(self.o3d_mesh.vertices)
                faces = np.asarray(self.o3d_mesh.triangles)
                self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                print(f"✓ Model loaded successfully with open3d")
                self.normalize_model()
                self.has_texture = False
                
            except Exception as e2:
                raise ValueError(f"Failed to load model: {e2}")
    
    def normalize_model(self):
        """
        Normalize model - size and position in space.
        """
        # We apply a 90-degree rotation around X to align with camera view
        theta = np.radians(-90)
        c, s = np.cos(theta), np.sin(theta)

        # Rotation matrix around X axis
        Rx = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        
        # Apply the rotation to all vertices
        self.mesh.vertices = (Rx @ self.mesh.vertices.T).T

        # Center model around (0, 0)
        centroid = self.mesh.centroid
        self.mesh.vertices -= centroid
        
        # Scale - fit to image size
        # Model will occupy ~60% of image size
        bounds = self.mesh.bounds
        size = bounds[1] - bounds[0]
        max_dim = max(size)
        
        scale_factor = (self.base_size * 0.6) / max_dim
        self.mesh.vertices *= scale_factor
        
        # Translate - position model at image center, slightly above surface
        h, w = self.ref_img.shape[:2]
        self.mesh.vertices[:, 0] += w / 2  # center X
        self.mesh.vertices[:, 1] += h / 2  # center Y
        self.mesh.vertices[:, 2] -= self.base_size * 0.3  # lift above surface
        
        # Lift above surface (Negative Z is "out" towards the camera)
        self.mesh.vertices[:, 2] -= self.base_size * 0.3

        print(f"✓ Model normalized to appropriate size")
    
    def find_pose(self, frame):
        """
        Find camera position using PnP.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_desc = self.sift.detectAndCompute(frame_gray, None)
        
        if frame_desc is None or len(frame_kp) < 4:
            return False, None, None, []
        
        matches = self.flann.knnMatch(self.ref_desc, frame_desc, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 8:
            return False, None, None, good_matches
        
        obj_points = []
        img_points = []

        for m in good_matches:
            # 3D Point: Get x,y from Reference Image and add Z=0
            px, py = self.ref_kp[m.queryIdx].pt
            obj_points.append([px, py, 0])
            
            # 2D Point: Get x,y from Current Frame
            img_points.append(frame_kp[m.trainIdx].pt)
            
        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # Use solvePnPRansac (Matches Part 2 logic)
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                iterationsCount=1000,
                reprojectionError=3.0,
                flags=cv2.SOLVEPNP_IPPE
            )
        except Exception:
            # Fallback if IPPE fails (sometimes happens with specific point configs)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                iterationsCount=1000,
                reprojectionError=3.0,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success:
            return False, None, None, good_matches

        # --- SMOOTHING LOGIC (The anti-jitter fix) ---
        # Alpha: 0.1 = Very Smooth (Slow), 1.0 = No Smoothing (Jittery)
        # 0.5 is a good balance.
        alpha = 0.3 
        
        if self.rvec_history is None:
            self.rvec_history = rvec
            self.tvec_history = tvec
        else:
            self.rvec_history = (alpha * rvec) + ((1 - alpha) * self.rvec_history)
            self.tvec_history = (alpha * tvec) + ((1 - alpha) * self.tvec_history)
            
        return True, self.rvec_history, self.tvec_history, good_matches
    
    def render_model(self, frame, rvec, tvec, wireframe=False):
        """
        Render model on frame.
        
        Args:
            frame: frame to draw on
            rvec: rotation vector
            tvec: translation vector
            wireframe: whether to draw as wireframe or solid
        """
        # Project all vertices
        vertices_3d = self.mesh.vertices.astype(np.float32)
        
        projected_points, _ = cv2.projectPoints(
            vertices_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        projected_points = projected_points.reshape(-1, 2).astype(np.int32)
        
        # Draw triangles
        for face in self.mesh.faces:
            pts = projected_points[face]
            
            # Check that all points are inside frame
            if np.all((pts[:, 0] >= 0) & (pts[:, 0] < frame.shape[1]) &
                     (pts[:, 1] >= 0) & (pts[:, 1] < frame.shape[0])):
                
                if wireframe:
                    # Only lines
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
                else:
                    # Solid triangle
                    # Calculate color based on normal
                    v1 = vertices_3d[face[1]] - vertices_3d[face[0]]
                    v2 = vertices_3d[face[2]] - vertices_3d[face[0]]
                    normal = np.cross(v1, v2)
                    normal = normal / (np.linalg.norm(normal) + 1e-6)
                    
                    # Simple lighting
                    light_dir = np.array([0, 0, -1])  # light from camera
                    intensity = max(0, np.dot(normal, light_dir))
                    
                    color = (int(100 + 155 * intensity),
                            int(100 + 155 * intensity),
                            int(200 + 55 * intensity))
                    
                    cv2.fillConvexPoly(frame, pts, color)
                    # Outline
                    cv2.polylines(frame, [pts], True, (0, 0, 0), 1)
        
        return frame
    
    def render_model_advanced(self, frame, rvec, tvec):
        """
        More advanced rendering with z-buffering and better lighting.
        """
        h, w = frame.shape[:2]
        
        # Create z-buffer
        z_buffer = np.full((h, w), np.inf)
        output = frame.copy()
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Project vertices
        vertices_3d = self.mesh.vertices.astype(np.float32)
        projected_points, _ = cv2.projectPoints(
            vertices_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate depth for each vertex
        vertices_cam = (R @ vertices_3d.T).T + tvec.reshape(1, 3)
        depths = vertices_cam[:, 2]
        
        # Sort triangles by average depth (painter's algorithm)
        face_depths = []
        for i, face in enumerate(self.mesh.faces):
            avg_depth = np.mean(depths[face])
            face_depths.append((avg_depth, i, face))
        
        face_depths.sort(reverse=True)  # far to near
        
        # Draw triangles
        for _, _, face in face_depths:
            pts = projected_points[face].astype(np.int32)
            
            # Basic check
            if not np.all((pts[:, 0] >= 0) & (pts[:, 0] < w) &
                         (pts[:, 1] >= 0) & (pts[:, 1] < h)):
                continue
            
            # Calculate normal
            v1 = vertices_3d[face[1]] - vertices_3d[face[0]]
            v2 = vertices_3d[face[2]] - vertices_3d[face[0]]
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len == 0: continue # Skip degenerate triangles
            normal = normal / norm_len

            # Lighting
            light_dir = np.array([0.3, 0.3, -1])
            light_dir = light_dir / np.linalg.norm(light_dir)
            intensity = max(0.2, np.dot(normal, light_dir))
            
            # Color
            base_color = np.array([100, 150, 255])
            # Safe casting for color
            color_vals = (base_color * intensity).astype(int)
            color = (int(color_vals[0]), int(color_vals[1]), int(color_vals[2]))            
            
            # Draw
            cv2.fillConvexPoly(output, pts, color)
            cv2.polylines(output, [pts], True, (50, 50, 50), 1)
        
        return output
    
    def process_video(self, video_path, output_path, advanced_render=True):
        """
        Process video and add 3D model.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        frame_count = 0
        successful_tracks = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            success, rvec, tvec, matches = self.find_pose(frame)
            
            if success:
                if advanced_render:
                    frame = self.render_model_advanced(frame, rvec, tvec)
                else:
                    frame = self.render_model(frame, rvec, tvec)
                
                cv2.putText(frame, f"Tracking: {len(matches)} matches",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                successful_tracks += 1
            else:
                cv2.putText(frame, "Tracking Lost",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames... ({successful_tracks/frame_count*100:.1f}%)")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Finished!")
        print(f"Total: {frame_count} frames")
        print(f"Tracking: {successful_tracks} ({successful_tracks/frame_count*100:.1f}%)")
        print(f"Saved to: {output_path}")


def download_sample_model():
    """
    Download sample model (optional).
    """
    import urllib.request
    
    # Examples of free models:
    models = {
        'teapot': 'https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/teapot.obj',
        'bunny': 'https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/bunny.obj',
    }
    
    print("Available models for download:")
    for name, url in models.items():
        print(f"  - {name}: {url}")
    
    # Example download:
    # urllib.request.urlretrieve(models['teapot'], 'teapot.obj')


def main():
    """
    Example usage.
    """
    print("=" * 60)
    print("Part 3: 3D Model Rendering")
    print("=" * 60)
    
    # 1. Load calibration
    print("\n1. Loading calibration...")
    try:
        calib_data = np.load('camera_calibration.npz')
        camera_matrix = calib_data['mtx']
        dist_coeffs = calib_data['dist']
        print("✓ Calibration loaded")
    except:
        print("✗ Calibration file not found!")
        return
    
    # 2. Initialize renderer
    print("\n2. Initializing renderer...")
    reference_image = "reference_image.jpg"
    model_path = "tree.obj"  # update to your model path
    
    try:
        renderer = Model3DRenderer(
            reference_image,
            model_path,
            camera_matrix,
            dist_coeffs
        )
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTips:")
        print("- Make sure you have a .obj/.ply/.stl model")
        print("- You can download free models from:")
        print("  * https://free3d.com")
        print("  * https://sketchfab.com")
        print("  * https://github.com/alecjacobson/common-3d-test-models")
        return
    
    # 3. Process video
    print("\n3. Processing video...")
    input_video = "old_reference.mp4"
    output_video = "output_3d_model.mp4"
    
    try:
        renderer.process_video(input_video, output_video, advanced_render=True)
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✓ Part 3 completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

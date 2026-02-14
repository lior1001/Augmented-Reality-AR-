"""
Part 4: Part 3 המקורי + אוקלוזיה בלבד
========================================

לוקח את Part 3 שכבר עובד מצוין
+ מוסיף רק אוקלוזיה (כמו שהחבר אמר)
ללא שינויים ב-tracking/stability!
"""

import cv2
import numpy as np
from collections import deque


class SimpleOBJLoader:
    """טוען .obj פשוט."""
    
    def __init__(self, filepath):
        self.vertices = []
        self.faces = []
        
        print(f"טוען מודל: {filepath}")
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    self.vertices.append([x, y, z])
                
                elif parts[0] == 'f':
                    face = []
                    for p in parts[1:]:
                        vertex_idx = int(p.split('/')[0]) - 1
                        face.append(vertex_idx)
                    
                    if len(face) >= 3:
                        self.faces.append(face[:3])
        
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.faces = np.array(self.faces, dtype=np.int32)
        
        print(f"✓ נטען: {len(self.vertices)} vertices, {len(self.faces)} faces")


class Part3WithOcclusion:
    """
    Part 3 המקורי + אוקלוזיה.
    ללא שינויים ב-tracking!
    """
    
    def __init__(self, reference_image_path, model_path, camera_matrix, dist_coeffs):
        """אתחול."""
        self.ref_img = cv2.imread(reference_image_path)
        if self.ref_img is None:
            raise ValueError(f"Failed to load image: {reference_image_path}")
        
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        
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

        # טעינת מודל
        print(f"\nטוען מודל: {model_path}")
        try:
            self.obj_loader = SimpleOBJLoader(model_path)
            self.vertices = self.obj_loader.vertices
            self.faces = self.obj_loader.faces
            self.normalize_model()
        except Exception as e:
            print(f"⚠️  שגיאה: {e}")
            raise

        # *** רק smoothing פשוט כמו ב-Part 3! ***
        self.rvec_history = None
        self.tvec_history = None
        
        # *** הוספה: זיהוי יד ***
        self.skin_lower1 = np.array([0, 40, 60], dtype=np.uint8)
        self.skin_upper1 = np.array([20, 255, 255], dtype=np.uint8)
        self.skin_lower2 = np.array([170, 40, 60], dtype=np.uint8)
        self.skin_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        
        self.hand_mask_buffer = deque(maxlen=3)
    
    def normalize_model(self):
        """נורמליזציה - בדיוק כמו Part 3."""
        # סיבוב 90 מעלות
        theta = np.radians(-90)
        c, s = np.cos(theta), np.sin(theta)
        Rx = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        
        self.vertices = (Rx @ self.vertices.T).T

        # מרכוז
        center = np.mean(self.vertices, axis=0)
        self.vertices -= center
        
        # סקילה - קטן יותר! (0.45 במקום 0.6)
        bounds_min = np.min(self.vertices, axis=0)
        bounds_max = np.max(self.vertices, axis=0)
        size = bounds_max - bounds_min
        max_dim = np.max(size)
        
        scale_factor = (self.base_size * 0.32) / max_dim  # 
        self.vertices *= scale_factor
        
        # מיקום - שמאל-למטה! (במקום מרכז)
        h, w = self.ref_img.shape[:2]
        self.vertices[:, 0] += w * 0.43  # 📍 שמאל 
        self.vertices[:, 1] += h * 0.57  # 📍 למטה
        self.vertices[:, 2] -= self.base_size * 0.3
        self.vertices[:, 2] -= self.base_size * 0.3

        print(f"✓ מודל נורמל")
    
    def find_pose(self, frame):
        """
        *** בדיוק כמו Part 3 - ללא שינויים! ***
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
            px, py = self.ref_kp[m.queryIdx].pt
            obj_points.append([px, py, 0])
            img_points.append(frame_kp[m.trainIdx].pt)
            
        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # solvePnPRansac - בדיוק כמו Part 3
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
        except:
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

        # *** Smoothing פשוט - בדיוק כמו Part 3! ***
        alpha = 0.3 
        
        if self.rvec_history is None:
            self.rvec_history = rvec
            self.tvec_history = tvec
        else:
            self.rvec_history = (alpha * rvec) + ((1 - alpha) * self.rvec_history)
            self.tvec_history = (alpha * tvec) + ((1 - alpha) * self.tvec_history)
            
        return True, self.rvec_history, self.tvec_history, good_matches
    
    def detect_hand_mask(self, frame):
        """
        *** זיהוי יד - כמו שהחבר אמר! ***
        1. זיהוי גוון עור (HSV)
        2. Erosion/Dilation לשיפור
        3. Temporal smoothing
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # זיהוי גוון עור
        mask1 = cv2.inRange(hsv, self.skin_lower1, self.skin_upper1)
        mask2 = cv2.inRange(hsv, self.skin_lower2, self.skin_upper2)
        hand_mask = cv2.bitwise_or(mask1, mask2)
        
        # *** Erosion/Dilation - כמו שהחבר אמר! ***
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        # Opening: הסרת רעש
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Closing: מילוי חורים
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        
        # Erosion: הקטנה
        hand_mask = cv2.erode(hand_mask, kernel_small, iterations=1)
        
        # Dilation: הגדלה
        hand_mask = cv2.dilate(hand_mask, kernel_medium, iterations=2)
        
        # Temporal smoothing
        self.hand_mask_buffer.append(hand_mask.astype(np.float32))
        
        if len(self.hand_mask_buffer) > 0:
            avg_mask = np.mean(self.hand_mask_buffer, axis=0)
            _, hand_mask = cv2.threshold(
                avg_mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
            )
        
        return hand_mask
    
    def render_model_with_occlusion(self, frame, rvec, tvec, hand_mask):
        """
        *** רינדור + אוקלוזיה - כמו שהחבר אמר! ***
        
        1. ציור מודל על layer נפרד
        2. יצירת מסכת אובייקט
        3. זיהוי חיתוך בין יד לאובייקט
        4. יד מעל אובייקט במקום חיתוך!
        """
        h, w = frame.shape[:2]
        model_layer = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert rotation
        R, _ = cv2.Rodrigues(rvec)
        
        # Project vertices
        projected_points, _ = cv2.projectPoints(
            self.vertices,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate depth
        vertices_cam = (R @ self.vertices.T).T + tvec.reshape(1, 3)
        depths = vertices_cam[:, 2]
        
        # Z-ordering (painter's algorithm)
        face_depths = []
        for i, face in enumerate(self.faces):
            avg_depth = np.mean(depths[face])
            face_depths.append((avg_depth, i, face))
        
        face_depths.sort(reverse=True)  # מאחור לקדימה
        
        # ציור פאות
        for _, _, face in face_depths:
            pts = projected_points[face].astype(np.int32)
            
            # בדיקת גבולות
            if not np.all((pts[:, 0] >= 0) & (pts[:, 0] < w) &
                         (pts[:, 1] >= 0) & (pts[:, 1] < h)):
                continue
            
            # חישוב normal לתאורה
            v1 = self.vertices[face[1]] - self.vertices[face[0]]
            v2 = self.vertices[face[2]] - self.vertices[face[0]]
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len == 0: 
                continue
            normal = normal / norm_len

            # תאורה - יותר עדינה!
            light_dir = np.array([0.3, 0.3, -1])
            light_dir = light_dir / np.linalg.norm(light_dir)
            intensity_raw = np.dot(normal, light_dir)
            
            # 🎨 צבע כהה + אחיד יותר!
            # במקום intensity משתנה (0.2-1.0), נעשה טווח קטן יותר (0.65-0.85)
            intensity = 0.65 + max(0, intensity_raw) * 0.2  # טווח צר!
            
            base_color = np.array([100, 120, 120])  
            color_vals = (base_color * intensity).astype(int)
            color = (int(color_vals[0]), int(color_vals[1]), int(color_vals[2]))            
            
            # ציור
            cv2.fillConvexPoly(model_layer, pts, color)
            cv2.polylines(model_layer, [pts], True, (50, 50, 50), 1)
        
        # *** שלב 2: מסכת אובייקט ***
        model_gray = cv2.cvtColor(model_layer, cv2.COLOR_BGR2GRAY)
        _, object_mask = cv2.threshold(model_gray, 1, 255, cv2.THRESH_BINARY)
        
        # *** שלב 3: זיהוי חיתוך - כמו שהחבר אמר! ***
        intersection_mask = cv2.bitwise_and(hand_mask, object_mask)
        
        # *** שלב 4: יד מעל אובייקט במקום חיתוך! ***
        # מודל ללא יד
        model_visible_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(intersection_mask))
        model_visible = cv2.bitwise_and(model_layer, model_layer, mask=model_visible_mask)
        
        # פריים (רקע + יד)
        frame_mask = cv2.bitwise_or(cv2.bitwise_not(object_mask), intersection_mask)
        frame_visible = cv2.bitwise_and(frame, frame, mask=frame_mask)
        
        # חיבור
        result = cv2.add(model_visible, frame_visible)
        
        return result, hand_mask, intersection_mask
    
    def process_video(self, video_path, output_path, show_debug=True):
        """עיבוד וידאו."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"נכשל לפתוח: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n{'='*60}")
        print(f"Part 4: Part 3 + אוקלוזיה בלבד")
        print(f"מעבד: {video_path}")
        print(f"רזולוציה: {width}x{height}, FPS: {fps}")
        print(f"{'='*60}\n")
        
        frame_count = 0
        successful_tracks = 0
        occlusion_events = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            success, rvec, tvec, matches = self.find_pose(frame)
            
            if success:
                # זיהוי יד
                hand_mask = self.detect_hand_mask(frame)
                
                # רינדור + אוקלוזיה
                result, hand_mask_vis, intersection = self.render_model_with_occlusion(
                    frame, rvec, tvec, hand_mask
                )
                
                # בדיקת אוקלוזיה
                has_occlusion = np.sum(intersection > 0) > 500
                if has_occlusion:
                    occlusion_events += 1
                
                # טקסט
                status = "OCCLUSION!" if has_occlusion else "Clear"
                color_status = (0, 0, 255) if has_occlusion else (0, 255, 0)
                
                cv2.putText(result, f"Status: {status}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
                
                cv2.putText(result, f"Matches: {len(matches)}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Debug view
                if show_debug:
                    hand_small = cv2.resize(hand_mask_vis, (120, 90))
                    hand_colored = cv2.applyColorMap(hand_small, cv2.COLORMAP_HOT)
                    result[10:100, width-140:width-20] = hand_colored
                    cv2.rectangle(result, (width-140, 10), (width-20, 100), (255, 255, 255), 2)
                    cv2.putText(result, "Hand", (width-130, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    inter_small = cv2.resize(intersection, (120, 90))
                    inter_colored = cv2.applyColorMap(inter_small, cv2.COLORMAP_JET)
                    result[110:200, width-140:width-20] = inter_colored
                    cv2.rectangle(result, (width-140, 110), (width-20, 200), (255, 255, 255), 2)
                    cv2.putText(result, "Overlap", (width-130, 210), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                frame = result
                successful_tracks += 1
            else:
                cv2.putText(frame, "Tracking Lost",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(frame)
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"התקדמות: {frame_count}/{total_frames} ({progress:.1f}%) | Occlusions: {occlusion_events}")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✓ הסתיים!")
        print(f"{'='*60}")
        print(f"סה\"כ פריימים: {frame_count}")
        print(f"Tracking: {successful_tracks} ({successful_tracks/frame_count*100:.1f}%)")
        print(f"אוקלוזיות: {occlusion_events} פעמים")
        print(f"נשמר: {output_path}")
        print(f"{'='*60}")


def main():
    """הרצה."""
    import os
    
    print("=" * 60)
    print("Part 4: פשוט + מתוקן!")
    print("=" * 60)
    print("מה בקוד:")
    print("  ✅ Tracking כמו Part 3 (alpha=0.3)")
    print("  ✅ זיהוי יד (HSV)")
    print("  ✅ Erosion/Dilation")
    print("  ✅ זיהוי חיתוך")
    print("  ✅ יד מעל אובייקט")
    print("")
    print("תיקונים חדשים:")
    print("  🔧 גודל: 0.45 (קטן יותר מ-0.6)")
    print("  🔧 מיקום: שמאל-למטה (35%, 65%)")
    print("  🔧 צבע: כהה + אחיד (טווח 0.65-0.85)")
    print("=" * 60)
    
    required_files = ['camera_calibration.npz', 'new_reference_image.jpg', 'part4video.mp4']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f"\n❌ חסרים קבצים: {', '.join(missing)}")
        return
    
    model_path = 'tree.obj' if os.path.exists('tree.obj') else None
    if not model_path:
        print("❌ לא נמצא tree.obj")
        return
    
    print("\n1. טוען כיול...")
    try:
        calib_data = np.load('camera_calibration.npz')
        camera_matrix = calib_data['mtx']
        dist_coeffs = calib_data['dist']
        print("   ✓ כיול נטען")
    except Exception as e:
        print(f"   ✗ שגיאה: {e}")
        return
    
    print("\n2. מאתחל...")
    try:
        renderer = Part3WithOcclusion(
            'new_reference_image.jpg',
            model_path,
            camera_matrix,
            dist_coeffs
        )
        print("   ✓ אתחול הצליח")
    except Exception as e:
        print(f"   ✗ שגיאה: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n3. מעבד וידאו...")
    try:
        renderer.process_video(
            'part4video.mp4',
            'output_part4.mp4',
            show_debug=True
        )
    except Exception as e:
        print(f"\n✗ שגיאה: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
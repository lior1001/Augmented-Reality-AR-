import cv2
import numpy as np
import glob
import pickle
import os

class CameraCalibrator:
    """
    Tool for camera calibration using chessboard pattern.
    
    Calibration allows us to:
    1. Know the intrinsic parameters of the camera (K matrix)
    2. Correct optical distortions (distortion coefficients)
    3. Calculate the position of the camera in 3D space
    """
    
    def __init__(self, chessboard_size=(8, 6), square_size=1.0):
        """
        Calibrate camera using chessboard images.
        
        Args:
            images_path: Path pattern to calibration images (e.g., 'calibration/*.jpg')
            pattern_size: Inner corners of chessboard (columns, rows)
            square_size: Size of chessboard square in your units (e.g., cm, mm)
            save_path: Where to save calibration results
            
        Returns:
            ret: RMS reprojection error
            K: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            rvecs: Rotation vectors for each image
            tvecs: Translation vectors for each image
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Prepare 3D points of the chessboard
        # Assume the board is in the Z=0 plane (flat on the table)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Lists for storing points from all images
        self.objpoints = []  # 3D points in real world
        self.imgpoints = []  # 2D points in image
        
        # Calibration parameters (will be computed)
        self.mtx = None          # K matrix - camera intrinsic matrix
        self.dist = None         # distortion coefficients
        self.rvecs = None        # rotation vectors
        self.tvecs = None        # translation vectors
        
    def capture_calibration_images(self, num_images=20, camera_id=0, save_dir='calibration_images'):
        """
        Capture images of a chessboard for calibration.
        
        Instructions:
        1. Print a chessboard (you can find an image on the internet)
        2. Run this function
        3. Hold the board in front of the camera at different angles and distances
        4. Press SPACE to capture an image
        5. Press ESC when finished
        
        Args:
            num_images: number of images to capture (recommended 15-25)
            camera_id: camera ID (0 = default camera)
            save_dir: directory to save the images
        """
        # Create directory
        os.makedirs(save_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError("Failed to open camera!")
        
        print(f"Capture mode - need {num_images} images")
        print("Instructions:")
        print("  - Hold the chessboard at different angles")
        print("  - Press SPACE to capture")
        print("  - Press ESC to finish")
        
        count = 0
        
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret_chess, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            # הצגה
            display_frame = frame.copy()
            if ret_chess:
                cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret_chess)
                cv2.putText(display_frame, "Board detected! Press SPACE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Chessboard not found", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(display_frame, f"Images: {count}/{num_images}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.imshow('Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # SPACE - save image
            if key == ord(' ') and ret_chess:
                filename = os.path.join(save_dir, f'calib_{count:02d}.jpg')
                cv2.imwrite(filename, frame)
                print(f"✓ Saved: {filename}")
                count += 1
            
            # ESC - exit
            elif key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Saved {count} images in {save_dir}")
        return save_dir
    
    def calibrate_from_images(self, images_path):
        """
        Perform camera calibration from existing images.
        
        Args:
            images_path: path to folder with calibration images or glob pattern (e.g., 'calibration_images/*.jpg')
        
        Returns:
            dict: dictionary with calibration parameters
        """

        # Load all images
        if os.path.isdir(images_path):
            images = glob.glob(os.path.join(images_path, '*.jpg'))
            images += glob.glob(os.path.join(images_path, '*.png'))
        else:
            images = glob.glob(images_path)
        
        if len(images) == 0:
            raise ValueError(f"No images found on {images_path}")
        
        print(f"Found {len(images)} images")
        print("Searching for chessboard corners...")
        
        successful_images = 0
        
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                self.objpoints.append(self.objp)
                
                # Improve corner precision (subpixel accuracy)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners_refined)
                
                successful_images += 1
                print(f"  ✓ Image {idx+1}/{len(images)}: detected")
            else:
                print(f"  ✗ Image {idx+1}/{len(images)}: board not detected")
        
        if successful_images < 10:
            raise ValueError(f"Only {successful_images} images detected. Need at least 10!")
        
        print(f"\n✓ Successfully detected {successful_images} images")
        print("Performing camera calibration...")
        
        # Perform calibration
        img_size = gray.shape[::-1]
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None
        )
        
        if not ret:
            raise ValueError("Calibration failed!")
        
        print("✓ Calibration finished successfully!")
        
        # Calculate reprojection error
        mean_error = self.calculate_reprojection_error()
        print(f"Reprojection error: {mean_error:.4f} pixels")
        
        return {
            'mtx': self.mtx,
            'dist': self.dist,
            'rvecs': self.rvecs,
            'tvecs': self.tvecs,
            'mean_error': mean_error
        }
    
    def calculate_reprojection_error(self):
        """
        Calculate reprojection error - measures how accurate the calibration is.
        Low error (<0.5) = good calibration
        """
        total_error = 0
        total_points = 0
        
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], 
                self.mtx, self.dist
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
            total_points += 1
        
        return total_error / total_points if total_points > 0 else 0
    
    def save_calibration(self, filename='camera_calibration.pkl'):
        """
        Save calibration parameters to file.
        """
        calib_data = {
            'mtx': self.mtx,
            'dist': self.dist,
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(calib_data, f)
        
        print(f"✓ Calibration saved to {filename}")
        
        # Also save as numpy file
        np.savez(filename.replace('.pkl', '.npz'),
                 mtx=self.mtx,
                 dist=self.dist)
        print(f"✓ Calibration also saved to {filename.replace('.pkl', '.npz')}")
    
    def load_calibration(self, filename='camera_calibration.pkl'):
        """
        Load calibration parameters from file.
        """
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                calib_data = pickle.load(f)
            self.mtx = calib_data['mtx']
            self.dist = calib_data['dist']
        else:  # .npz
            data = np.load(filename)
            self.mtx = data['mtx']
            self.dist = data['dist']
        
        print(f"✓ Calibration loaded from {filename}")
        print(f"Camera Matrix (K):\n{self.mtx}")
        print(f"\nDistortion Coefficients:\n{self.dist}")
        
        return self.mtx, self.dist
    
    def undistort_image(self, img):
        """
        Correct distortion in image.
        """
        if self.mtx is None or self.dist is None:
            raise ValueError("Must caliberate first!")
        
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        
        # Correct distortion
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        
        # Crop according to ROI
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        return dst


def main():
    """
    Main function to run camera calibration.
    """

    print("=" * 30)
    print("Camera Calibration")
    print("=" * 30)
    
    # 1. Initialize calibrator
    # The checkerboard has 8x6 internal corners (9x7 squares)
    calibrator = CameraCalibrator(chessboard_size=(8, 6))
    
    # Run calibration
    
    print("\nRunning calibration from existing images...")

    try:
        # This runs the calibration and stores the result inside the 'calibrator' object
        calibrator.calibrate_from_images('calibration_images')

        # 3. Save the result of calibration
        # We need to save the K matrix and distortion coefficients to a file for later use in Part 2
        print("\nSaving calibration data...")
        calibrator.save_calibration('camera_calibration.pkl') 
     
    except ValueError as e:
        print(f"Error during calibration: {e}")
        print("Make sure your 'calibration_images' folder exists and contains valid chessboard .jpg files.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    print("\n" + "=" *30)
    print("✓ Finished!")
    print("=" * 30)


if __name__ == "__main__":
    main()

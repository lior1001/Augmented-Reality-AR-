import cv2 
import numpy as np
import matplotlib.pyplot as plt

class PerspectiveWarper:
    def __init__(self, reference_image_path, template_image_path):
        """
        Initialize the perspective warper with reference and template images.
        Args:
            reference_image_path (str): Path to the reference image to track.
            template_image_path (str): Path to the template image to warp onto reference.
        """

        # Load images
        self.ref_img = cv2.imread(reference_image_path)
        self.template_img = cv2.imread(template_image_path)

        if self.ref_img is None or self.template_img is None:
            raise ValueError("Could not load one or both images. Check the file paths.")
        
        # Convert images to grayscale for SIFT
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()

        # Detect and compute keypoints and descriptors for reference image
        self.ref_kp, self.ref_desc = self.sift.detectAndCompute(self.ref_gray, None)

        # Initialize FLANN matcher for fast descriptor matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Resize template image to match reference dimensions
        self.template_img = cv2.resize(self.template_img, (self.ref_img.shape[1], self.ref_img.shape[0]))


    def find_homography(self, frame):
        """
        Find homography between the reference image and the current frame.
        Args:
            frame (numpy.ndarray): Current video frame (BRG).
        Returns:
            H: 3x3 Homography matrix (or None if not enough matches found).
            matches: List of good matches.
            frame_kp: Keypoints detected in the current frame.
        """

        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and compute keypoints and descriptors for the current frame
        frame_kp, frame_desc = self.sift.detectAndCompute(frame_gray, None)

        if frame_desc is None or len(frame_kp) < 4:
            return None, [], frame_kp
        
        # Match descriptors using FLANN matcher
        matches = self.flann.knnMatch(self.ref_desc, frame_desc, k=2)

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:   # Ratio threshold
                    good_matches.append(m)

        # Need at least 4 good matches to compute homography
        if len(good_matches) < 4:
            return None, good_matches, frame_kp
        
        # Extract matched keypoint locations
        ref_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches])
        frame_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches])

        # Compute homography using RANSAC
        H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)

        return H, good_matches, frame_kp
        
    def warp_template(self, frame, H):
        """
        Warp the template image onto the current frame using the homography matrix.
        Args:
            frame (numpy.ndarray): Current video frame (BGR).
            H (numpy.ndarray): 3x3 Homography matrix from reference to frame.
        Returns:
            result: Frame with the template image warped onto it.
        """

        if H is None:
            return frame

        h, w = frame.shape[:2]
        warped_template = cv2.warpPerspective(self.template_img, H, (w, h))

        # Create a mask of the warped region
        mask = cv2.warpPerspective(np.ones_like(self.template_img)*255, H, (w, h))
        mask_grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Blend warped template onto the frame
        result = frame.copy()
        result[mask_grey > 0] = warped_template[mask_grey > 0]

        return result

    def draw_matches_debug(self, frame, matches, frame_kp):
        """
        Draw matched keypoints for debugging.
        Args:
            frame (numpy.ndarray): Current video frame (BGR).
            good_matches (list): List of good matches.
            frame_kp (list): Keypoints detected in the current frame.
        Returns:
            Image with matches drawn.
        """

        # Draw first 50 matches
        match_img = cv2.drawMatches(self.ref_img, self.ref_kp, frame, frame_kp, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return match_img
    
    def process_video(self, video_path, output_path, show_matches=False):
        """
        Process video and apply perspective warping.
        Args:
            video_path (str): Path to input video.
            output_path (str): Path to save output video.
            show_matches (bool): Whether to show matching visualization.
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video file. Check the file path.")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        successful_tracks = 0

        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Find homography
            H, good_matches, frame_kp = self.find_homography(frame)

            # Warp template if homography found
            if H is not None:
                result = self.warp_template(frame, H)
                successful_tracks += 1

                # Add tracking status text
                cv2.putText(result, f"Tracking: {len(good_matches)} matches", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                result = frame.copy()
                cv2.putText(result, "Tracking Lost", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame
            out.write(result)
            
            # Display (optional)
            if show_matches and H is not None:
                match_img = self.draw_matches_debug(frame, good_matches, frame_kp)
                cv2.imshow('Matches', cv2.resize(match_img, (1200, 400)))
            
            # cv2.imshow('Result', cv2.resize(result, (800, 600)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Print statistics
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Successfully tracked: {successful_tracks} ({successful_tracks/frame_count*100:.1f}%)")
        print(f"Output saved to: {output_path}")

def main():
    """
    Main function to run perspective warping.
    """
    # Paths - UPDATE THESE
    reference_image = "new_reference_image.jpg"  # Your printed reference image
    template_image = "template_image.jpg"    # Image to warp onto reference
    input_video = "reference.mp4"           # Video of moving around printed image
    output_video = "output_warped.mp4"       # Output with warped template
    
    # Create warper
    warper = PerspectiveWarper(reference_image, template_image)
    
    # Process video
    warper.process_video(input_video, output_video, show_matches=False)


if __name__ == "__main__":
    main()
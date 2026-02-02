import cv2
import numpy as np
from scipy.spatial.distance import cdist


class SimpleDotMatcher:
    
    def __init__(self):
        # This will store our templates
        # Format: {'A': [dot positions], 'B': [dot positions], ...}
        self.templates = {}
    
    
    def extract_dots(self, image_path):
        """
        Step 1: Find all dots in an image
        
        ASSUMES: White background (255), black dots (0)
        
        Returns: Array of (x, y) positions like [[10, 20], [15, 25], ...]
        """
        # Load image as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Remove dark edges (vignette) by cropping 25% off each side
        h, w = img.shape
        margin = int(min(h, w) * 0.25)
        img = img[margin:h-margin, margin:w-margin]
        
        # Convert to binary
        # Now background is WHITE (255), dots are BLACK (0)
        # Threshold makes: background = black (0), dots = white (255) for contour detection
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find all white blobs (these are our dots)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the center of each dot
        dot_positions = []
        for contour in contours:
            # Skip tiny specs (noise)
            area = cv2.contourArea(contour)
            if area > 10:
                # Calculate center of this dot
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    x = int(M['m10'] / M['m00'])
                    y = int(M['m01'] / M['m00'])
                    dot_positions.append([x, y])
        
        return np.array(dot_positions)
    
    
    def normalize_dots(self, dots):
        """
        Step 2: Normalize dot positions
        
        This makes the pattern size and position independent.
        An "A" will look like an "A" no matter where it is or how big.
        
        How:
        - Move all dots so their center is at (0, 0)
        - Scale so the pattern fits in a 1x1 box
        - Sort dots consistently
        """
        if len(dots) == 0:
            return np.array([])
        
        dots = np.array(dots, dtype=float)
        
        # Step 2a: Center the dots at origin (0, 0)
        # Find the average position of all dots
        center = np.mean(dots, axis=0)
        # Subtract it from all dots
        centered = dots - center
        
        # Step 2b: Scale to fit in 1x1 box
        # Find the farthest dot from center
        max_distance = np.max(np.abs(centered))
        # Divide all positions by this distance
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
        
        # Step 2c: Sort dots left-to-right, top-to-bottom
        # This ensures we compare dots in the same order
        sorted_indices = np.lexsort((normalized[:, 1], normalized[:, 0]))
        normalized = normalized[sorted_indices]
        
        return normalized
    
    
    def add_template(self, letter, image_path):
        """
        Step 3: Save a template for a letter
        
        Example: matcher.add_template('A', 'goodA.jpeg')
        """
        print(f"Adding template for letter '{letter}'...")
        
        # Extract and normalize dots from this image
        dots = self.extract_dots(image_path)
        normalized = self.normalize_dots(dots)
        
        # Save it
        self.templates[letter] = normalized
        
        print(f"  ✓ Saved {len(dots)} dots for '{letter}'")

    def build_library(self):   
        # Add templates (do this once for each letter you want to recognize)

        print("STEP 1: Adding templates for every letter...")
        print("-"*70)
        self.add_template('A', 'A.jpeg')
        self.add_template('B', 'B.jpeg')
        self.add_template('C', 'C.jpeg')
        self.add_template('D', 'D.jpeg')
        self.add_template('E', 'E.jpeg')
        self.add_template('F', 'F.jpeg')
        self.add_template('G', 'G.jpeg')
        self.add_template('H', 'H.jpeg')
        self.add_template('I', 'I.jpeg')
        self.add_template('J', 'J.jpeg')
        self.add_template('K', 'K.jpeg')
        self.add_template('L', 'L.jpeg')
        self.add_template('M', 'M.jpeg')
        self.add_template('N', 'N.jpeg')
        self.add_template('O', 'O.jpeg')
        self.add_template('P', 'P.jpeg')
        self.add_template('Q', 'Q.jpeg')
        self.add_template('R', 'R.jpeg')
        self.add_template('S', 'S.jpeg')
        self.add_template('T', 'T.jpeg')
        self.add_template('U', 'U.jpeg')
        self.add_template('V', 'V.jpeg')
        self.add_template('W', 'W.jpeg')
        self.add_template('X', 'X.jpeg')
        self.add_template('Y', 'Y.jpeg')
        self.add_template('Z', 'Z.jpeg')
        print("-"*70)
        print()
        
    
    def calculate_similarity(self, dots1, dots2):
        """
        Step 4: Compare two dot patterns
        
        Returns: Similarity score (0-100)
        - 100 = perfect match
        - 0 = completely different
        
        How it works:
        For each dot in pattern1, find closest dot in pattern2.
        Average all these distances.
        Convert to a 0-100 score.
        """
        # Handle edge cases
        if len(dots1) == 0 or len(dots2) == 0:
            return 0.0
        
        # If very different number of dots, probably different letters
        dot_diff = abs(len(dots1) - len(dots2))
        if dot_diff > 20:
            return 0.0
        
        # Make both patterns same length by padding shorter one
        max_len = max(len(dots1), len(dots2))
        
        # Pad with zeros if needed
        if len(dots1) < max_len:
            padding = np.zeros((max_len - len(dots1), 2))
            dots1 = np.vstack([dots1, padding])
        if len(dots2) < max_len:
            padding = np.zeros((max_len - len(dots2), 2))
            dots2 = np.vstack([dots2, padding])
        
        # Calculate distance between corresponding dots
        distances = np.sqrt(np.sum((dots1 - dots2) ** 2, axis=1))
        
        # Average distance
        avg_distance = np.mean(distances)
        
        # Convert to similarity score (0-100)
        # Distance of 0 = 100% similar
        # Distance of 0.5 = 50% similar
        # Distance of 1+ = 0% similar
        similarity = max(0, 100 - (avg_distance * 100))
        
        return similarity
    
    
    def recognize(self, image_path):
        """
        Step 5: Recognize which letter a dot image is
        
        Returns: (best_letter, confidence, all_scores)
        
        Example:
            letter, confidence, scores = matcher.recognize('mystery.jpg')
            print(f"This is '{letter}' with {confidence}% confidence")
        """
        print(f"\nRecognizing {image_path}...")
        
        # Extract and normalize dots from unknown image
        dots = self.extract_dots(image_path)
        normalized = self.normalize_dots(dots)
        
        print(f"  Found {len(dots)} dots")
        
        # Compare against all templates
        scores = {}
        for letter, template in self.templates.items():
            similarity = self.calculate_similarity(normalized, template)
            scores[letter] = similarity
            print(f"  {letter}: {similarity:.1f}% match")
        
        # Find best match
        if scores:
            best_letter = max(scores, key=scores.get)
            confidence = scores[best_letter]
        else:
            best_letter = None
            confidence = 0
        
        return best_letter, confidence, scores


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("SIMPLE DOT PATTERN MATCHER")
    print("="*70)
    print()
    
    # Create the matcher
    matcher = SimpleDotMatcher()
    
    # Build templates (do this once for each letter you want to recognize)
    print("STEP 1: Adding templates...")
    print("-"*70)
    matcher.build_library()
    print()
    
    # Now recognize unknown images
    print("="*70)
    print("STEP 2: Testing recognition...")
    print("="*70)
    
    # Test 1: Recognize goodA (should match 'A')
    letter1, conf1, scores1 = matcher.recognize('badA.jpeg')
    print(f"\n  → RESULT: '{letter1}' with {conf1:.1f}% confidence")
    
    print()
    print("="*70)
    print("DONE!")
    print("="*70)
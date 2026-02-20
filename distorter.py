import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def apply_fisheye_distortion(image):
    """Apply extreme fisheye distortion - FIXED"""
    h, w = image.shape[:2]
    
    # Extreme distortion
    k1 = random.uniform(1.5, 3.0)
    k2 = random.uniform(0.8, 1.5)
    k3 = random.uniform(-0.3, 0.3)
    k4 = random.uniform(-0.3, 0.3)
    
    # Camera matrix - MUST be 3x3 float32
    f = float(w)
    K = np.array([
        [f, 0, w/2],
        [0, f, h/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # CRITICAL: Exactly 4 coefficients as float32
    D = np.array([k1, k2, k3, k4], dtype=np.float32)
    
    # Verify shapes
    assert K.shape == (3, 3), f"K shape is {K.shape}, expected (3,3)"
    assert len(D) == 4, f"D length is {len(D)}, expected 4"
    
    try:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
        )
        
        distorted = cv2.remap(
            image, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return distorted
    except Exception as e:
        print(f"Fisheye error: {e}")
        return image  # Return original on error

def create_dataset(input_path, output_path, limit=1000):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(input_path.rglob(ext))
        image_paths.extend(input_path.rglob(ext.upper()))
    
    image_paths = list(set(image_paths))[:limit]
    random.shuffle(image_paths)
    
    print(f"Processing {len(image_paths)} images...")
    
    success = 0
    errors = 0
    
    for idx, img_path in enumerate(tqdm(image_paths)):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                continue
            
            # Resize
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            
            # Apply distortion
            distorted = apply_fisheye_distortion(img)
            
            # Combine: [Distorted | Clean]
            combined = np.hstack((distorted, img))
            
            # Save
            cv2.imwrite(str(output_path / f"pair_{success:06d}.jpg"), combined)
            success += 1
            
        except Exception as e:
            errors += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Complete!")
    print(f"  Success: {success}")
    print(f"  Errors: {errors}")
    print(f"{'='*60}")

if __name__ == "__main__":
    create_dataset(
        input_path="data/raw",
        output_path="data/processed",
        limit=1000  # Start with 100 for testing
    )
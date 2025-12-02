import cv2
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def resize_image(img, min_dimension=1024):
    """Resize image while maintaining aspect ratio"""
    height, width = img.shape[:2]
    
    if min(height, width) < min_dimension:
        scale = min_dimension / min(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    return img

def normalize_lighting(img):
    """Apply CLAHE for better contrast and lighting"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def remove_shadows(img):
    """Remove shadows and glare using morphological operations"""
    rgb_planes = cv2.split(img)
    result_planes = []
    
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    
    return cv2.merge(result_planes)

def denoise_image(img):
    """Apply noise reduction while preserving text edges"""
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised

def binarize_for_handwriting(img):
    """Apply adaptive thresholding for better handwriting visibility"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Gaussian thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    
    # Convert back to BGR for consistency
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return binary_bgr

def deskew_image(img):
    """Correct any slight skew/tilt in the document"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Find all non-zero points
    coords = np.column_stack(np.where(thresh > 0))
    
    if len(coords) == 0:
        return img
    
    # Calculate the angle
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Only deskew if angle is significant (more than 0.5 degrees)
    if abs(angle) < 0.5:
        return img
    
    # Rotate the image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated

def preprocess_image(img, apply_binarization=False):
    """
    Complete preprocessing pipeline for form images
    
    Args:
        img: Input image (BGR format)
        apply_binarization: If True, apply binarization (useful for handwriting)
    
    Returns:
        Processed image
    """
    # Step 1: Resize if needed
    img = resize_image(img, min_dimension=1024)
    
    # Step 2: Deskew first (before other operations)
    img = deskew_image(img)
    
    # Step 3: Remove shadows
    img = remove_shadows(img)
    
    # Step 4: Normalize lighting and contrast
    img = normalize_lighting(img)
    
    # Step 5: Denoise
    img = denoise_image(img)
    
    # Step 6: Optional binarization (for handwriting enhancement)
    if apply_binarization:
        img = binarize_for_handwriting(img)
    else:
        # Step 6 Alternative: Sharpen text
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    
    return img

def process_single_image(input_path, output_path, apply_binarization=False):
    """Process a single image file"""
    try:
        # Read image
        img = cv2.imread(input_path)
        
        if img is None:
            print(f"Warning: Could not read image {input_path}")
            return False
        
        # Process image
        processed_img = preprocess_image(img, apply_binarization)
        
        # Save processed image
        cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def batch_process_dataset(input_folder, output_folder, apply_binarization=False):
    """
    Process all images in the dataset folder
    
    Args:
        input_folder: Path to input dataset folder
        output_folder: Path to output folder (will be created)
        apply_binarization: If True, apply binarization to all images
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Get all image files
    image_files = [
        f for f in input_path.rglob('*')
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Binarization: {'Enabled' if apply_binarization else 'Disabled'}")
    print("-" * 60)
    
    # Process images with progress bar
    successful = 0
    failed = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # Preserve directory structure
        relative_path = img_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        # Create subdirectories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process image
        if process_single_image(str(img_file), str(output_file), apply_binarization):
            successful += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Processing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(
        description="Process dataset images with multiple enhancement techniques"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to input dataset folder"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to output folder (processed images will be saved here)"
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Apply binarization for handwriting enhancement (default: False)"
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return
    
    # Process dataset
    batch_process_dataset(
        args.input_folder,
        args.output_folder,
        apply_binarization=args.binarize
    )

if __name__ == "__main__":
    main()
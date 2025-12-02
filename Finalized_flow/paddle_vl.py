import os
from PIL import Image
# Optimize PaddlePaddle memory allocation
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

from paddleocr import PaddleOCRVL

def resize_image_if_needed(image_path, max_side=1800):
    """Resizes image if it exceeds max_side to prevent OOM errors."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                # Use LANCZOS for high quality downsampling
                resample_method = getattr(Image, 'Resampling', Image).LANCZOS
                img = img.resize((new_w, new_h), resample_method)
                
                # Create a temporary path
                dir_name = os.path.dirname(image_path)
                file_name = os.path.basename(image_path)
                name, ext = os.path.splitext(file_name)
                new_path = os.path.join(dir_name, f"{name}_resized{ext}")
                
                img.save(new_path)
                print(f"Image resized from {w}x{h} to {new_w}x{new_h} to save memory.")
                return new_path, True
    except Exception as e:
        print(f"Warning: Could not resize image: {e}")
    return image_path, False

# Initialize the pipeline
# Disabling orientation and unwarping to save memory. 
# The dataset seems to be already orientation corrected.
pipeline = PaddleOCRVL(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_layout_detection=True
)

original_image_path = "/home/museum/Fine-tuned-TrOCR/orientation_corrected_nov21/image_1.jpg"
image_path, is_resized = resize_image_if_needed(original_image_path)

try:
    output = pipeline.predict(image_path)
    for res in output:
        res.print() ## Print the structured prediction output
        res.save_to_json(save_path="output_paddlevl") ## Save the current image's structured result in JSON format
        res.save_to_markdown(save_path="output_paddlevl") ## Save the current image's result in Markdown format
finally:
    # Clean up resized image
    if is_resized and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"Removed temporary file: {image_path}")
        except OSError:
            pass
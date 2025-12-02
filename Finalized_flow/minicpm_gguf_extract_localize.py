import os
import argparse
import json
import re
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
from huggingface_hub import hf_hub_download

# llama-cpp-python is required for GGUF models
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    raise ImportError(
        "llama-cpp-python or huggingface-hub is required. Please install them, e.g.:\n"
        "CMAKE_ARGS='-DLLAMA_CUBLAS=on' FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir\n"
        "pip install huggingface-hub"
    )

def image_to_base64(image):
    """Converts a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def setup_model(model_repo, model_filename):
    """Downloads (if needed) and loads the GGUF model."""
    print(f"Loading GGUF model: {model_filename}...")

    # Download the projector file and get its local path
    try:
        clip_model_path = hf_hub_download(
            repo_id=model_repo,
            filename="mmproj-model-f16.gguf"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download clip model. Ensure you have internet and the repo '{model_repo}' is correct. Error: {e}")

    # This requires a specific chat handler for MiniCPM-V (Llava 1.5 format)
    chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path)

    # Download the main model file
    try:
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_filename
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download main GGUF model. Ensure filename '{model_filename}' is correct. Error: {e}")


    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=2048,  # Context window
        logits_all=True,
        n_gpu_layers=-1, # Offload all layers to GPU
        verbose=False
    )
    print("Model loaded successfully.")
    return llm

def extract_bbox_from_response(response):
    """Extracts bounding box coordinates from the model's response string."""
    match = re.search(r"<box>([\d\s]+)</box>", response)
    if match:
        bbox_str = match.group(1)
        bbox = list(map(int, bbox_str.strip().split()))
        return bbox
    else:
        raise ValueError("Can't find bbox in the grounding response")

def crop_image_with_bbox(image, bbox):
    """Crops the image using the provided bounding box."""
    w, h = image.size
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image

def run_gguf_inference(llm, image, prompt_text):
    """Runs inference with the GGUF model."""
    image_base64 = image_to_base64(image)
    
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        # Parameters for deterministic output
        temperature=0.0,
    )
    return response['choices'][0]['message']['content']

def main():
    parser = argparse.ArgumentParser(description="Extract and localize from images using a MiniCPM GGUF model.")
    parser.add_argument("images_folder", type=str, help="Path to the folder containing image files.")
    args = parser.parse_args()

    if not os.path.isdir(args.images_folder):
        print(f"Error: The specified image folder does not exist: {args.images_folder}")
        return

    # --- Model and Prompt Setup ---
    # Define the repo and filename for the GGUF model you want to use.
    # This assumes you have downloaded the files from Hugging Face Hub.
    # You might need to adjust the filename based on the quantization you downloaded (e.g., Q4_0, Q5_K_M).
    MODEL_REPO = "openbmb/MiniCPM-V-4_5-gguf"
    MODEL_FILENAME = "ggml-model-Q5_1.gguf" # <-- IMPORTANT: Change this to your downloaded GGUF file name
    
    llm = setup_model(MODEL_REPO, MODEL_FILENAME)
    
    extraction_prompt = (
        "You are a precision-focused OCR extraction system for handwritten Nepali student forms. Analyze the provided scanned image and extract only the specified fields with strict adherence to the following rules. The 'phone number' and 'guardian's number' will be handled by a separate process, so you must ignore them.\n\n"
        "### Output Requirements\n"
        "Return ONLY a JSON object with these exact keys. Leave 'phone number' and 'guardian's number' as empty strings:\n"
        "{\n"
        "    \"Name\": \"\", \"class\": \"\", \"address\": \"\", \"faculty\": \"\", \"phone number\": \"\", "
        "    \"guardian's number\": \"\", \"email\": \"\", \"interested course\": \"\", \"school name\": \"\", "
        "    \"who gave seminar\": \"\", \"rate us\": \"\"\n"
        "}\n\n"
        "Begin extraction now. Remember: Do not extract the phone or guardian numbers."
    )
    
    fields_to_crop = ['phone number', "guardian's number"]

    # --- Output Directories ---
    model_name_slug = MODEL_FILENAME.replace('.gguf', '').lower()
    current_date = datetime.now().strftime("%B_%d").lower()
    output_folder = f"output_{model_name_slug}_{current_date}"
    cropped_output_dir = os.path.join(output_folder, 'cropped_numbers')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(cropped_output_dir, exist_ok=True)

    # --- Image Processing Loop ---
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in os.listdir(args.images_folder) if os.path.splitext(f)[1].lower() in supported_extensions]

    for i, image_file in enumerate(image_files, 1):
        print(f"\n{'='*20} Processing image {i}/{len(image_files)}: {image_file} {'='*20}")
        image_path = os.path.join(args.images_folder, image_file)
        base_name, _ = os.path.splitext(image_file)

        try:
            original_image = Image.open(image_path).convert('RGB')

            # --- Step 1: Full OCR Extraction ---
            print("--- Step 1: Extracting all fields... ---")
            extraction_result = run_gguf_inference(llm, original_image, extraction_prompt)
            print(f"Extraction output: {extraction_result}")

            output_filename = f"{base_name}_output.json"
            output_filepath = os.path.join(output_folder, output_filename)
            
            try:
                json_str = extraction_result.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                
                parsed_json = json.loads(json_str)
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(parsed_json, f, indent=4)
                print(f"Successfully saved extracted JSON to: {output_filepath}")
            except (IndexError, json.JSONDecodeError) as e:
                with open(output_filepath.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
                    f.write(extraction_result)

            # --- Step 2: Localize and Crop ---
            print("\n--- Step 2: Localizing and cropping number fields... ---")
            for field in fields_to_crop:
                try:
                    grounding_prompt = f'Please provide the bounding box coordinate of the region this sentence describes: <ref>{field}</ref>'
                    grounding_result = run_gguf_inference(llm, original_image, grounding_prompt)
                    print(f"Grounding model output for '{field}': {grounding_result}")
                    
                    bbox = extract_bbox_from_response(grounding_result)
                    cropped_image = crop_image_with_bbox(original_image, bbox)
                    
                    field_slug = field.replace("'", "").replace(" ", "_")
                    cropped_out_path = os.path.join(cropped_output_dir, f'{base_name}_cropped_{field_slug}.jpg')
                    cropped_image.save(cropped_out_path)
                    print(f"Saved cropped '{field}' to: {cropped_out_path}")

                except Exception as e:
                    print(f"Could not process field '{field}': {e}")
        
        except Exception as e:
            print(f"FATAL: Error processing {image_file}: {e}")

if __name__ == "__main__":
    main()

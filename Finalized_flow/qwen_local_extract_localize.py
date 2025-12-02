import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
import argparse
import json
import re
from datetime import datetime
import ast

def setup_model_and_processor(model_path):
    """Loads the Qwen3-VL model and processor from Hugging Face."""
    print(f"Loading local model: {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto", 
        trust_remote_code=True
    ).eval()
    print("Model loaded successfully.")
    return model, processor

def run_qwen_inference(model, processor, image_path, prompt):
    """
    Runs inference using the local Qwen3-VL model.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # The output is a list containing one string
    return output_text[0] if output_text else ""

def extract_bbox_from_qwen_response(response_text):
    """
    Parses a bounding box from the Qwen model's JSON response, even if malformed.
    """
    try:
        # Use a more general regex to find anything that looks like a JSON object
        match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the response.")
        
        json_str = match.group(1)
        
        # The model sometimes includes the label in the bbox list, so we need to clean it
        # e.g., "bbox_2d": [231, 353, 322, 393, "guardian's number"]
        # We will use regex to extract just the numbers.
        bbox_match = re.search(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', json_str)
        if not bbox_match:
            raise ValueError("Could not find a valid 'bbox_2d' with 4 integer coordinates.")
            
        bbox = [int(coord) for coord in bbox_match.groups()]
        return bbox
            
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse bounding box from response: '{response_text}'. Error: {e}")

def crop_image_with_bbox(image, bbox):
    """Crops the image using the provided bounding box (0-1000 scale)."""
    w, h = image.size
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image

def main():
    parser = argparse.ArgumentParser(description="Extract and localize from a single image using a local Qwen-VL model.")
    parser.add_argument("image_path", type=str, help="Path to the single image file.")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: The specified image file does not exist: {args.image_path}")
        return

    # --- Model and Prompt Setup ---
    MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
    model, processor = setup_model_and_processor(MODEL_PATH)
    
    extraction_prompt = (
        "You are a precision-focused OCR extraction system for handwritten Nepali student forms. Your task is to analyze the provided scanned image and extract only the specified fields into a JSON object. "
        "It is critical that you DO NOT extract 'phone number' and 'guardian's number'. These fields will be processed by a different system. You MUST leave them as empty strings.\n\n"
        "### Output Requirements\n"
        "Return ONLY a JSON object with these exact keys. The 'phone number' and 'guardian's number' fields must be empty strings. Do not add any extra text or explanations.\n"
        "{\n"
        "    \"Name\": \"\", \"class\": \"\", \"address\": \"\", \"faculty\": \"\", \"phone number\": \"\", "
        "    \"guardian's number\": \"\", \"email\": \"\", \"interested course\": \"\", \"school name\": \"\", "
        "    \"who gave seminar\": \"\", \"rate us\": \"\"\n"
        "}"
    )
    
    fields_to_crop = ['phone number', "guardian's number"]

    # --- Output Directories ---
    model_name_slug = MODEL_PATH.split('/')[-1].lower()
    current_date = datetime.now().strftime("%B_%d").lower()
    output_folder = f"output_{model_name_slug}_{current_date}"
    cropped_output_dir = os.path.join(output_folder, 'cropped_numbers')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(cropped_output_dir, exist_ok=True)
    print(f"JSON output will be saved to: {output_folder}")
    print(f"Cropped images will be saved to: {cropped_output_dir}")

    # --- Image Processing ---
    print(f"\n{'='*20} Processing image: {args.image_path} {'='*20}")
    base_name, _ = os.path.splitext(os.path.basename(args.image_path))

    try:
        original_image = Image.open(args.image_path).convert('RGB')

        # --- Step 1: Full OCR Extraction ---
        print("--- Step 1: Extracting all fields... ---")
        extraction_result = run_qwen_inference(model, processor, args.image_path, extraction_prompt)
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
            print(f"Could not parse JSON, saving raw output. Error: {e}")
            with open(output_filepath.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
                f.write(extraction_result)

        # --- Step 2: Localize and Crop specific fields ---
        print("\n--- Step 2: Localizing and cropping number fields... ---")
        for field in fields_to_crop:
            try:
                # Ask for just the label, which is more reliable
                grounding_prompt = f'Please locate the field label "{field}" in the image. Report its bbox coordinates in JSON format like this: {{"bbox_2d": [x1, y1, x2, y2], "label": "{field}"}}'
                
                grounding_result = run_qwen_inference(model, processor, args.image_path, grounding_prompt)
                print(f"Grounding model output for '{field}': {grounding_result}")
                
                # Get the bounding box for the label
                bbox = extract_bbox_from_qwen_response(grounding_result)

                # --- Expand the bounding box to include the value ---
                # The model returns the box for the label, so we expand it to the right 
                # to include the handwritten value area.
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                # Expand the width by 250% to the right to capture the value
                expanded_x2 = x2 + (width * 2.5) 
                # Ensure the expanded box doesn't go beyond the image boundary (1000)
                expanded_x2 = min(1000, expanded_x2)
                
                # Create the new, expanded bounding box
                expanded_bbox = [x1, y1, int(expanded_x2), y2]
                print(f"Original bbox for '{field}': {bbox}")
                print(f"Expanded bbox for '{field}': {expanded_bbox}")

                cropped_image = crop_image_with_bbox(original_image, expanded_bbox)
                
                field_slug = field.replace("'", "").replace(" ", "_")
                cropped_out_path = os.path.join(cropped_output_dir, f'{base_name}_cropped_{field_slug}.jpg')
                cropped_image.save(cropped_out_path)
                print(f"Saved cropped '{field}' to: {cropped_out_path}")

            except ValueError as e:
                print(f"Could not process field '{field}': {e}")
            except Exception as e:
                print(f"An unexpected error occurred while cropping '{field}': {e}")
    
    except Exception as e:
        print(f"FATAL: Error processing {args.image_path}: {e}")

    print(f"\nProcessing finished!")

if __name__ == "__main__":
    main()

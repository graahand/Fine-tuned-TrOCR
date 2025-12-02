import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
import json
import re
from datetime import datetime

def setup_model_and_tokenizer(model_path):
    """Loads the MiniCPM-AWQ model and tokenizer."""
    print(f"Loading AWQ model: {model_path}...")
    torch.manual_seed(100)
    
    # AWQ models are loaded with AutoModelForCausalLM and device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        device_map="auto",
        use_flash_attention_2=False # Add this line to bypass the AWQ integration issue
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Model loaded successfully.")
    return model, tokenizer

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

def get_full_extraction(image, prompt_text, model, tokenizer):
    """Runs inference to extract all fields as a JSON object."""
    msgs = [{'role': 'user', 'content': [image, prompt_text]}]
    # Note: The chat method might be slightly different for CausalLM vs AutoModel
    # but we will try with the same interface first.
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=False,
        stream=False
    )
    return answer

def get_and_crop_field(image, field_name, model, tokenizer):
    """Runs grounding inference for a specific field and returns the cropped image."""
    question = f'Please provide the bounding box coordinate of the region this sentence describes: <ref>{field_name}</ref>'
    msgs = [{'role': 'user', 'content': [question, image]}]
    
    with torch.inference_mode():
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=1024,
            max_inp_length=8192,
            use_image_id=True,
        )
    
    print(f"Grounding model output for '{field_name}': {res}")
    bbox = extract_bbox_from_response(res)
    cropped_image = crop_image_with_bbox(image, bbox)
    return cropped_image

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Extract data and localize fields from a single image using MiniCPM-AWQ.")
    parser.add_argument("image_path", type=str, help="Path to the single image file.")
    args = parser.parse_args()
    image_path = args.image_path

    if not os.path.isfile(image_path):
        print(f"Error: The specified image file does not exist: {image_path}")
        return

    # --- Model and Prompt Setup ---
    model_path = 'openbmb/MiniCPM-V-4_5-AWQ'
    model, tokenizer = setup_model_and_tokenizer(model_path)
    
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

    # --- Dynamic Output Directories ---
    model_name_slug = model_path.split('/')[-1]
    current_date = datetime.now().strftime("%B_%d").lower()
    output_folder = f"output_{model_name_slug}_{current_date}"
    cropped_output_dir = os.path.join(output_folder, 'cropped_numbers')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(cropped_output_dir, exist_ok=True)
    print(f"JSON output will be saved to: {output_folder}")
    print(f"Cropped images will be saved to: {cropped_output_dir}")

    # --- Image Processing ---
    print(f"\n{'='*20} Processing image: {image_path} {'='*20}")
    base_name, ext = os.path.splitext(os.path.basename(image_path))

    try:
        original_image = Image.open(image_path).convert('RGB')

        # --- Step 1: Full OCR Extraction ---
        print("--- Step 1: Extracting all fields... ---")
        extraction_result = get_full_extraction(original_image, extraction_prompt, model, tokenizer)
        print(f"Extraction output: {extraction_result}")

        # Save the raw JSON output
        output_filename = f"{base_name}_output.json"
        output_filepath = os.path.join(output_folder, output_filename)
        
        try:
            json_str = extraction_result.strip()
            if json_str.startswith("```json"):
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif json_str.startswith("```"):
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
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
                cropped_image = get_and_crop_field(original_image, field, model, tokenizer)
                
                field_slug = field.replace("'", "").replace(" ", "_")
                cropped_out_path = os.path.join(cropped_output_dir, f'{base_name}_cropped_{field_slug}.jpg')
                cropped_image.save(cropped_out_path)
                print(f"Saved cropped '{field}' to: {cropped_out_path}")

            except ValueError as e:
                print(f"Could not process field '{field}': {e}")
            except Exception as e:
                print(f"An unexpected error occurred while cropping '{field}': {e}")
    
    except Exception as e:
        print(f"FATAL: Error processing {image_path}: {e}")

    print(f"\nProcessing finished!")

if __name__ == "__main__":
    main()

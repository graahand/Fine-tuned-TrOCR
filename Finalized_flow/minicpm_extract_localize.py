import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
import time
import argparse
import json
import re
from datetime import datetime

def setup_model_and_tokenizer(model_path):
    """Loads the MiniCPM model and tokenizer."""
    print("Loading MiniCPM-V-4_5 model...")
    torch.manual_seed(100)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        attn_implementation='sdpa', 
        torch_dtype=torch.bfloat16
    )
    model = model.eval().cuda()
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
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=False,
        # enable_thinking=True,
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
    parser = argparse.ArgumentParser(description="Extract data and localize fields from a folder of images using MiniCPM.")
    parser.add_argument("images_folder", type=str, help="Path to the folder containing image files.")
    args = parser.parse_args()
    images_folder = args.images_folder

    if not os.path.isdir(images_folder):
        print(f"Error: The specified image folder does not exist: {images_folder}")
        return

    # --- Model and Prompt Setup ---
    # model_path = 'openbmb/MiniCPM-V-4_5'
    model_path = "openbmb/MiniCPM-V-4"
    model, tokenizer = setup_model_and_tokenizer(model_path)
    
    # This is the detailed extraction prompt, modified to exclude number fields.
    extraction_prompt = (
        "You are a precision-focused OCR extraction system for handwritten Nepali student forms. Analyze the provided scanned image and extract only the specified fields with strict adherence to the following rules. The 'phone number' and 'guardian's number' will be handled by a separate process, so you must ignore them.\n\n"
        "### Critical Extraction Rules\n"
        "1. **Digit Ambiguity Resolution** (Apply ONLY to email digits):\n"
        "   - '1' vs '7': '1' has straight vertical line; '7' has horizontal bar\n"
        "   - '5' vs '3': '5' has flat top; '3' has rounded top\n\n"
        "2. **Field-Specific Requirements**:\n"
        "   - **Email**:\n"
        "     • Must contain '@' and end with '.com', '.net', or '.org'\n"
        "     • Correct common spacing errors (e.g., 'gmail .com' → 'gmail.com')\n"
        "     • Reject if domain isn't @gmail.com/@icloud.com (e.g., 'yahoo' → \"\")\n"
        "   - **Interested Course**:\n"
        "     • Only accept 'GPPC', 'SEP', or 'Not interested'\n"
        "     • If multiple boxes checked, prioritize the non-crossed option\n"
        "   - **Class**:\n"
        "     • Convert Roman numerals I-XII to Arabic (e.g., 'XII'→'12')\n"
        "     • Reject non-numeric values (e.g., 'bachelor's' → \"\")\n"
        "   - **Address**:\n"
        "     • Standardize only these known Nepali terms: Ktm→Kathmandu, Btl→Butwal, Ltp→Lalitpur\n"
        "     • Do NOT correct other spellings (e.g., 'Kapan' remains 'Kapan')\n\n"
        "3. **Mandatory Exclusions**:\n"
        "   • DO NOT EXTRACT 'phone number' or 'guardian's number'.\n"
        "   • Ignore any text with horizontal strikethroughs or erasure marks\n"
        "   • Skip fields with illegible handwriting.\n"
        "   • Never invent values - use an empty string (\"\") for uncertain entries.\n\n"
        "### Output Requirements\n"
        "Return ONLY a JSON object with these exact keys. Leave 'phone number' and 'guardian's number' as empty strings:\n"
        "{\n"
        "    \"Name\": \"\",\n"
        "    \"class\": \"\",\n"
        "    \"address\": \"\",\n"
        "    \"faculty\": \"\",\n"
        "    \"phone number\": \"\",\n"
        "    \"guardian's number\": \"\",\n"
        "    \"email\": \"\",\n"
        "    \"interested course\": \"\",\n"
        "    \"school name\": \"\",\n"
        "    \"who gave seminar\": \"\",\n"
        "    \"rate us\": \"\"\n"
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

    # --- Image Processing Loop ---
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.heic'}
    image_files = [f for f in os.listdir(images_folder) if os.path.splitext(f)[1].lower() in supported_extensions]

    if not image_files:
        print("No supported image files found in the specified folder.")
        return

    for i, image_file in enumerate(image_files, 1):
        print(f"\n{'='*20} Processing image {i}/{len(image_files)}: {image_file} {'='*20}")
        image_path = os.path.join(images_folder, image_file)
        base_name, ext = os.path.splitext(image_file)

        try:
            original_image = Image.open(image_path).convert('RGB')

            # --- Step 1: Full OCR Extraction ---
            print("--- Step 1: Extracting all fields... ---")
            extraction_result = get_full_extraction(original_image, extraction_prompt, model, tokenizer)
            print(f"Extraction output: {extraction_result}")

            # Save the raw JSON output
            output_filename = f"{base_name}_output.json"
            output_filepath = os.path.join(output_folder, output_filename)
            
            # Clean and save JSON
            try:
                json_str = extraction_result.strip()
                # Check if the output is wrapped in markdown-style code blocks
                if json_str.startswith("```json"):
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif json_str.startswith("```"):
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                # Now, parse the cleaned string
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
                    
                    # Save the cropped image
                    field_slug = field.replace("'", "").replace(" ", "_")
                    cropped_out_path = os.path.join(cropped_output_dir, f'{base_name}_cropped_{field_slug}.jpg')
                    cropped_image.save(cropped_out_path)
                    print(f"Saved cropped '{field}' to: {cropped_out_path}")

                except ValueError as e:
                    print(f"Could not process field '{field}' for image {image_file}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while cropping '{field}' for {image_file}: {e}")
        
        except Exception as e:
            print(f"FATAL: Error processing {image_file}: {e}")

    print(f"\nAll images processed successfully!")

if __name__ == "__main__":
    main()

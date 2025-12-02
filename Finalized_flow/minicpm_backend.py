import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
import time
import argparse
import json
import re
from datetime import datetime

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process a folder of images with the MiniCPM-V-4_5 Model.")
    parser.add_argument("images_folder", type=str, help="Path to the folder containing the image files.")
    args = parser.parse_args()
    images_folder = args.images_folder

    if not os.path.isdir(images_folder):
        print(f"Error: The specified image folder does not exist: {images_folder}")
        return

    # --- Model and Tokenizer Loading ---
    torch.manual_seed(100)
    print("Loading MiniCPM-V-4_5 model...")
    model_path = 'openbmb/MiniCPM-V-4_5'
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        attn_implementation='sdpa', 
        torch_dtype=torch.bfloat16
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded successfully.")

    # --- Dynamic Output Directory ---
    model_name_slug = model_path.split('/')[-1] # e.g., MiniCPM-V-4_5
    current_date = datetime.now().strftime("%B_%d").lower() # e.g., november_23
    output_folder = f"output_{model_name_slug}_{current_date}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output will be saved to: {output_folder}")

    # Get all image files from the folder
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.heic'}
    image_files = [f for f in os.listdir(images_folder) if 
                   os.path.splitext(f)[1].lower() in supported_extensions]

    if not image_files:
        print("No supported image files found in the specified folder.")
        return

    # Define the prompt text
    prompt_text = (
    "You are a precision-focused OCR extraction system for handwritten Nepali student forms. Analyze the provided scanned image and extract only the specified fields with strict adherence to the following rules. Ignore all crossed-out text (marked with horizontal lines/erasures) and prioritize accuracy for phone numbers, guardian's numbers, and emails above all other fields.\n\n" 
    "### Critical Extraction Rules\n"
    "1. **Digit Ambiguity Resolution** (Apply ONLY to phone/guardian numbers and email digits):\n"
    "   - '4' vs '0': In phone numbers, '0' is more common in 2nd/3rd positions (e.g., 980XXXXXXX), while '4' appears in middle positions\n"
    "   - '3' vs '8': '8' is more frequent in final digits of Nepali numbers (e.g., XXXXXXXX8); check stroke continuity (8 has closed loop)\n"
    "   - '1' vs '7': '1' has straight vertical line; '7' has horizontal bar\n"
    "   - '5' vs '3': '5' has flat top; '3' has rounded top\n"
    "   - When uncertain, default to the digit that maintains a valid 10-digit Nepali phone number structure (98X-XXXXXXX or 97X-XXXXXXX)\n\n"

    "2. **Field-Specific Requirements**:\n"
    "   - **Phone/Guardian Numbers**:\n"
    "     • Must be 10 digits (Nepali format). If 9 digits, assume missing '9' at start\n"
    "     • Reject entries with letters (e.g., 'G' in '970329360G' → treat as invalid → return \"\")\n"
    "     • If ambiguous digits persist after rule application, return \"\"\n"
    "   - **Email**:\n"
    "     • Must contain '@' and end with '.com', '.net', or '.org'\n"
    "     • Correct common spacing errors (e.g., 'gmail .com' → 'gmail.com')\n"
    "     • Reject if domain isn't @gmail.com/@icloud.com (e.g., 'yahoo' → \"\")\n"
    "   - **Interested Course**:\n"
    "     • Only accept 'GPPC', 'SEP', or 'Not interested'\n"
    "     • If multiple boxes checked, prioritize the non-crossed option\n"
    "     • Crossed checkboxes (e.g., 'GPPC' with strikethrough) are invalid\n"
    "   - **Class**:\n"
    "     • Convert Roman numerals I-XII to Arabic (e.g., 'XII'→'12')\n"
    "     • Reject non-numeric values (e.g., 'bachelor's' → \"\")\n"
    "   - **Address**:\n"
    "     • Standardize only these known Nepali terms: Ktm→Kathmandu, Btl→Butwal, Ltp→Lalitpur\n"
    "     • Do NOT correct other spellings (e.g., 'Kapan' remains 'Kapan')\n\n"

    "3. **Mandatory Exclusions**:\n"
    "   • Ignore any text with horizontal strikethroughs or erasure marks\n"
    "   • Skip fields with illegible handwriting after 3 context checks\n"
    "   • Never invent values - empty string (\"\") for uncertain entries\n\n"

    "### Output Requirements\n"
    "Return ONLY a JSON object with these exact keys. Prioritize phone/guardian/email accuracy:\n"
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

    "### Verification Protocol (Apply Before Output)\n"
    "1. For phone/guardian numbers: Confirm 10-digit count and valid Nepali prefix (97/98)\n"
    "2. For email: Validate domain structure after space correction\n"
    "3. For all fields: Double-check against crossed-out content\n"
    "4. If any priority field (phone/guardian/email) fails verification → set to \"\"\n\n"

    "Begin extraction now. Remember: When in doubt, prioritize accuracy over completeness. Never hallucinate."

) 

    # Track total time
    total_start_time = time.time()

    # Process each image in the folder
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {image_file}")
        
        # Track time per image
        image_start_time = time.time()
        
        try:
            # Full path to the image
            image_path = os.path.join(images_folder, image_file)
            image = Image.open(image_path).convert('RGB')
            
            # Construct messages for MiniCPM
            msgs = [{'role': 'user', 'content': [image, prompt_text]}]

            # Inference
            # We use stream=False to get the full response at once
            answer = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                # enable_thinking=True,  # Deterministic generation
                stream=False
            )
            
            output_text = answer
            
            # Calculate time taken for this image
            image_end_time = time.time()
            image_time_taken = image_end_time - image_start_time
            
            print(f"Output for {image_file}: {output_text}")
            print(f"Time taken for {image_file}: {image_time_taken:.2f} seconds")

            # Extract the image name without extension to create the output filename
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            output_filename = f"{image_name}_output_minicpm.txt"
            output_filepath = os.path.join(output_folder, output_filename)

            # Save the output to a text file in the output folder
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(output_text)

            print(f"Output saved to: {output_filepath}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

        print("-" * 50)  # Separator between images

    # Calculate total time
    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time

    print(f"All images processed successfully!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total time taken: {total_time_taken:.2f} seconds")
    print(f"Average time per image: {total_time_taken / len(image_files):.2f} seconds" if image_files else "No images processed")
    print(f"Output files saved in folder: {output_folder}")

if __name__ == "__main__":
    main()

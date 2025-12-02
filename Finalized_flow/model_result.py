from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import os
import time
import argparse
from datetime import datetime

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process a folder of images with a Vision Language Model.")
    parser.add_argument("images_folder", type=str, help="Path to the folder containing the image files.")
    args = parser.parse_args()
    images_folder = args.images_folder

    if not os.path.isdir(images_folder):
        print(f"Error: The specified image folder does not exist: {images_folder}")
        return

    # --- Model and Processor Loading ---
    model_path_str = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading model: {model_path_str}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path_str,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path_str)
    print("Model loaded successfully.")

    # --- Dynamic Output Directory ---
    model_name_slug = model_path_str.split('/')[-1] # e.g., Qwen3-VL-8B-Instruct
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

    prompt_text = (
    "You are a precision-focused OCR extraction system for handwritten Nepali student forms. Analyze the provided scanned image and extract only the specified fields. "
    "Your highest priority is to NEVER guess digits. It is better to return a partial result with uncertainty markers than a hallucinated complete number.\n\n"

    "### Critical Extraction Rules\n"
    "1. **Digit Ambiguity Resolution (The '?' Protocol)**:\n"
    "   - Analyze pixel features strictly. If a digit is smeared, overwritten, or ambiguous, you must insert a '?' for that specific character.\n"
    "   - **Strictly FORBIDDEN:** Do not 'guess' a number just to satisfy a 10-digit format. Do not auto-complete based on probability.\n"
    "   - **Visual Heuristics (Use these to identify, but if still unsure, use '?'):**\n"
    "     • '4' vs '0': Look for the open top of '4'. If completely round/closed, it is '0'. If ambiguous -> '?'\n"
    "     • '3' vs '8': Look for the closed loop of '8'. If the loops are open or disconnected -> '3'. If ambiguous -> '?'\n"
    "     • '1' vs '7': Look for the horizontal top bar. No bar -> '1'. Bar -> '7'. If ambiguous -> '?'\n"
    "     • '5' vs '3': Look for the flat top ('5') vs rounded top ('3'). If ambiguous -> '?'\n\n"

    "2. **Field-Specific Requirements**:\n"
    "   - **Phone/Guardian Numbers**:\n"
    "     • Standard length is 10 digits. However, you must output exactly what you see.\n"
    "     • Example: If the 4th digit is blurry, output '984?123456'.\n"
    "     • Reject entries containing letters (e.g., 'G').\n"
    "     • If the number has missing digits (less than 10) and no '?' markers, assume missing '9' at start ONLY if it creates a valid sequence.\n"
    "   - **Email**:\n"
    "     • Must contain '@'. \n"
    "     • If a letter in the username is illegible, use '?'. (e.g. 'sanj?v@gmail.com').\n"
    "     • Auto-correct spacing (e.g., 'gmail .com' → 'gmail.com') but do NOT auto-correct the username characters.\n"
    "   - **Interested Course**:\n"
    "     • Only accept 'GPPC', 'SEP', or 'Not interested'\n"
    "     • If multiple boxes checked, prioritize the non-crossed option.\n"
    "   - **Class**:\n"
    "     • Convert Roman numerals I-XII to Arabic (e.g., 'XII'→'12')\n"
    "     • Reject non-numeric values.\n"
    "   - **Address**:\n"
    "     • Standardize only these known Nepali terms: Ktm→Kathmandu, Btl→Butwal, Ltp→Lalitpur\n"
    "     • Do NOT correct other spellings.\n\n"

    "3. **Mandatory Exclusions**:\n"
    "   • Ignore any text with horizontal strikethroughs or erasure marks\n"
    "   • Never invent values - return empty string (\"\") if the field is completely empty.\n\n"

    "### Output Requirements\n"
    "Return ONLY a JSON object with these exact keys. \n"
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

    "### Verification Protocol\n"
    "1. For phone/guardian numbers: If the output contains '?', DO NOT change it to a number. Leave it as '?'.\n"
    "2. For email: Validate domain structure (@gmail.com, etc), but allow '?' in the local part (before @).\n"
    "3. If a field is completely unreadable (more than 50% '?'), set it to \"\".\n\n"

    "Begin extraction now. Prioritize visual fidelity over formatting."
)
    # Track total time
    total_start_time = time.time()

    # Process each image in the folder
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {image_file}")
        
        # Track time per image
        image_start_time = time.time()
        
        # Full path to the image
        image_path = os.path.join(images_folder, image_file)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt_text},
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
        generated_ids = model.generate(**inputs, max_new_tokens=300, return_dict_in_generate=True, output_scores=True).sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Calculate time taken for this image
        image_end_time = time.time()
        image_time_taken = image_end_time - image_start_time
        
        print(f"Output for {image_file}: {output_text}")
        print(f"Time taken for {image_file}: {image_time_taken:.2f} seconds")

        # Extract the image name without extension to create the output filename
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        output_filename = f"{image_name}_output_8b.txt"
        output_filepath = os.path.join(output_folder, output_filename)

        # Save the output to a text file in the output folder
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # batch_decode returns a list; write the first (and only) element as raw text
            content_to_write = output_text[0] if isinstance(output_text, list) and output_text else str(output_text)
            f.write(content_to_write)

        print(f"Output saved to: {output_filepath}")
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
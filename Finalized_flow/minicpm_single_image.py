import torch
from torch.nn.functional import softmax
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
import time
import argparse
import json
import re
from datetime import datetime

def calculate_confidence_for_field(field_value, full_text, token_confidences):
    """
    Calculates the average confidence for a specific field value within the full text.
    """
    if not field_value or not isinstance(field_value, str):
        return 100.0
    
    # Normalize text for search
    # field_value = field_value.strip()
    
    start_index = full_text.find(field_value)
    if start_index == -1:
        return 0.0
    
    end_index = start_index + len(field_value)
    
    current_char_pos = 0
    relevant_confidences = []
    
    for token_text, score in token_confidences:
        token_len = len(token_text)
        token_start = current_char_pos
        token_end = current_char_pos + token_len
        
        # Check overlap
        if token_end > start_index and token_start < end_index:
            relevant_confidences.append(score)
            
        current_char_pos += token_len
        
        if current_char_pos >= end_index:
            break
            
    if not relevant_confidences:
        return 0.0
        
    return sum(relevant_confidences) / len(relevant_confidences) * 100

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process a single image with the MiniCPM-V-4_5 Model.")
    parser.add_argument("image_path", type=str, help="Path to the single image file.")
    args = parser.parse_args()
    image_path = args.image_path

    if not os.path.exists(image_path):
        print(f"Error: The specified image file does not exist: {image_path}")
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

    # Define the prompt text
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


    print(f"Processing image: {image_path}")
    
    # Track time
    start_time = time.time()
    
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Construct messages for MiniCPM
        msgs = [{'role': 'user', 'content': [image, prompt_text]}]

        # Inference
        # Try to get scores if possible
        try:
            answer = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                stream=False,
                # enable_thinking=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        except TypeError:
            # Fallback if arguments are not supported
            print("Warning: 'return_dict_in_generate' not supported by model.chat, falling back to default.")
            answer = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                stream=False
            )

        token_confidences = []
        output_text = ""

        if isinstance(answer, str):
            output_text = answer
        else:
            # Process GenerateOutput
            if hasattr(answer, 'sequences') and hasattr(answer, 'scores'):
                generated_ids = answer.sequences
                scores = answer.scores
                
                gen_len = len(scores)
                input_len = generated_ids.shape[1] - gen_len
                
                new_tokens = generated_ids[:, input_len:]
                
                # Decode full text
                output_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                
                # Calculate token confidences
                for i, step_logits in enumerate(scores):
                    probs = softmax(step_logits, dim=-1)
                    token_id = new_tokens[0, i].item()
                    confidence = probs[0, token_id].item()
                    token_string = tokenizer.decode([token_id])
                    token_confidences.append((token_string, confidence))
            else:
                    output_text = str(answer)

        # Parse JSON and calculate confidence
        try:
            # Clean markdown code blocks
            json_str = output_text.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            # Calculate confidence
            conf_phone = 100.0
            conf_guardian = 100.0
            
            if token_confidences:
                conf_phone = calculate_confidence_for_field(data.get("phone number"), output_text, token_confidences)
                conf_guardian = calculate_confidence_for_field(data.get("guardian's number"), output_text, token_confidences)
            
            data["confidence_phone"] = round(conf_phone, 2)
            data["confidence_guardian"] = round(conf_guardian, 2)
            
            # Mark incorrect data
            # If either number has confidence < 95, mark as incorrect
            if conf_phone < 95 or conf_guardian < 95:
                data["is_data_correct"] = False
            else:
                data["is_data_correct"] = True
                
            output_text = json.dumps(data, indent=4)
            
        except json.JSONDecodeError:
            print(f"Error decoding JSON for {image_path}")
        except Exception as e:
            print(f"Error processing confidence: {e}")
        
        # Calculate time taken
        end_time = time.time()
        time_taken = end_time - start_time
        
        print("\n" + "="*60)
        print("FINAL GENERATED OUTPUT")
        print("="*60)
        print(output_text)
        print("="*60 + "\n")
        print(f"Time taken: {time_taken:.2f} seconds")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    main()

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown
import os
import time
import argparse
import json
import re
from datetime import datetime

def extract_json_from_text(text):
    """Extract JSON from text response"""
    # First try to find JSON block
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # If no JSON found, try to parse the markdown structure
    lines = text.strip().split('\n')
    extracted_data = {}
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            # Remove markdown formatting
            line = re.sub(r'\*\*|__', '', line)  # Remove bold
            line = re.sub(r'\*|_', '', line)     # Remove italic
            
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Map common field names
            field_mapping = {
                'name': 'Name',
                'class': 'class',
                'address': 'address',
                'faculty': 'faculty',
                'phone': 'phone number',
                'phone number': 'phone number',
                'guardian': "guardian's number",
                "guardian's number": "guardian's number",
                'guardian number': "guardian's number",
                'email': 'email',
                'course': 'interested course',
                'interested course': 'interested course',
                'school': 'school name',
                'school name': 'school name',
                'seminar': 'who gave seminar',
                'who gave seminar': 'who gave seminar',
                'rating': 'rate us',
                'rate us': 'rate us'
            }
            
            mapped_key = field_mapping.get(key, key)
            extracted_data[mapped_key] = value
    
    return json.dumps(extracted_data)

def process_chandra_response(response_text):
    """Process Chandra response and convert to structured JSON"""
    
    # Standard field template
    result_template = {
        "Name": "",
        "class": "",
        "address": "",
        "faculty": "",
        "phone number": "",
        "guardian's number": "",
        "email": "",
        "interested course": "",
        "school name": "",
        "who gave seminar": "",
        "rate us": ""
    }
    
    try:
        # Try to extract JSON first
        json_str = extract_json_from_text(response_text)
        parsed_data = json.loads(json_str)
        
        # Fill template with parsed data
        for key in result_template:
            if key in parsed_data:
                result_template[key] = parsed_data[key]
            elif key.lower() in parsed_data:
                result_template[key] = parsed_data[key.lower()]
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error parsing response, extracting from text: {e}")
        
        # Fallback: extract from raw text
        lines = response_text.strip().split('\n')
        current_field = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains field information
            lower_line = line.lower()
            
            if 'name' in lower_line and ':' in line:
                result_template['Name'] = line.split(':', 1)[1].strip()
            elif 'class' in lower_line and ':' in line:
                result_template['class'] = line.split(':', 1)[1].strip()
            elif 'address' in lower_line and ':' in line:
                result_template['address'] = line.split(':', 1)[1].strip()
            elif 'faculty' in lower_line and ':' in line:
                result_template['faculty'] = line.split(':', 1)[1].strip()
            elif 'phone' in lower_line and 'guardian' not in lower_line and ':' in line:
                result_template['phone number'] = line.split(':', 1)[1].strip()
            elif 'guardian' in lower_line and ':' in line:
                result_template["guardian's number"] = line.split(':', 1)[1].strip()
            elif 'email' in lower_line and ':' in line:
                result_template['email'] = line.split(':', 1)[1].strip()
            elif ('course' in lower_line or 'interested' in lower_line) and ':' in line:
                result_template['interested course'] = line.split(':', 1)[1].strip()
            elif 'school' in lower_line and ':' in line:
                result_template['school name'] = line.split(':', 1)[1].strip()
            elif 'seminar' in lower_line and ':' in line:
                result_template['who gave seminar'] = line.split(':', 1)[1].strip()
            elif ('rate' in lower_line or 'rating' in lower_line) and ':' in line:
                result_template['rate us'] = line.split(':', 1)[1].strip()
    
    return result_template

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process a single image with the Chandra OCR Model.")
    parser.add_argument("image_path", type=str, help="Path to the single image file.")
    args = parser.parse_args()
    image_path = args.image_path

    if not os.path.exists(image_path):
        print(f"Error: The specified image file does not exist: {image_path}")
        return

    # --- Model and Processor Loading ---
    torch.manual_seed(100)
    print("Loading Chandra OCR model...")
    
    try:
        model = AutoModel.from_pretrained("datalab-to/chandra").cuda()
        model.processor = AutoProcessor.from_pretrained("datalab-to/chandra")
        print("Chandra OCR model loaded successfully.")
    except Exception as e:
        print(f"Error loading Chandra OCR model: {e}")
        print("Make sure you have installed the chandra package:")
        print("pip install chandra-ocr")
        return

    print(f"Processing image: {image_path}")
    
    # Track time
    start_time = time.time()
    
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Create batch input for Chandra
        batch = [
            BatchInputItem(
                image=image,
                prompt_type="ocr_layout"  # Use layout OCR for structured forms
            )
        ]
        
        # Run inference
        print("Running Chandra OCR inference...")
        result = generate_hf(batch, model)[0]
        
        # Parse the markdown output
        markdown_output = parse_markdown(result.raw)
        
        print("Raw Chandra Output:")
        print("-" * 50)
        print(result.raw)
        print("-" * 50)
        
        print("Parsed Markdown:")
        print("-" * 50)
        print(markdown_output)
        print("-" * 50)
        
        # Process the response to extract structured data
        structured_data = process_chandra_response(result.raw)
        
        # Add metadata
        structured_data["processing_time"] = round(time.time() - start_time, 2)
        structured_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        structured_data["model"] = "chandra-ocr"
        structured_data["raw_response"] = result.raw
        structured_data["markdown_output"] = markdown_output
        
        # Output final JSON
        output_json = json.dumps(structured_data, indent=4, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("FINAL STRUCTURED JSON OUTPUT")
        print("="*60)
        print(output_json)
        print("="*60 + "\n")
        
        # Calculate time taken
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Total processing time: {time_taken:.2f} seconds")
        
        # Save output to file
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_chandra_output.json"
        output_path = os.path.join(os.path.dirname(image_path), output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_json)
        
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
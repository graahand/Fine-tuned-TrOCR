import gradio as gr
import torch
from PIL import Image, ImageDraw
import os
import json
import pandas as pd
import numpy as np
import time
import re
import tempfile
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
import sys

# Add the Finalized_flow directory to path to import existing functions
sys.path.append('/home/museum/Fine-tuned-TrOCR/Finalized_flow')

# Try to import existing modules
try:
    from minicpm_single_image import calculate_confidence_for_field
    from compare_csv import create_diff_csv
    from count_accuracy import count_metrics
    from csv_converter import create_master_csv
    from html_embed import create_html_report
    from localize_number import model_infer_and_draw, setup_model_and_tokenizer
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

# Global variables for models
models_cache = {}

def get_ocr_prompt():
    """Get the standard OCR prompt"""
    return """You are a precision-focused OCR extraction system for handwritten Nepali student forms. Analyze the provided scanned image and extract only the specified fields with strict adherence to the following rules. Ignore all crossed-out text (marked with horizontal lines/erasures) and prioritize accuracy for phone numbers, guardian's numbers, and emails above all other fields.

### Critical Extraction Rules
1. **Digit Ambiguity Resolution** (Apply ONLY to phone/guardian numbers and email digits):
   - '4' vs '0': In phone numbers, '0' is more common in 2nd/3rd positions (e.g., 980XXXXXXX), while '4' appears in middle positions
   - '3' vs '8': '8' is more frequent in final digits of Nepali numbers (e.g., XXXXXXXX8); check stroke continuity (8 has closed loop)
   - '1' vs '7': '1' has straight vertical line; '7' has horizontal bar
   - '5' vs '3': '5' has flat top; '3' has rounded top
   - When uncertain, default to the digit that maintains a valid 10-digit Nepali phone number structure (98X-XXXXXXX or 97X-XXXXXXX)

2. **Field-Specific Requirements**:
   - **Phone/Guardian Numbers**:
     • Must be 10 digits (Nepali format). If 9 digits, assume missing '9' at start
     • Reject entries with letters (e.g., 'G' in '970329360G' → treat as invalid → return "")
     • If ambiguous digits persist after rule application, return ""
   - **Email**:
     • Must contain '@' and end with '.com', '.net', or '.org'
     • Correct common spacing errors (e.g., 'gmail .com' → 'gmail.com')
     • Reject if domain isn't @gmail.com/@icloud.com (e.g., 'yahoo' → "")
   - **Interested Course**:
     • Only accept 'GPPC', 'SEP', or 'Not interested'
     • If multiple boxes checked, prioritize the non-crossed option
     • Crossed checkboxes (e.g., 'GPPC' with strikethrough) are invalid
   - **Class**:
     • Convert Roman numerals I-XII to Arabic (e.g., 'XII'→'12')
     • Reject non-numeric values (e.g., 'bachelor's' → "")
   - **Address**:
     • Standardize only these known Nepali terms: Ktm→Kathmandu, Btl→Butwal, Ltp→Lalitpur
     • Do NOT correct other spellings (e.g., 'Kapan' remains 'Kapan')

3. **Mandatory Exclusions**:
   • Ignore any text with horizontal strikethroughs or erasure marks
   • Skip fields with illegible handwriting after 3 context checks
   • Never invent values - empty string ("") for uncertain entries

### Output Requirements
Return ONLY a JSON object with these exact keys. Prioritize phone/guardian/email accuracy:
{
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

### Verification Protocol (Apply Before Output)
1. For phone/guardian numbers: Confirm 10-digit count and valid Nepali prefix (97/98)
2. For email: Validate domain structure after space correction
3. For all fields: Double-check against crossed-out content
4. If any priority field (phone/guardian/email) fails verification → set to ""

Begin extraction now. Remember: When in doubt, prioritize accuracy over completeness. Never hallucinate."""

def load_minicpm_model():
    """Load MiniCPM model"""
    if 'minicpm' not in models_cache:
        try:
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
            models_cache['minicpm'] = (model, tokenizer)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading MiniCPM model: {e}")
            return None, None
    
    return models_cache['minicpm']

def extract_single_image_ocr(image):
    """Extract OCR data from single image"""
    if image is None:
        return None, "No image provided"
    
    try:
        model, tokenizer = load_minicpm_model()
        if model is None:
            return None, "Failed to load model"
        
        # Process image
        msgs = [{'role': 'user', 'content': [image, get_ocr_prompt()]}]
        
        start_time = time.time()
        answer = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=False,
            # enable_thinking=True,
            stream=False
        )
        processing_time = time.time() - start_time
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', answer, re.DOTALL)
        if json_match:
            try:
                extracted_data = json.loads(json_match.group(0))
                
                # Add metadata
                result = {
                    "extracted_data": extracted_data,
                    "processing_time": round(processing_time, 2),
                    "raw_response": answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return result, "Extraction completed successfully"
                
            except json.JSONDecodeError as e:
                return None, f"JSON parsing error: {e}"
        else:
            return None, f"No JSON found in response. Raw response: {answer[:200]}..."
            
    except Exception as e:
        return None, f"Error during processing: {str(e)}"

def process_multiple_images(files):
    """Process multiple images and return results"""
    if not files:
        return None, "No files provided", None
    
    results = []
    total_files = len(files)
    
    for i, file in enumerate(files):
        try:
            # Open image
            image = Image.open(file.name)
            
            # Process image
            result, message = extract_single_image_ocr(image)
            
            if result:
                # Add file info
                result["file_name"] = os.path.basename(file.name)
                result["file_index"] = i
                results.append(result)
            else:
                # Add error record
                results.append({
                    "file_name": os.path.basename(file.name),
                    "file_index": i,
                    "error": message
                })
        
        except Exception as e:
            results.append({
                "file_name": os.path.basename(file.name) if hasattr(file, 'name') else f"file_{i}",
                "file_index": i,
                "error": str(e)
            })
    
    # Create CSV from results
    csv_records = []
    for result in results:
        if "extracted_data" in result:
            record = result["extracted_data"].copy()
            record["file_name"] = result["file_name"]
            record["processing_time"] = result.get("processing_time", "")
            csv_records.append(record)
    
    if csv_records:
        df = pd.DataFrame(csv_records)
        csv_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
        df.to_csv(csv_path, index=False)
        
        summary = f"Processed {total_files} files. {len(csv_records)} successful extractions."
        return results, summary, csv_path
    else:
        return results, f"Processed {total_files} files. No successful extractions.", None

def localize_and_crop_phone(image):
    """Localize phone numbers in image"""
    if image is None:
        return None, None, "No image provided"
    
    try:
        model, tokenizer = load_minicpm_model()
        if model is None:
            return None, None, "Failed to load model"
        
        # Grounding prompt
        question = """Please identify and provide the bounding box coordinates for any phone numbers in this image. Phone numbers in Nepal typically start with 97 or 98 and are 10 digits long. Return the coordinates in the format <box>x1 y1 x2 y2</box> where coordinates are normalized to 1000."""
        
        msgs = [{'role': 'user', 'content': [question, image]}]
        
        with torch.inference_mode():
            response = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                # enable_thinking=True,
                stream=False
            )
        
        # Extract bounding box
        match = re.search(r"<box>([\d\s]+)</box>", response)
        if match:
            bbox_str = match.group(1)
            bbox = list(map(int, bbox_str.strip().split()))
            
            # Draw bounding box
            w, h = image.size
            x1 = int(bbox[0] / 1000 * w)
            y1 = int(bbox[1] / 1000 * h)
            x2 = int(bbox[2] / 1000 * w)
            y2 = int(bbox[3] / 1000 * h)
            
            # Create image with bounding box
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            
            # Crop phone number region
            cropped_image = image.crop((x1, y1, x2, y2))
            
            return draw_image, cropped_image, f"Phone number found and localized. Confidence: {response}"
        else:
            return image, None, f"No phone number detected. Response: {response}"
            
    except Exception as e:
        return None, None, f"Error during localization: {str(e)}"

def compare_two_csvs(file1, file2):
    """Compare two CSV files"""
    if not file1 or not file2:
        return None, "Please provide both CSV files"
    
    try:
        # Read files
        df1 = pd.read_csv(file1.name).fillna('')
        df2 = pd.read_csv(file2.name).fillna('')
        
        # Get key column (assume first column or image-related column)
        key_col = None
        for col in df1.columns:
            if 'image' in col.lower() or 'file' in col.lower():
                key_col = col
                break
        
        if key_col is None:
            key_col = df1.columns[0]
        
        if key_col not in df2.columns:
            return None, f"Key column '{key_col}' not found in second file"
        
        # Merge and compare
        merged = pd.merge(df1, df2, on=key_col, how='outer', suffixes=('_file1', '_file2'))
        
        # Create diff report
        diff_data = []
        common_cols = [c for c in df1.columns if c in df2.columns and c != key_col]
        
        for _, row in merged.iterrows():
            diff_record = {key_col: row[key_col]}
            
            for col in common_cols:
                val1 = row.get(f'{col}_file1', '[MISSING]')
                val2 = row.get(f'{col}_file2', '[MISSING]')
                
                diff_record[f'{col}_file1'] = val1
                diff_record[f'{col}_file2'] = val2
                diff_record[f'{col}_match'] = 'YES' if val1 == val2 else 'NO'
        
        # Save diff report
        diff_df = pd.DataFrame(diff_data)
        diff_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
        diff_df.to_csv(diff_path, index=False)
        
        # Calculate summary
        total_comparisons = len(diff_data) * len(common_cols)
        matches = sum(1 for record in diff_data for col in common_cols if record.get(f'{col}_match') == 'YES')
        
        summary = f"""Comparison Summary:
- Total records: {len(diff_data)}
- Fields compared: {len(common_cols)}
- Total comparisons: {total_comparisons}
- Matches: {matches}
- Differences: {total_comparisons - matches}
- Match rate: {(matches/total_comparisons*100):.1f}%"""
        
        return diff_path, summary
        
    except Exception as e:
        return None, f"Error comparing files: {str(e)}"

def calculate_csv_accuracy(file, row_limit):
    """Calculate accuracy metrics from CSV"""
    if not file:
        return None, "Please provide a CSV file"
    
    try:
        df = pd.read_csv(file.name)
        df_subset = df.iloc[:row_limit] if row_limit > 0 else df
        
        metrics = []
        total_true = 0
        total_false = 0
        
        for col in df_subset.columns:
            if "Unnamed" in col or col.lower() in ['file_name', 'processing_time', 'timestamp']:
                continue
            
            series = df_subset[col].astype(str).str.strip().str.upper()
            
            true_count = (series == 'TRUE').sum()
            false_count = (series == 'FALSE').sum()
            blank_count = df_subset[col].isna().sum() + (df_subset[col] == '').sum()
            
            total_entries = len(df_subset)
            accuracy = (true_count / (true_count + false_count) * 100) if (true_count + false_count) > 0 else 0
            
            metrics.append({
                'Field': col,
                'True Count': true_count,
                'False Count': false_count,
                'Blank Count': blank_count,
                'Total': total_entries,
                'Accuracy (%)': round(accuracy, 2)
            })
            
            total_true += true_count
            total_false += false_count
        
        metrics_df = pd.DataFrame(metrics)
        
        overall_accuracy = (total_true / (total_true + total_false) * 100) if (total_true + total_false) > 0 else 0
        
        summary = f"""Analysis Summary:
Rows analyzed: {len(df_subset)}
Total True values: {total_true}
Total False values: {total_false}
Overall Accuracy: {overall_accuracy:.2f}%"""
        
        return metrics_df, summary
        
    except Exception as e:
        return None, f"Error analyzing CSV: {str(e)}"

def generate_html_report(csv_file):
    """Generate HTML report from CSV"""
    if not csv_file:
        return None, "Please provide a CSV file"
    
    try:
        df = pd.read_csv(csv_file.name).fillna('')
        
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OCR Extraction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                h1 {{ color: #333; text-align: center; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                .record {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #fafafa; }}
                .field {{ margin: 8px 0; }}
                .field-name {{ font-weight: bold; color: #555; display: inline-block; width: 150px; }}
                .field-value {{ color: #333; }}
                .metadata {{ color: #888; font-size: 0.9em; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>OCR Extraction Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Records: {len(df)}</p>
        """
        
        for idx, row in df.iterrows():
            html_content += f'<div class="record"><h3>Record {idx + 1}</h3>'
            
            # Main fields
            main_fields = ['Name', 'phone number', "guardian's number", 'email', 'class', 'address']
            for field in main_fields:
                if field in row:
                    value = row[field] if pd.notna(row[field]) and row[field] != '' else '<em>Not provided</em>'
                    html_content += f'<div class="field"><span class="field-name">{field}:</span> <span class="field-value">{value}</span></div>'
            
            # Other fields
            other_fields = [col for col in row.index if col not in main_fields and col not in ['file_name', 'processing_time', 'timestamp']]
            if other_fields:
                html_content += '<hr>'
                for field in other_fields:
                    value = row[field] if pd.notna(row[field]) and row[field] != '' else '<em>Not provided</em>'
                    html_content += f'<div class="field"><span class="field-name">{field}:</span> <span class="field-value">{value}</span></div>'
            
            # Metadata
            if 'processing_time' in row:
                html_content += f'<div class="metadata">Processing time: {row.get("processing_time", "N/A")} seconds</div>'
            
            html_content += '</div>'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML
        html_path = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8').name
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path, f"HTML report generated successfully with {len(df)} records"
        
    except Exception as e:
        return None, f"Error generating HTML report: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="OCR Processing Suite") as demo:
    gr.Markdown("# OCR Processing Suite")
    gr.Markdown("Comprehensive OCR processing toolkit for Nepali student forms")
    
    with gr.Tabs():
        # Single Image Processing
        with gr.TabItem("Single Image"):
            gr.Markdown("### Extract structured data from a single image")
            
            with gr.Row():
                with gr.Column(scale=1):
                    single_image = gr.Image(type="pil", label="Upload Image")
                    single_process_btn = gr.Button("Process Image", variant="primary")
                
                with gr.Column(scale=2):
                    single_result = gr.JSON(label="Extraction Results")
                    single_status = gr.Textbox(label="Status", lines=2, interactive=False)
            
            single_process_btn.click(
                extract_single_image_ocr,
                inputs=[single_image],
                outputs=[single_result, single_status]
            )
        
        # Batch Processing
        with gr.TabItem("Batch Processing"):
            gr.Markdown("### Process multiple images and export results")
            
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(file_count="multiple", label="Upload Images", file_types=["image"])
                    batch_process_btn = gr.Button("Process All Images", variant="primary")
                
                with gr.Column():
                    batch_results = gr.JSON(label="Batch Results")
                    batch_status = gr.Textbox(label="Processing Status", lines=3, interactive=False)
                    batch_csv = gr.File(label="Download CSV Results")
            
            batch_process_btn.click(
                process_multiple_images,
                inputs=[batch_files],
                outputs=[batch_results, batch_status, batch_csv]
            )
        
        # Phone Localization
        with gr.TabItem("Phone Localization"):
            gr.Markdown("### Detect and crop phone number regions")
            
            with gr.Row():
                with gr.Column():
                    loc_image = gr.Image(type="pil", label="Upload Image")
                    loc_btn = gr.Button("Localize Phone Numbers", variant="primary")
                
                with gr.Column():
                    bbox_image = gr.Image(type="pil", label="Image with Bounding Box")
                    cropped_phone = gr.Image(type="pil", label="Cropped Phone Number")
                    loc_status = gr.Textbox(label="Localization Status", lines=3, interactive=False)
            
            loc_btn.click(
                localize_and_crop_phone,
                inputs=[loc_image],
                outputs=[bbox_image, cropped_phone, loc_status]
            )
        
        # CSV Analysis
        with gr.TabItem("Analysis"):
            gr.Markdown("### Analyze CSV files and calculate metrics")
            
            with gr.Row():
                with gr.Column():
                    analysis_csv = gr.File(label="Upload CSV File", file_types=[".csv"])
                    row_limit = gr.Number(label="Row Limit (0 for all)", value=100, minimum=0)
                    analyze_btn = gr.Button("Analyze CSV", variant="primary")
                
                with gr.Column():
                    metrics_table = gr.Dataframe(label="Accuracy Metrics")
                    analysis_summary = gr.Textbox(label="Summary", lines=5, interactive=False)
            
            analyze_btn.click(
                calculate_csv_accuracy,
                inputs=[analysis_csv, row_limit],
                outputs=[metrics_table, analysis_summary]
            )
        
        # CSV Comparison
        with gr.TabItem("Compare CSVs"):
            gr.Markdown("### Compare two CSV files and generate difference report")
            
            with gr.Row():
                with gr.Column():
                    csv1 = gr.File(label="First CSV File", file_types=[".csv"])
                    csv2 = gr.File(label="Second CSV File", file_types=[".csv"])
                    compare_btn = gr.Button("Compare Files", variant="primary")
                
                with gr.Column():
                    diff_csv = gr.File(label="Download Difference Report")
                    compare_summary = gr.Textbox(label="Comparison Summary", lines=8, interactive=False)
            
            compare_btn.click(
                compare_two_csvs,
                inputs=[csv1, csv2],
                outputs=[diff_csv, compare_summary]
            )
        
        # HTML Report
        with gr.TabItem("HTML Report"):
            gr.Markdown("### Generate visual HTML reports from CSV data")
            
            with gr.Row():
                with gr.Column():
                    report_csv = gr.File(label="Upload CSV File", file_types=[".csv"])
                    report_btn = gr.Button("Generate HTML Report", variant="primary")
                
                with gr.Column():
                    html_report = gr.File(label="Download HTML Report")
                    report_status = gr.Textbox(label="Report Status", lines=3, interactive=False)
            
            report_btn.click(
                generate_html_report,
                inputs=[report_csv],
                outputs=[html_report, report_status]
            )
    
    gr.Markdown("""
    ### Quick Guide
    
    **Single Image**: Upload an image to extract structured OCR data  
    **Batch Processing**: Upload multiple images and get CSV results  
    **Phone Localization**: Automatically detect phone number regions  
    **Analysis**: Calculate accuracy metrics from result CSV files  
    **Compare CSVs**: Find differences between two result files  
    **HTML Report**: Create visual reports for easy review  
    
    **Tip**: For best results, ensure images are clear and well-lit
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
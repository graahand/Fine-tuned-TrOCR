import os
import json
import re
import pandas as pd
from pathlib import Path
import argparse
import shutil

def create_master_csv(image_folder_path, txt_folder_path, output_csv_path):
    """
    Creates a master CSV file from JSON-formatted TXT files, adding the corresponding image filename.
    """
    all_records = []
    
    # Get lists of txt and image files
    txt_files = {os.path.splitext(f)[0]: f for f in os.listdir(txt_folder_path) if f.lower().endswith('.txt')}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic']
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_folder_path)
                   if any(f.lower().endswith(ext) for ext in image_extensions)}

    for txt_stem, txt_filename in txt_files.items():
        txt_path = os.path.join(txt_folder_path, txt_filename)
        try:
            with open(txt_path, 'r', encoding='utf-8') as fh:
                content = fh.read().strip()

            if not content:
                print(f"Warning: empty file {txt_filename} — skipping")
                continue

            # Parse the strict JSON content
            record = json.loads(content)
            if not isinstance(record, dict):
                print(f"Warning: content of {txt_filename} is not a JSON object — skipping")
                continue

            # Find the corresponding image file
            # Strips suffixes like '_output' or '_output_8b' from the txt filename stem
            base_name = re.sub(r'(_output(_\w+)*)$', '', txt_stem)
            
            image_filename_found = image_files.get(base_name)
            if not image_filename_found:
                image_filename_found = image_files.get(txt_stem)

            if image_filename_found:
                record['image_file_path'] = image_filename_found
            else:
                record['image_file_path'] = 'N/A'
                print(f"Warning: No matching image found for {txt_filename}")

            all_records.append(record)

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from file {txt_filename}. Skipping.")
        except Exception as e:
            print(f"Error processing file {txt_filename}: {e}")

    if not all_records:
        print("No records were extracted. The CSV file will not be created.")
        return None # Return None on failure

    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_records)

    # Reorder columns to put image_file_path last if it exists
    if 'image_file_path' in df.columns:
        cols = [c for c in df.columns if c != 'image_file_path'] + ['image_file_path']
        df = df[cols]

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\nMaster CSV file created successfully at: {output_csv_path}")
    print(f"Total records: {len(df)}")
    return df

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Convert a folder of JSON-like TXT files to a master CSV file.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing the original image files.")
    parser.add_argument("txt_folder", type=str, help="Path to the folder containing the TXT output files.")
    args = parser.parse_args()

    image_folder_path = args.image_folder
    txt_folder_path = args.txt_folder

    if not os.path.isdir(image_folder_path):
        print(f"Error: Image folder does not exist: {image_folder_path}")
        return
    if not os.path.isdir(txt_folder_path):
        print(f"Error: TXT folder does not exist: {txt_folder_path}")
        return

    # Set the output CSV path to be inside the image folder, with a dynamic name
    txt_folder_name = os.path.basename(os.path.normpath(txt_folder_path))
    output_csv_path = os.path.join(image_folder_path, f'{txt_folder_name}.csv')
    
    df = create_master_csv(image_folder_path, txt_folder_path, output_csv_path)
    
    if df is not None and not df.empty:
        print("\nFirst 5 rows of the created CSV:")
        print(df.head())
        
        # Delete the txt folder after successful CSV creation
        try:
            shutil.rmtree(txt_folder_path)
            print(f"\nSuccessfully deleted the text folder: {txt_folder_path}")
        except Exception as e:
            print(f"\nError: Could not delete the text folder {txt_folder_path}. Reason: {e}")

if __name__ == "__main__":
    main()
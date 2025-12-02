import pandas as pd
import os
import argparse
import base64
from pathlib import Path
from PIL import Image, ImageOps
import io

def create_html_report(csv_path, image_folder, output_html_path):
    """
    Creates a self-contained HTML file with resized embedded images and their
    corresponding extracted data from a CSV file.
    """
    try:
        df = pd.read_csv(csv_path).fillna('')
    except FileNotFoundError:
        print(f"Error: The CSV file was not found at '{csv_path}'")
        return

    # --- HTML and CSS Boilerplate ---
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Extraction Report</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f0f2f5; color: #333; }
            h1 { text-align: center; background-color: #4a5568; color: white; padding: 20px 0; margin: 0; }
            .main-container { padding: 20px; max-width: 1200px; margin: 0 auto; }
            
            /* Flex container for the row */
            .item-container { 
                display: flex; 
                flex-wrap: nowrap; /* Prevent wrapping on large screens */
                background-color: white; 
                border: 1px solid #e2e8f0; 
                border-radius: 8px; 
                margin-bottom: 20px; 
                overflow: hidden; 
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); 
            }
            
            /* Image on the LEFT */
            .image-container { 
                flex: 0 0 40%; /* Fixed width of 40% */
                padding: 15px; 
                background-color: #f8fafc;
                border-right: 1px solid #e2e8f0; 
                display: flex; 
                flex-direction: column; 
                justify-content: center; 
                align-items: center; 
            }
            
            .image-container img { 
                max-width: 100%; 
                max-height: 400px; /* Limit height to keep rows compact */
                height: auto; 
                border-radius: 4px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            }
            
            .image-caption { font-size: 0.75em; color: #718096; margin-top: 8px; word-break: break-all; }
            
            /* Data on the RIGHT */
            .data-container { 
                flex: 1 1 60%; /* Takes remaining space */
                padding: 15px; 
                overflow-x: auto; /* Scroll if table is too wide */
            }
            
            table { 
                width: 100%; 
                border-collapse: collapse; 
                font-size: 0.9em; 
            }
            
            th, td { 
                text-align: left; 
                padding: 8px 12px; 
                border-bottom: 1px solid #edf2f7; 
                vertical-align: top;
            }
            
            /* Reduce gap: Field column takes minimal necessary width */
            th { 
                background-color: #f7fafc; 
                font-weight: 600; 
                color: #4a5568;
                width: 1%; /* Trick to shrink column to content width */
                white-space: nowrap; /* Prevent wrapping of field names */
                padding-right: 20px;
            }
            
            tr:last-child td { border-bottom: none; }

            .page-number {
                text-align: center;
                margin-top: -10px;
                margin-bottom: 40px;
                color: #718096;
                font-weight: bold;
                font-size: 0.9em;
            }
            
            @media (max-width: 768px) {
                .item-container { flex-direction: column; }
                .image-container { border-right: none; border-bottom: 1px solid #e2e8f0; flex: auto; width: auto; }
                .data-container { flex: auto; width: auto; }
            }
        </style>
    </head>
    <body>
        <h1>Image Extraction Report</h1>
        <div class="main-container">
    """

    # --- Process each row in the DataFrame ---
    if 'image_file_path' not in df.columns:
        print(f"Error: The CSV must contain an 'image_file_path' column.")
        return

    for i, (index, row) in enumerate(df.iterrows(), 1):
        image_filename = row['image_file_path']
        full_image_path = os.path.join(image_folder, image_filename)

        # --- Resize image and encode to base64 ---
        try:
            with Image.open(full_image_path) as img:
                # Auto-rotate the image based on EXIF data
                img = ImageOps.exif_transpose(img)

                # Calculate new dimensions (50% of original pixel size for file size reduction)
                new_width = img.width // 2
                new_height = img.height // 2
                
                # Resize the image using a high-quality filter
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save resized image to an in-memory buffer
                buffer = io.BytesIO()
                # Preserve original format if possible, otherwise default to JPEG
                img_format = img.format if img.format in ['JPEG', 'PNG', 'GIF'] else 'JPEG'
                resized_img.save(buffer, format=img_format)
                
                # Get base64 string from the buffer's content
                encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_data_uri = f"data:image/{img_format.lower()};base64,{encoded_string}"

        except FileNotFoundError:
            print(f"Warning: Image not found for '{image_filename}'. Skipping image embedding.")
            image_data_uri = "" # No image to show
        except Exception as e:
            print(f"Warning: Could not process image '{image_filename}'. Reason: {e}")
            image_data_uri = ""

        # Start item container
        html_content += '<div class="item-container">'

        # --- 1. Add Image Container (LEFT) ---
        html_content += f"""
            <div class="image-container">
                <img src="{image_data_uri}" alt="{image_filename}">
                <p class="image-caption">{image_filename}</p>
            </div>
        """

        # --- 2. Add Data Container (RIGHT) ---
        html_content += '<div class="data-container"><table>'
        html_content += '<tr><th>Field</th><th>Value</th></tr>'
        for col, value in row.items():
            if col != 'image_file_path': # Don't show the image path in the table
                html_content += f'<tr><td>{col}</td><td>{value}</td></tr>'
        html_content += '</table></div>'

        # End item container
        html_content += '</div>'

        # Add page number
        html_content += f'<div class="page-number">Page {i}</div>'

    # --- Finalize HTML ---
    html_content += """
        </div>
    </body>
    </html>
    """

    # --- Write to file ---
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully generated HTML report: {output_html_path}")
    except Exception as e:
        print(f"Error writing HTML file: {e}")


def main():
    """Main function to parse arguments and run the report generator."""
    parser = argparse.ArgumentParser(description="Generate an HTML report from a CSV file and an image folder.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing the images.")
    parser.add_argument("-o", "--output", type=str, help="Path for the output HTML file. If not provided, it will be named after the CSV file.")
    args = parser.parse_args()

    # Determine the output path
    if args.output:
        # Use the user-provided output path
        output_html_path = args.output
    else:
        # Create a dynamic name based on the CSV file
        csv_basename = os.path.basename(args.csv_file)
        html_filename = os.path.splitext(csv_basename)[0] + '.html'
        output_html_path = html_filename

    create_html_report(args.csv_file, args.image_folder, output_html_path)

if __name__ == "__main__":
    main()
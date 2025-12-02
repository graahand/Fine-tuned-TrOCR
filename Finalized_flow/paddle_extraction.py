# # # Initialize PaddleOCR instance with your fine-tuned model
# # from paddleocr import PaddleOCR

# # # Specify the path to your local fine-tuned recognition model
# # # The directory should contain inference.pdiparams, inference.pdmodel, and inference.yml
# # # local_rec_model_path = r"/home/graahand/vlm-scratch/LeadsAutomation/PaddleOCR/PP-OCRv5_server_rec_infer"
# # local_rec_model_path = r"/home/graahand/vlm-scratch/LeadsAutomation/PaddleOCR/PP-OCRv5_server_rec_infer"

# # ocr = PaddleOCR(
# #     lang='ne',
# #     use_doc_orientation_classify=False,
# #     use_doc_unwarping=False,
# #     use_textline_orientation=False,
# #     text_recognition_model_dir=local_rec_model_path  # <-- Point to your local model
# # )

# # # Run OCR inference on a sample image
# # image_path = r"/home/graahand/vlm-scratch/LeadsAutomation/main_dataset/image17.png"
# # # image_path = r"/home/graahand/vlm-scratch/LeadsAutomation/datasettt/image_122.png"
# # result = ocr.predict(
# #     input=image_path)

# # # Visualize the results and save the JSON results
# # for res in result:
# #     res.print()
# #     res.save_to_img("output_FineTuned_ANNOTATED_133.png")
# #     res.save_to_json("output_FineTuned_ANNOTATED_133.json")


import os
from paddleocr import PaddleOCR
local_rec_model_path = r"/home/museum/Fine-tuned-TrOCR/PaddleOCR/PP-OCRv5_server_rec_infer_customDict"


# Use default model (remove text_recognition_model_dir for default)
ocr = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False,
    use_doc_unwarping=True,
    precision='fp16',
    use_textline_orientation=True,
    # text_recognition_model_dir=local_rec_model_path  # <-- Point to your local model
)

# Define the path to the single image you want to process
image_path = r"/home/museum/Fine-tuned-TrOCR/Finalized_flow/cropped_numbers/dataset_scanned_2_cropped_phone_number.jpg"
output_folder = r"/home/museum/Fine-tuned-TrOCR/orientation_corrected_nov25_resultsCustomDictScannedCroppedVLM"
os.makedirs(output_folder, exist_ok=True)

# Check if the image exists
if os.path.exists(image_path):
    print(f"Processing image: {image_path}")
    
    # Run OCR prediction
    result = ocr.predict(input=image_path)
    
    # Get the base name of the image file
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the output for each result object
    for idx, res in enumerate(result):
        img_out = os.path.join(output_folder, f"{base_name}_ocr_{idx}.png")
        json_out = os.path.join(output_folder, f"{base_name}_ocr_{idx}.json")
        
        # Save the annotated image and the JSON output
        res.save_to_img(img_out)
        res.save_to_json(json_out)
        
        print(f"Saved annotated image to: {img_out}")
        print(f"Saved JSON output to: {json_out}")
        
        # Optionally, print the results to the console
        res.print()
else:
    print(f"Error: Image not found at {image_path}")


# import os
# from paddleocr import PaddleOCR

# local_rec_model_path = r"/home/museum/Fine-tuned-TrOCR/PaddleOCR/PP-OCRv5_server_rec_infer"

# ocr = PaddleOCR(
#     lang='en',
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=True,
#     precision='fp16',
#     use_textline_orientation=True,
#     text_recognition_model_dir=local_rec_model_path
# )

# input_folder = "/home/museum/Downloads/input_dataset"
# output_folder = "output/92_epochs_74_val_acc_model/results"
# os.makedirs(output_folder, exist_ok=True)

# # Path to the summary .txt file
# summary_txt_path = os.path.join(output_folder, "image_texts.txt")

# # Open the summary file once (append mode not needed; we'll write fresh)
# with open(summary_txt_path, 'w', encoding='utf-8') as txt_file:
#     # Process each image in the folder
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#             image_path = os.path.join(input_folder, filename)
#             result = ocr.predict(input=image_path)
            
#             base_name = os.path.splitext(filename)[0]
            
#             # Collect all extracted text from this image
#             all_texts = []
#             for res in result:
#                 # Save individual result files
#                 idx = 0  # since predict returns list, but usually one item per image
#                 img_out = os.path.join(output_folder, f"{base_name}_ocr_{idx}.png")
#                 json_out = os.path.join(output_folder, f"{base_name}_ocr_{idx}.json")
#                 res.save_to_img(img_out)
#                 res.save_to_json(json_out)

#                 # Extract recognized text
#                 json_res = res.json
#                 texts = json_res.get("texts", [])
#                 all_texts.extend(texts)

#             # Combine all lines into a single string (space or newline separated)
#             full_text = " ".join(all_texts).strip()

#             # Write to the summary .txt file: <image_path> <extracted_text>
#             txt_file.write(f"{image_path} {full_text}\n")

# print(f"Summary text file saved to: {summary_txt_path}")
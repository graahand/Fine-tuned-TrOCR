import easyocr
reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
result = reader.readtext('/home/museum/Fine-tuned-TrOCR/Finalized_flow/cropped_numbers/dataset_scanned_2_cropped_phone_number.jpg')
print(result)
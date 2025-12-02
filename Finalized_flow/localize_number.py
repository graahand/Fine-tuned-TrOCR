import re
import os
import argparse
from PIL import Image, ImageDraw
import torch
from transformers import AutoModel, AutoTokenizer

def setup_model_and_tokenizer(model_path):
    dtype = torch.bfloat16
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model = model.to(dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def extract_bbox_from_response(response):
    match = re.search(r"<box>([\d\s]+)</box>", response)
    if match:
        bbox_str = match.group(1)
        bbox = list(map(int, bbox_str.strip().split()))
        return bbox
    else:
        raise ValueError("Can't find bbox in response")

def draw_bbox_on_image(image, bbox):
    w, h = image.size
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    return image

def crop_image_with_bbox(image, bbox):
    w, h = image.size
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image

def model_infer_and_draw(img_path, question, model, tokenizer):
    image = Image.open(img_path)
    msgs = [
        {'role': 'user', 'content': [question, image]},
    ]
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
    print("Model output:", res)
    bbox = extract_bbox_from_response(res)
    cropped_image = crop_image_with_bbox(image, bbox)
    return cropped_image

def main():
    parser = argparse.ArgumentParser(description="Localize and crop phone number and guardian's number from an image.")
    parser.add_argument("img_path", type=str, help="Path to the input image file.")
    args = parser.parse_args()

    model_path = 'openbmb/MiniCPM-V-4_5'
    fields_to_crop = ['phone number', "guardian's number"]
    output_dir = 'cropped_numbers'
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = setup_model_and_tokenizer(model_path)

    base_name = os.path.basename(args.img_path)
    name, ext = os.path.splitext(base_name)

    for field in fields_to_crop:
        print(f"--- Processing field: {field} ---")
        question = f'Please provide the bounding box coordinate of the region this sentence describes: <ref>{field}</ref>'
        
        try:
            cropped_image = model_infer_and_draw(args.img_path, question, model, tokenizer)

            # Save the cropped image
            field_slug = field.replace("'", "").replace(" ", "_")
            cropped_out_path = os.path.join(output_dir, f'{name}_cropped_{field_slug}.jpg')
            cropped_image.save(cropped_out_path)
            print(f"Saved cropped {field} to: {cropped_out_path}")

        except ValueError as e:
            print(f"Could not process field '{field}' for image {args.img_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{field}' for {args.img_path}: {e}")
        print("-" * 20)

if __name__ == "__main__":
    main()
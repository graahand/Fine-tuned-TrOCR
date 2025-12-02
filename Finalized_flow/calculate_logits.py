import argparse
import torch
import os
from torch.nn.functional import softmax
from transformers import AutoProcessor

# Try importing Qwen3, fallback to Qwen2 if necessary (based on your environment)
try:
    from transformers import Qwen3VLForConditionalGeneration
    ModelClass = Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as ModelClass

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate text and analyze logits/confidence for a single image.")
    parser.add_argument("image_path", type=str, help="Path to the single image file.")
    args = parser.parse_args()

    image_path = args.image_path

    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    # --- Model and Processor Loading ---
    model_path_str = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading model: {model_path_str}...")
    
    model = ModelClass.from_pretrained(
        model_path_str,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path_str)
    print("Model loaded successfully.")

    # --- Prompt Definition ---
    prompt_text = (
        "Carefully analyze the provided image, which contains a filled-out student registration or feedback form. "
        "Extract **only** the following fields exactly as they appear: "
        "'Name', 'class', 'address', 'faculty', 'phone number', \"guardian's number\", 'email', 'interested course', "
        "'school name', 'who gave seminar', and 'rate us'.\n"
        "Return your response as a valid JSON object."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # --- Inference ---
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    print(f"Processing image: {image_path}...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200,        # Changed from 300 to 8192 as requested
            do_sample= False,             # Required for temperature and top_p
            # temperature=0.1,            # Added
            # top_p=0.0000001,                # Added
            repetition_penalty=1.05,    # Added
            return_dict_in_generate=True, 
            output_scores=True
            # stop_token_ids=[]         # Optional: Add specific IDs if needed, otherwise default is used
        )

    # --- Process Logits and Confidence ---
    generated_ids = outputs.sequences
    scores = outputs.scores  # Tuple of tensors (one per generation step)

    # Slice generated_ids to exclude input prompt tokens
    input_len = inputs.input_ids.shape[1]
    new_tokens = generated_ids[:, input_len:]

    # Decode the full text
    full_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    
    print("\n" + "="*60)
    print("FINAL GENERATED TEXT")
    print("="*60)
    print(full_text)
    print("="*60 + "\n")

    print(f"{'Token String':<25} | {'Confidence':<10} | {'Logit':<10}")
    print("-" * 55)

    confidences = []

    # Iterate through each step of generation
    for i, step_logits in enumerate(scores):
        # step_logits shape is (batch_size, vocab_size)
        probabilities = softmax(step_logits, dim=-1)

        # Get the token ID for the generated token at this step
        token_id = new_tokens[0, i].item()

        # Look up the probability (confidence) of the generated token
        confidence = probabilities[0, token_id].item()

        # Decode the token ID to get the actual token string
        token_string = processor.tokenizer.decode([token_id])

        confidences.append((token_string, confidence))

        # Print token, confidence, and logit
        print(f"{token_string:<25} | {confidence:<10.4f} | {step_logits[0, token_id]:<10.4f}")

    # Optionally, you can return or save the confidences for further analysis
    return confidences

if __name__ == "__main__":
    main()
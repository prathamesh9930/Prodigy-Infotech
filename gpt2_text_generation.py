# Import necessary libraries
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, set_seed

# Set a seed for reproducibility
set_seed(42)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

# Set the padding token to be the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text
def generate_text(prompt, max_length=50, top_k=50, top_p=0.95, num_return_sequences=3):
    # Tokenize input prompt and move it to the same device as the model
    input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generate text
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences
    )
    
    # Decode generated outputs and return
    generated_texts = []
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

# Example usage
if __name__ == "__main__":
    # Input prompt
    prompt = "Once upon a time"
    
    # Generate text based on the prompt
    generated_texts = generate_text(prompt)

    # Print generated outputs
    print("Generated Texts:")
    for idx, text in enumerate(generated_texts):
        print(f"\nGenerated text {idx + 1}:\n{text}")

# inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model():
    """
    Load the base model and fine-tuned LoRA adapter.
    """
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")

    # Load the fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(base_model, "vignesh0007/Hubermangpt")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", use_fast=True)

    # Move the model to GPU (if available)
    if torch.cuda.is_available():
        model.to("cuda")

    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    """
    Generate a response using the fine-tuned model.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate the response
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=280)

    # Decode the generated tokens to text
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return response

def main():
    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Example prompt
    prompt = '''<s>[INST] HubermanGPT, functioning as a virtual neuroscience expert, communicates complex scientific concepts in an accessible manner.
    It escalates to deeper details on request and responds to feedback thoughtfully. HubermanGPT adapts the length of its responses based on the user's input, providing concise answers for brief comments or deeper explanations for detailed inquiries.

    Please respond to the following question: "How does dopamine influence motivation?" [/INST]'''

    # Generate and print the response
    response = generate_response(model, tokenizer, prompt)
    print("Response:", response)

if __name__ == "__main__":
    main()
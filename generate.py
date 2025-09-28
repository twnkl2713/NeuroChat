from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel
import torch
import re

# Load tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained("./mini_gpt_model")
tokenizer.pad_token = tokenizer.eos_token
base_model = GPT2LMHeadModel.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./mini_gpt_model")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")

def clean_response(text):
    """Clean up the generated response"""
    # Remove any "Assistant:" prefixes that might be repeated
    text = re.sub(r'^Assistant:\s*', '', text)
    # Remove any new "User:" that might have been generated
    text = text.split('User:')[0]
    # Remove extra whitespace and EOS tokens
    text = text.strip()
    text = re.sub(r'<\|endoftext\|>', '', text)
    return text

def generate_response(prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    """Generate response for the given prompt using the trained format"""
    # Format the prompt to match training format
    formatted_prompt = f"User: {prompt}\nAssistant:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    
    # Extract the full generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "Assistant:" in full_output:
        response = full_output.split("Assistant:")[1].strip()
    else:
        # Fallback: if format is lost, try to extract after our prompt
        response = full_output[len(formatted_prompt):].strip()
    
    # Clean up the response
    response = clean_response(response)
    
    return response

# Simple generation without formatting (fallback)
def generate_simple(prompt, max_new_tokens=80, temperature=0.7):
    """Simpler generation without formatting"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    # Get only the new tokens (response part)
    response_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response.strip()

print("Mini-GPT Ready! Type 'quit' to exit...")
print("You can try questions from the training data like:")
print("- What is your name?")
print("- Tell me a joke")
print("- What is AI?")
print("- Who created Python?")
print()

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Mini-GPT: Goodbye! Have a great day! 👋")
        break
    
    if not user_input:
        continue
        
    try:
        # Use the formatted generation
        response = generate_response(user_input)
        
        # If response is empty or too short, try simple generation
        if not response or len(response) < 2:
            response = generate_simple(user_input)
            
        # If still no response, provide a default
        if not response:
            response = "I'm not sure how to respond to that. Could you try asking something else?"
            
        print("Mini-GPT:", response)
        
    except Exception as e:
        print(f"Mini-GPT: Sorry, I encountered an error. Please try again.")
        if "cuda" in str(e).lower():
            print("(This might be a GPU memory issue. Try reducing max_new_tokens)")

# Print some sample prompts to try
print("\nSample questions you trained on:")
sample_questions = [
    "What is your name?",
    "Tell me a joke",
    "What is AI?",
    "Who created Python?",
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who is Albert Einstein?",
    "Tell me a fun fact"
]
for q in sample_questions:
    print(f"- {q}")
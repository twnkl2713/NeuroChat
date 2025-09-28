import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

# Load tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained("./mini_gpt_model")
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./mini_gpt_model")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Chat function
def chat_with_gpt(message, history):
    history = history or []
    prompt = ""

    for msg in history:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            prompt += f"User: {content}\n"
        else:
            prompt += f"Mini-GPT: {content}\n"

    prompt += f"User: {message}\nMini-GPT:"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    if inputs.input_ids.shape[1] == 0:
        return history, history

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=150,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = response.split("Mini-GPT:")[-1].strip()

    history.append({'role': 'user', 'content': message})
    history.append({'role': 'assistant', 'content': response})

    return history, history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 NeuroChat")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type your message here...")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_with_gpt, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch()

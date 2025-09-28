import json
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# Load dataset
with open("data/custom_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} training examples")
dataset = Dataset.from_list(raw_data)

# Load tokenizer and model
model_name = "gpt2" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize data with conversation format
def tokenize_fn(example):
    prompt = example["prompt"]
    response = example["response"]
    # Format: "User: <prompt>\nAssistant: <response><eos>"
    text = f"User: {prompt}\nAssistant: {response}{tokenizer.eos_token}"
    tokenized = tokenizer(text, truncation=True, max_length=256, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=["prompt", "response"])

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# Training arguments optimized for your dataset
training_args = TrainingArguments(
    output_dir="./mini_gpt_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=15,  # More epochs for better learning
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="no",
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=30,
    fp16=False,
    save_total_limit=2,
    prediction_loss_only=True,
    remove_unused_columns=False
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model and tokenizer
trainer.save_model("./mini_gpt_model")
tokenizer.save_pretrained("./mini_gpt_model")

print("Training complete! Model saved in ./mini_gpt_model")
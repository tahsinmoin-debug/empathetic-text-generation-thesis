import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -----------------------------
# 1. MODEL PATH
# -----------------------------

MODEL_PATH = "../checkpoints/gpt2_medium/final"
# or use a checkpoint:
# MODEL_PATH = "../checkpoints/gpt2_medium/checkpoint-16800"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2. LOAD MODEL & TOKENIZER
# -----------------------------

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# -----------------------------
# 3. GENERATION FUNCTION
# -----------------------------

def generate_response(
    text,
    emotion,
    max_new_tokens=80,
    temperature=0.8,
    top_p=0.9
):
    prompt = f"<emotion={emotion}> {text}"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# -----------------------------
# 4. INTERACTIVE TESTING
# -----------------------------

print("\n🧪 GPT-2 Medium Empathy Testing")
print("Type 'quit' to exit\n")

while True:
    emotion = input("Emotion (Loss / Anger / Fear / Self-Conscious / Disgust): ").strip()
    if emotion.lower() == "quit":
        break

    user_text = input("User text: ").strip()
    if user_text.lower() == "quit":
        break

    print("\n🤖 Model output:")
    print(generate_response(user_text, emotion))
    print("-" * 50)

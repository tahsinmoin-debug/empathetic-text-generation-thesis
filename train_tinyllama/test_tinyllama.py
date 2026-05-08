import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. MODEL PATH
# -----------------------------

MODEL_PATH = "../checkpoints/tinyllama_lora/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2. LOAD TOKENIZER & MODEL
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    load_in_4bit=True
)

model.eval()

print("✅ TinyLLaMA model loaded successfully")

# -----------------------------
# 3. GENERATION FUNCTION
# -----------------------------

def generate_response(
    text,
    emotion,
    max_new_tokens=80,
    temperature=0.9,
    top_p=0.95
):
    prompt = f"<emotion={emotion}> {text}"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# -----------------------------
# 4. INTERACTIVE TESTING LOOP
# -----------------------------

print("\n🧪 TinyLLaMA Empathy Testing")
print("Type 'quit' to exit\n")

while True:
    emotion = input("Emotion (Loss / Anger / Fear / Self-Conscious / Disgust): ").strip()
    if emotion.lower() == "quit":
        break

    text = input("User text: ").strip()
    if text.lower() == "quit":
        break

    print("\n🤖 Model output:")
    print(generate_response(text, emotion))
    print("-" * 60)

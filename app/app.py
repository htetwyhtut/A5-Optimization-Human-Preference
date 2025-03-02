from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the trained model and tokenizer
model_path = "htet22/dpo-trained-gpt2" 
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate a response
def generate_response(prompt: str, max_length: int = 100) -> str:
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.5, 
            top_p=0.85,   
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    response = full_text[len(prompt):].strip()
    
    return response

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    response = None
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if prompt:
            response = generate_response(prompt)
    return render_template("index.html", response=response)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

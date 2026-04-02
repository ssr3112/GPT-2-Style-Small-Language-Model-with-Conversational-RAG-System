import torch
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel


from model.config import SLMConfig
from optimised_KVmodel import GPTWithKVCache


# ==========================
# Request Schema

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: int = 40


# ==========================
# Initialize App

app = FastAPI(title="SLM Text Generation API")


# ==========================
# Load Model 

config = SLMConfig()

model = GPTWithKVCache(config)


state_dict = torch.load("best_model_parameter.pt", map_location="cpu")

new_state_dict = {}

for k, v in state_dict.items():

    if k.startswith("transformer_blocks"):
        k = k.replace("transformer_blocks", "blocks")

    if k.startswith("output_head"):
        k = k.replace("output_head", "head")

    new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)

model.eval()



tokenizer = tiktoken.get_encoding("gpt2")


# ==========================
# API Endpoint

@app.post("/generate")
def generate_text(request: GenerateRequest):

    prompt = request.prompt

    tokens = tokenizer.encode(prompt)

    x = torch.tensor(tokens).unsqueeze(0)

    generated = model.generate(
        x,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k
    )

    text = tokenizer.decode(generated[0].tolist())

    return {
        "input": prompt,
        "generated_text": text
    }


# ==========================
# Root Checking

@app.get("/")
def home():
    return {"message": "SLM API is running "}
# Purpose: Module for chat.
# Created: 2026-01-05
# Author: MWR

import torch
from unsloth import FastLanguageModel
import gradio as gr
from threading import Thread
from transformers import TextIteratorStreamer

# -------------------------
# CONFIG
# -------------------------
# Options: 
# "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
# "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit"
# "unsloth/Qwen2.5-14B-Instruct-bnb-4bit" (Updated to stable version)

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are a helpful AI assistant running locally on a 3080 Ti. Respond concisely and accurately."

# -------------------------
# LOAD MODEL
# -------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    device_map="auto",
)
FastLanguageModel.for_inference(model)

# -------------------------
# GENERATION LOGIC
# -------------------------
def generate_response(message, history):
    # Format history for the chat template
    formatted_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        formatted_history.append({"role": "assistant", "content": assistant_msg})
    
    formatted_history.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        formatted_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True, # Required for temperature to work
        use_cache=True,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    output = ""
    for token in streamer:
        output += token
        yield output

# -------------------------
# UI SETUP
# -------------------------
demo = gr.ChatInterface(
    fn=generate_response,
    title="Local Unsloth Dev Node (RTX 3080 Ti)",
    description=f"Current Model: **{MODEL_NAME}**",
    examples=["Explain quantum entanglement briefly.", "Write a Python script to scrape a website."],
    theme="soft",
)

if __name__ == "__main__":
    # Changed to 0.0.0.0 so you can access it from other devices on your LAN
    demo.launch(server_name="0.0.0.0", server_port=7860)
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def benchmark_speed(model_path):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    # tokenizer.apply_chat_template()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare input
    # prompt = "Who are you?"
    prompt = "Write an article about spring"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # Warmup (not timed)
    _ = model.generate(**inputs, max_new_tokens=5)

    # Timed inference with streaming
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=235,
            temperature=0.67,
            do_sample=False,
            repetition_penalty=1.1,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    elapsed = time.time() - start_time

    # Calculate metrics
    total_tokens = len(output_ids[0]) - len(inputs.input_ids[0])
    tokens_per_sec = total_tokens / elapsed

    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Tokens per second: {tokens_per_sec:.2f}")


if __name__ == "__main__":
    default_path = "checkpoints/Qwen2.5-0.5B-Instruct"
    model_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    benchmark_speed(model_path)

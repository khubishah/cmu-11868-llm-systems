from datasets import load_dataset
import sglang as sgl
import asyncio
import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run inference with a specific model path.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs.jsonl",
    )
    args = parser.parse_args()

    # Load dataset using json format instead of the deprecated dataset script
    dataset = load_dataset("json", data_files="hf://datasets/tatsu-lab/alpaca_eval/alpaca_eval.json", split="train")
    model_path = args.model_path

    # TODO: initialize sglang egnine here
    # you may want to explore different args we can pass here to make the inference faster
    # e.g. dp_size, mem_fraction_static
    llm = sgl.Engine(
        model_path=model_path,
        tp_size=1,  # Tensor parallelism size (number of GPUs for model parallelism)
        dp_size=2,  # Data parallelism size (number of GPUs for data parallelism)
        mem_fraction_static=0.65,  # Lower memory allocation to allow CUDA graph to work
        attention_backend="dual_chunk_flash_attn",  # Use dual_chunk_flash_attn backend for this model
        cuda_graph_max_bs=16,  # Limit CUDA graph batch size to reduce memory usage
    )

    prompts = []

    for i in dataset:
        prompts.append(i['instruction'])

    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 2048}  # Reduced from 8192

    outputs = []

    # TODO: you may want to explore different batch_size
    batch_size = 32  # Process 32 prompts at a time for better memory management 

    
    for i in tqdm(range(0, len(prompts), batch_size)):
        # TODO: prepare the batched prompts and use llm.generate
        # save the output in outputs
        batch_prompts = prompts[i:i+batch_size]
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        for output in batch_outputs:
            outputs.append(output["text"])

    with open(args.output_file, "w") as f:
        for i in range(0, len(outputs), 10):
            instruction = prompts[i]
            output = outputs[i]
            f.write(json.dumps({
                "output": output,
                "instruction": instruction
            }) + "\n")

if __name__ == "__main__":
    main()

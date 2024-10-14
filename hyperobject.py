import argparse
import csv
import json
import logging
import os
import pickle
import hashlib
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import torch
import tkinter as tk
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import math

# from huggingface_hub import login
# login(token="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")

# check if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

prompts_file_path = "prompts.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global cache dictionary
generation_cache = {}
cache_writes = 0
current_seed = 0

def save_cache():
    with open('generation_cache.pkl', 'wb') as f:
        pickle.dump(generation_cache, f)
    logger.info("Cache saved to generation_cache.pkl")

def load_cache():
    global generation_cache
    if os.path.exists('generation_cache.pkl'):
        with open('generation_cache.pkl', 'rb') as f:
            generation_cache = pickle.load(f)
        logger.info("Cache loaded from generation_cache.pkl")
    else:
        logger.info("No existing cache found. Starting with an empty cache.")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare language models.")
    parser.add_argument("--cache_dir", type=str, help="Directory to cache models")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate per prompt")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for prompt processing")
    parser.add_argument("--step_size", type=int, default=200, help="Number of tokens to generate per step")
    parser.add_argument("--prompts_file", type=str, default=prompts_file_path, help="CSV file containing prompts")
    parser.add_argument("--logs_dir", type=str, default="./logs", help="Directory to save logs and results")
    return parser.parse_args()


def load_models_and_tokenizers(model_names: List[str], cache_dir: str) -> Dict[
    str, Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    models_and_tokenizers = {}
    for model_name in model_names:
        logger.info(f"Loading model and tokenizer: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'left'
        models_and_tokenizers[model_name] = (model, tokenizer)
        logger.info(f"Successfully loaded model and tokenizer: {model_name}")
    return models_and_tokenizers


def read_prompts(file_path: str) -> List[str]:
    prompts = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                prompts.append(row[1])
    logger.info(f"Successfully read {len(prompts)} prompts from {file_path}")
    return prompts


def compute_entropy_and_varentropy(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    epsilon = 1e-10
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + epsilon)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    varentropy = torch.sum(probs * (log_probs + entropy.unsqueeze(-1)) ** 2, dim=-1)

    return entropy, varentropy


def generate_and_compute_metrics(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_tokens: int,
        step_size: int,
        model_name: str
) -> List[Dict[str, Any]]:
    global generation_cache, cache_writes, current_seed

    # Create a unique key for this generation
    key = hashlib.md5(f"{prompt}_{model_name}_{current_seed}".encode()).hexdigest()

    if key in generation_cache:
        logger.info(f"Using cached result for prompt: {prompt[:30]}...")
        current_seed += 1
        return generation_cache[key]

    metrics = []
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)  # Create attention mask
    logger.info(f"Initial input_ids shape: {input_ids.shape}")
    generated_text = prompt

    for step in range(0, max_tokens, step_size):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,  # Pass the attention mask
                max_new_tokens=step_size,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        if not outputs.scores:
            continue

        new_tokens = outputs.sequences[0, input_ids.shape[1]:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_text += new_text
        print(f"\nGenerated text (step {step}):\n{new_text}")

        logits = torch.stack(outputs.scores, dim=1)
        logger.info(f"Step {step}: logits shape: {logits.shape}")

        entropy, varentropy = compute_entropy_and_varentropy(logits)
        logger.info(f"Step {step}: entropy shape: {entropy.shape}, varentropy shape: {varentropy.shape}")

        valid_indices = ~(
                    torch.isnan(entropy) | torch.isinf(entropy) | torch.isnan(varentropy) | torch.isinf(varentropy))
        entropy = entropy[valid_indices]
        varentropy = varentropy[valid_indices]
        if entropy.numel() > 0 and varentropy.numel() > 0:
            base_metric = {
                "prompt": prompt,
                "model": model_name,
            }
            metrics.extend({
                               **base_metric,
                               "prompt #": i + step,
                               "entropy": e.item(),
                               "varentropy": v.item()
                           } for i, (e, v) in enumerate(zip(entropy, varentropy)))

        input_ids = outputs.sequences
        logger.debug(f"Updated input_ids shape: {input_ids.shape}")
    logger.info(f"Total metrics generated: {len(metrics)}")
    print(f"\nFull generated text:\n{generated_text}")

    # Cache the result
    generation_cache[key] = metrics
    cache_writes += 1
    if cache_writes % 2 == 0:
        save_cache()

    current_seed += 1
    return metrics


def plot_metrics(metrics_by_model: Dict[str, List[Dict[str, float]]]):
    plt.figure(figsize=(12, 8))
    valid_data = False
    for model_name, metrics in metrics_by_model.items():
        entropies = []
        varentropies = []
        for m in metrics:
            try:
                e, v = float(m["entropy"]), float(m["varentropy"])
                if not (math.isnan(e) or math.isinf(e) or math.isnan(v) or math.isinf(v)):
                    entropies.append(e)
                    varentropies.append(v)
            except (ValueError, TypeError):
                pass

        if entropies and varentropies:
            plt.scatter(entropies, varentropies, label=model_name, alpha=0.7)
            valid_data = True

    if not valid_data:
        return

    plt.xlabel("Entropy")
    plt.ylabel("Varentropy")
    plt.title("Entropy vs Varentropy for Different Models")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def print_debug_info(metrics_by_model: Dict[str, List[Dict[str, float]]]):
    for model_name, metrics in metrics_by_model.items():
        logger.info(f"Debug info for {model_name}:")
        logger.info(f"Number of metrics: {len(metrics)}")
        if metrics:
            logger.info(f"First metric: {metrics[0]}")
            logger.info(f"Last metric: {metrics[-1]}")
        else:
            logger.info("No metrics available")
        logger.info("---")


def save_results(metrics_by_model: Dict[str, List[Dict[str, Any]]], logs_dir: str, batch_num: int):
    os.makedirs(logs_dir, exist_ok=True)
    file_path = os.path.join(logs_dir, f"metrics_results_batch_{batch_num}.json")

    organized_metrics = {}
    for model_name, metrics in metrics_by_model.items():
        for metric in metrics:
            prompt = metric['prompt']
            if prompt not in organized_metrics:
                organized_metrics[prompt] = {}
            if model_name not in organized_metrics[prompt]:
                organized_metrics[prompt][model_name] = {
                    'entropy': [],
                    'varentropy': []
                }
            organized_metrics[prompt][model_name]['entropy'].append(metric['entropy'])
            organized_metrics[prompt][model_name]['varentropy'].append(metric['varentropy'])

    with open(file_path, 'w') as f:
        json.dump(organized_metrics, f, indent=2)
    logger.info(f"Results for batch {batch_num} saved to {file_path}")

    csv_file_path = os.path.join(logs_dir, f"metrics_results_batch_{batch_num}.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['prompt', 'model', 'token_number', 'entropy', 'varentropy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for prompt, models in organized_metrics.items():
            for model, data in models.items():
                for i, (entropy, varentropy) in enumerate(zip(data['entropy'], data['varentropy'])):
                    writer.writerow({
                        'prompt': prompt,
                        'model': model,
                        'token_number': i,
                        'entropy': entropy,
                        'varentropy': varentropy
                    })
    logger.info(f"CSV results for batch {batch_num} saved to {csv_file_path}")


def create_heatmap(metrics_by_model: Dict[str, List[Dict[str, float]]], logs_dir: str):
    width, height = 250, 150  # Reduced to account for 2x2 pixels
    heatmap = np.zeros((height, width, 3), dtype=np.float32)  # Start with black

    model_names = ["gpt2", "HuggingFaceTB/SmolLM-360M", "meta-llama/Llama-3.2-1B-Instruct"]
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue for each model
    
    all_entropies = []
    all_normalized_varentropies = []

    for model_name, color in zip(model_names, colors):
        metrics = metrics_by_model[model_name]
        entropies = []
        normalized_varentropies = []
        for m in metrics:
            try:
                e, v = float(m["entropy"]), float(m["varentropy"])
                if not (math.isnan(e) or math.isinf(e) or math.isnan(v) or math.isinf(v)) and e != 0:
                    entropies.append(e)
                    normalized_varentropies.append(math.log(e / math.sqrt(v)))
            except (ValueError, TypeError):
                pass

        all_entropies.extend(entropies)
        all_normalized_varentropies.extend(normalized_varentropies)

        if entropies and normalized_varentropies:
            x = np.array(entropies)
            y = np.array(normalized_varentropies)
            
            # Use log scale for intensity
            intensity, _, _ = np.histogram2d(x, y, bins=(width, height))
            intensity = np.log1p(intensity)
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            
            # Add the color layer
            color_layer = np.zeros((height, width, 3), dtype=np.float32)
            for i in range(3):
                color_layer[:,:,i] = color[i] * intensity.T
            
            # Combine with existing heatmap
            heatmap = np.maximum(heatmap, color_layer)

    # Convert to white background
    heatmap = 1 - heatmap

    # Ensure the heatmap values are in the correct range
    heatmap = np.clip(heatmap, 0, 1)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display the heatmap with colors and 2x2 pixels
    im = ax.imshow(heatmap, aspect='auto', 
                   extent=[min(all_entropies), max(all_entropies), 
                           min(all_normalized_varentropies), max(all_normalized_varentropies)],
                   interpolation='nearest')  # This ensures sharp 2x2 pixels
    
    # Set labels and title
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Normalized Varentropy")
    ax.set_title("Hyperobject Heatmap")
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=model_name, 
                       markerfacecolor=color, markersize=10)
                       for model_name, color in zip(model_names, colors)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the figure
    fig_path = os.path.join(logs_dir, "hyperobject_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Heatmap saved to {fig_path}")

    # Display the heatmap in a separate window
    plt.show()


def main(args=None):
    if args is None:
        args = parse_arguments()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    config_path = os.path.join(args.logs_dir, "run_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Run configuration saved to {config_path}")

    # Load the cache at startup
    load_cache()

    model_names = ["gpt2", "HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-360M", "meta-llama/Llama-3.2-1B-Instruct"]

    models_and_tokenizers = load_models_and_tokenizers(model_names, args.cache_dir)

    prompts = read_prompts(args.prompts_file)

    metrics_by_model = {model_name: [] for model_name in model_names}

    for batch_num, i in enumerate(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]

        for model_name, (model, tokenizer) in models_and_tokenizers.items():
            logger.info(f"Processing batch {batch_num + 1} for model: {model_name}")
            for prompt in tqdm(batch_prompts, desc=f"Generating with {model_name}"):
                metrics = generate_and_compute_metrics(
                    model_name=model_name, model=model, tokenizer=tokenizer, prompt=prompt, step_size=args.step_size,
                    max_tokens=args.max_tokens
                )
                metrics_by_model[model_name].extend(metrics)

            print_debug_info(metrics_by_model)

        save_results(metrics_by_model, args.logs_dir, batch_num)
        
        # Create and display the heatmap after each batch
        create_heatmap(metrics_by_model, args.logs_dir)

    # Final heatmap creation after processing all batches
    create_heatmap(metrics_by_model, args.logs_dir)


if __name__ == "__main__":
    main()

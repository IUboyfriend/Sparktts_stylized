import sys
import os
# ensure parent folder (project root) is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import gc
import re

from cli.SparkTTS import SparkTTS
import torch
from baukit import TraceDict
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help='local directory with model data')
    parser.add_argument("--test_folder", type=str, required=True, help='Test folder with transcript data')
    parser.add_argument("--gender_neutral", choices=["male", "female"], default="female")
    parser.add_argument("--pitch_neutral", choices=["very_low", "low", "moderate", "high", "very_high"], default="high")
    parser.add_argument("--speed_neutral", choices=["very_low", "low", "moderate", "high", "very_high"], default="high")
    parser.add_argument("--save_dir", type=str, default="token_activations", help='directory to save activations')
    parser.add_argument("--target_token", type=int, default=165156, help='Target token ID to trace')
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()

def get_activations_for_token(model, tokenizer, prompt_ids, target_token, device):
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    
    token_detected = False
    token_activations = None
    generated_tokens = []
    
    # Initialize with prompt
    current_tokens = prompt_ids.to(device)
    generated_tokens.extend(current_tokens[0].cpu().numpy().tolist())
    
    # Generate token by token
    with torch.no_grad():
        for _ in range(600):  # max length for safety
            # Forward pass
            outputs = model(current_tokens)
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([[probs.argmax().item()]]).to(device)
            
            # Check if this is the target token
            if token.item() == target_token:
                token_detected = True
                print(f"Target token {target_token} found!")
                print(f"Decoded token: {tokenizer.decode([target_token])}")
                
                # Capture activations using TraceDict
                with TraceDict(model, HEADS, retain_input=True) as ret:
                    # We need to run forward on the entire sequence that produced this token
                    # Directly use the current tokens which already contain the full sequence
                    model(current_tokens, output_hidden_states=True)
                    
                    # Get activations just for the last token (which is our target)
                    head_wise_hidden_states = []
                    for head in HEADS:
                        head_activations = ret[head].input[0, -1].detach().cpu()
                        head_wise_hidden_states.append(head_activations)
                    
                    # Stack activations for all heads
                    token_activations = torch.stack(head_wise_hidden_states, dim=0).numpy()
                    print(f"Collected activations of shape: {token_activations.shape}")
                    break  # Stop generation after finding the target token
            
            # Add token to sequence
            current_tokens = torch.cat((current_tokens, token), dim=1)
            generated_tokens.append(token.item())
            
            # Check if we've reached a stopping token
            if token.item() in [151643, 151644, 151645]:  # End tokens
                break
    
    return token_detected, token_activations, np.array(generated_tokens)

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading SparkTTS model from {args.model_dir}")
    model = SparkTTS(args.model_dir, device)
    tokenizer = model.tokenizer
    
    # Find transcript file in test folder
    for file in os.listdir(args.test_folder):
        if file.endswith(".json"):
            transcript_file = os.path.join(args.test_folder, file)
            break
    
    # Load transcript data
    print(f"Loading transcripts from {transcript_file}")
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Process prompts
    all_token_activations = []
    all_sequences = []
    found_count = 0
    
    for filename in tqdm(transcript_data):
        transcript_sample = transcript_data[filename]
        transcript_text = transcript_sample["text"]
        
        # Create prompt
        control_tts_inputs = model.process_prompt_control(
            args.gender_neutral, 
            args.pitch_neutral, 
            args.speed_neutral, 
            transcript_text
        )
        
        prompt = "".join(control_tts_inputs)
        prompt_ids = model.tokenizer([prompt], return_tensors='pt').input_ids
        
        # Get activations for target token
        found, activations, sequence = get_activations_for_token(
            model.model,  # Using the inner model (Qwen) 
            tokenizer,
            prompt_ids,
            args.target_token,
            device
        )
        
        if found:
            found_count += 1
            all_token_activations.append(activations)
            all_sequences.append({
                "filename": filename,
                "text": transcript_text,
                "sequence": sequence.tolist()
            })
            
            # Save after each successful finding to prevent loss of data
            if found_count % 5 == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                np.save(os.path.join(args.save_dir, f'token_{args.target_token}_activations_temp.npy'), np.array(all_token_activations))
                with open(os.path.join(args.save_dir, f'token_{args.target_token}_sequences_temp.json'), 'w') as f:
                    json.dump(all_sequences, f, indent=2)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save final results
    if found_count > 0:
        os.makedirs(args.save_dir, exist_ok=True)
        np.save(os.path.join(args.save_dir, f'token_{args.target_token}_activations.npy'), np.array(all_token_activations))
        with open(os.path.join(args.save_dir, f'token_{args.target_token}_sequences.json'), 'w') as f:
            json.dump(all_sequences, f, indent=2)
        
        print(f"Found target token in {found_count} sequences")
        print(f"Saved activations to {os.path.join(args.save_dir, f'token_{args.target_token}_activations.npy')}")
    else:
        print(f"Target token {args.target_token} not found in any sequence")

if __name__ == '__main__':
    main() 
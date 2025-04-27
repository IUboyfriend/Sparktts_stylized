# python get_tts_activations.py SparkTTS --stylized_dir "../dataset/stylized" --neutral_dir "../dataset/original" --transcript_file "../dataset/transcription_results.json" --model_dir "../pretrained_models/Spark-TTS-0.5B"
# python get_tts_activations.py SparkTTS --stylized_dir "../dataset/testing/stylized_test" --neutral_dir "../dataset/testing/original_test" --transcript_file "../dataset/testing/transcription_results_testing.json" --model_dir "../pretrained_models/Spark-TTS-0.5B"
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

from cli.SparkTTS import SparkTTS
import torch
from baukit import TraceDict

def get_sparktts_activations(model, prompt_ids, device):

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    with torch.no_grad():
        prompt = prompt_ids.to(device)
        with TraceDict(model, HEADS,retain_input=True) as ret:
            output = model(prompt, output_hidden_states = True)
        head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS] # each is (batch_size, seq_len, num_heads * head_dim)
        # print(len(head_wise_hidden_states)) 24
        # print(head_wise_hidden_states[0].shape)
        # print(head_wise_hidden_states[1].shape) [249, 896]
        # print(head_wise_hidden_states[2].shape)
        # print(head_wise_hidden_states[3].shape) 
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        print(head_wise_hidden_states.shape) # (24, 249, 896)
    return head_wise_hidden_states 






def tokenized_speech_dataset(stylized_dir, neutral_dir, transcript_data, model):
    """
    Prepare speech datasets for activation extraction with transcripts
    
    Args:
        stylized_dir: Directory with stylized speech files
        neutral_dir: Directory with neutral speech files
        transcript_data: Dictionary mapping filenames to transcript data
        model: SparkTTS
    
    Returns:
        all_prompts: List of tokenized prompts
        all_labels: List of labels (1 for stylized, 0 for neutral)
    """
    all_prompts = []
    all_labels = []
    
    # Get matching files from both directories
    stylized_files = sorted(Path(stylized_dir).glob('*.wav'))
    neutral_files = sorted(Path(neutral_dir).glob('*.wav'))
    if len(stylized_files) != len(neutral_files):
        print(f"Warning: Number of files in stylized ({len(stylized_files)}) and neutral ({len(neutral_files)}) directories don't match")
    
    # Process each file pair
    for stylized_file, neutral_file in zip(stylized_files, neutral_files):
        # Get filename with extension for transcript lookup
        filename = stylized_file.name  # Use .name to get filename with extension
        
        # Get transcript text for this file
        if filename in transcript_data:
            transcript_text = transcript_data[filename]["text"]
        else:
            print(f"Warning: No transcript found for {filename}, using empty string")
            transcript_text = ""
        
        # Get global and semantic tokens for stylized speech
        global_token_ids, semantic_token_ids = model.audio_tokenizer.tokenize(str(stylized_file))
        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]) # use squeeze() to remove extra dimension of size 1 since only one audio file
        semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()])
        

        #########################################################这个地方可能需要修改 #########################################################
        # Format prompt for stylized speech with transcript
        stylized_prompt = (
            "<|tts|><|start_content|>" + transcript_text + "<|end_content|>"
            "<|start_global_token|>" + global_tokens + "<|end_global_token|>"
            "<|start_semantic_token|>" + semantic_tokens
        )
        stylized_prompt_ids = model.tokenizer(stylized_prompt, return_tensors='pt').input_ids
        all_prompts.append(stylized_prompt_ids)
        all_labels.append(1)  # 1 for stylized
        
        # Get global and semantic tokens for neutral speech
        global_token_ids, semantic_token_ids = model.audio_tokenizer.tokenize(str(neutral_file))
        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])
        semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()])
        
        # for neutral speech, we don't have transcript text
        neutral_prompt = (
            "<|tts|><|start_content|>" + transcript_text + "<|end_content|>"
            "<|start_global_token|>" + global_tokens + "<|end_global_token|>"
        )
        neutral_prompt_ids = model.tokenizer(neutral_prompt, return_tensors='pt').input_ids
        all_prompts.append(neutral_prompt_ids)
        all_labels.append(0)  # 0 for neutral
    
    return all_prompts, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='SparkTTS')
    parser.add_argument('--stylized_dir', type=str, required=True, help='directory with stylized speech files')
    parser.add_argument('--neutral_dir', type=str, required=True, help='directory with neutral speech files')
    parser.add_argument('--transcript_file', type=str, required=True, help='JSON file mapping filenames to transcripts')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, required=True, help='local directory with model data')
    parser.add_argument("--save_dir", type=str, default="activations", help='local directory to save activations')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Loading SparkTTS model from {args.model_dir}")
    model = SparkTTS(args.model_dir, device)
    for name, module in model.model.named_modules():
        print(name)

    # sys.exit()
    
    # Load transcripts
    print(f"Loading transcripts from {args.transcript_file}")
    with open(args.transcript_file, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    

    print("Tokenizing speech data")
    prompts, labels = tokenized_speech_dataset(args.stylized_dir, args.neutral_dir, transcript_data, model)
    print(f"Processed {len(prompts)}")
    
    # # 打印prompts
    # print(prompts[0].shape) # [1, 249]
    # print(prompts[1].shape) # [1, 55]
    # print(prompts[2].shape) # [1, 256]
    # print(prompts[3].shape) # [1, 55]
    # print(labels)
    

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    Qwen = model.model
    for prompt in tqdm(prompts):
        head_wise_activations = get_sparktts_activations(Qwen, prompt, device)
        
        # Get activations for the last token
        head_wise_activations_wanted = head_wise_activations[:,-1,:].copy()
        
        # Free memory
        del head_wise_activations
        all_head_wise_activations.append(head_wise_activations_wanted)
        
        gc.collect()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Saving labels")
    np.save(os.path.join(args.save_dir, f'{args.model_name}_labels.npy'), labels)
    
    print("Saving head wise activations")
    np.save(os.path.join(args.save_dir, f'{args.model_name}_head_wise.npy'), all_head_wise_activations)


if __name__ == '__main__':
    main()
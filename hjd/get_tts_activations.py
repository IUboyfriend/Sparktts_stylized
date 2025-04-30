# python get_tts_activations.py SparkTTS --gender_stylized male --gender_neutral female --pitch_stylized low --pitch_neutral high --speed_stylized low --speed_neutral high --transcript_file "../dataset/training/train_transcription_results.json" --model_dir "../pretrained_models/Spark-TTS-0.5B"
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
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP


def get_sparktts_activations(model, prompt_ids, device,tokenizer):

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    with torch.no_grad():
        prompt = prompt_ids.to(device)
        outputs = model.generate(
            prompt,
            max_length=600, 
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Get the generated sequence as tensor
        generated_sequence = outputs.sequences
        print(generated_sequence[0].shape)
        # Get hidden states for all generated tokens
        with TraceDict(model, HEADS, retain_input=True) as ret:
            # Run forward pass with the complete sequence
            model(generated_sequence, output_hidden_states=True)
            
        head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
        
        # If you need to decode the generated sequence
        # decoded_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
        # print(generated_sequence[0])
        # print(f"Generated text: {decoded_text}")
        
    return head_wise_hidden_states




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
        # stylized_prompt = (
        #     "<|tts|><|start_content|>" + transcript_text + "<|end_content|>"
        #     "<|start_global_token|>" + global_tokens + "<|end_global_token|>"
        #     "<|start_semantic_token|>" + semantic_tokens
        # )
        stylized_prompt = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                transcript_text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        stylized_prompt = "".join(stylized_prompt)
        stylized_prompt_ids = model.tokenizer([stylized_prompt], return_tensors='pt').input_ids
        all_prompts.append(stylized_prompt_ids)
        all_labels.append(1)  # 1 for stylized
        
        # Get global and semantic tokens for neutral speech
        global_token_ids, semantic_token_ids = model.audio_tokenizer.tokenize(str(neutral_file))
        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])
        semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()])
        
        # for neutral speech
        # neutral_prompt = (
        #     "<|tts|><|start_content|>" + transcript_text + "<|end_content|>"
        #     "<|start_global_token|>" + global_tokens + "<|end_global_token|>"
        #     "<|start_semantic_token|>" + semantic_tokens
        # )

        neutral_prompt = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                transcript_text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        neutral_prompt = "".join(neutral_prompt)
        neutral_prompt_ids = model.tokenizer([neutral_prompt], return_tensors='pt').input_ids
        all_prompts.append(neutral_prompt_ids)
        all_labels.append(0)  # 0 for neutral
    
    return all_prompts, all_labels

def tokenized_speech_dataset_with_stylized(gender_stylized, gender_neutral, pitch_stylized, pitch_neutral, speed_stylized, speed_neutral, transcript_data, model):

    all_prompts = []
    all_labels = []
    # Process each pair

    for filename in transcript_data:
        transcript_sample = transcript_data[filename]
        transcript_text = transcript_sample["text"]
        control_tts_inputs_stylized  = model.process_prompt_control(gender_stylized, pitch_stylized, speed_stylized, transcript_text)
        control_tts_inputs_neutral  = model.process_prompt_control(gender_neutral, pitch_neutral, speed_neutral, transcript_text)

        stylized_prompt = "".join(control_tts_inputs_stylized)
        stylized_prompt_ids = model.tokenizer([stylized_prompt], return_tensors='pt').input_ids
        all_prompts.append(stylized_prompt_ids)

        neutral_prompt = "".join(control_tts_inputs_neutral)
        neutral_prompt_ids = model.tokenizer([neutral_prompt], return_tensors='pt').input_ids
        all_prompts.append(neutral_prompt_ids)
    return all_prompts

def tokenized_speech_dataset(gender_neutral, pitch_neutral, speed_neutral, transcript_data, model):

    all_prompts = []
    # Process each pair

    for filename in transcript_data:
        transcript_sample = transcript_data[filename]
        transcript_text = transcript_sample["text"]
        control_tts_inputs_neutral  = model.process_prompt_control(gender_neutral, pitch_neutral, speed_neutral, transcript_text)

        neutral_prompt = "".join(control_tts_inputs_neutral)
        neutral_prompt_ids = model.tokenizer([neutral_prompt], return_tensors='pt').input_ids
        all_prompts.append(neutral_prompt_ids)
    return all_prompts




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='SparkTTS')
    # parser.add_argument('--stylized_dir', type=str, required=True, help='directory with stylized speech files')
    # parser.add_argument('--neutral_dir', type=str, required=True, help='directory with neutral speech files')
    parser.add_argument("--gender_stylized", choices=["male", "female"])
    parser.add_argument("--gender_neutral", choices=["male", "female"])
    parser.add_argument(
        "--pitch_stylized", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    parser.add_argument(
        "--pitch_neutral", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    parser.add_argument(
        "--speed_stylized", choices=["very_low", "low", "moderate", "high", "very_high"]
    )   
    parser.add_argument(
        "--speed_neutral", choices=["very_low", "low", "moderate", "high", "very_high"]
    )   
    parser.add_argument('--transcript_file', type=str, required=True, help='JSON file mapping filenames to transcripts')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, required=True, help='local directory with model data')
    parser.add_argument("--save_dir", type=str, default="activations", help='local directory to save activations')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Loading SparkTTS model from {args.model_dir}")
    model = SparkTTS(args.model_dir, device)
    # for name, module in model.model.named_modules():
    #     print(name)

    # sys.exit()
    
    # Load transcripts
    print(f"Loading transcripts from {args.transcript_file}")
    with open(args.transcript_file, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    

    print("Tokenizing speech data")
    prompts = tokenized_speech_dataset_with_stylized(gender_stylized = args.gender_stylized, gender_neutral = args.gender_neutral, pitch_stylized = args.pitch_stylized, pitch_neutral = args.pitch_neutral, speed_stylized = args.speed_stylized, speed_neutral = args.speed_neutral, transcript_data = transcript_data, model = model)
    print(f"Processed {len(prompts)}")
    # print(prompts[0].shape) # stylized prompt
    # print(prompts[1].shape) # neutral prompt
    # print(prompts[2].shape) # stylized prompt
    # print(prompts[3].shape) # neutral prompt
    # # 打印prompts
    # print(prompts[0].shape) # [1, 249]
    # print(prompts[1].shape) # [1, 55]
    # print(prompts[2].shape) # [1, 256]
    # print(prompts[3].shape) # [1, 55]
    # print(labels)
    

    all_head_wise_activations = []

    print("Getting activations")
    tokenizer = model.tokenizer
    Qwen = model.model
    count = 0
    for prompt in tqdm(prompts):
        if count == 200:
            break
        else:
            count+=1
        head_wise_activations = get_sparktts_activations(Qwen, prompt, device, tokenizer)
        print(head_wise_activations.shape)
        # Get activations for the last token
        head_wise_activations_wanted = head_wise_activations[:,-1,:].copy()

        # print(head_wise_activations_wanted.shape)
        # Free memory
        del head_wise_activations
        all_head_wise_activations.append(head_wise_activations_wanted)
        
        gc.collect()

    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Saving head wise activations")
    np.save(os.path.join(args.save_dir, f'{args.model_name}_head_wise.npy'), all_head_wise_activations)


if __name__ == '__main__':
    main()
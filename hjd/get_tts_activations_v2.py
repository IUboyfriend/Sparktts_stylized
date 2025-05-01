# python get_tts_activations_v2.py SparkTTS --gender_stylized male --gender_neutral female --pitch_stylized low --pitch_neutral high --speed_stylized low --speed_neutral high --transcript_file "../dataset/training/train_transcription_results.json" --model_dir "../pretrained_models/Spark-TTS-0.5B" --save_dir "activations_v2"
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


def get_sparktts_activations_v2(model, prompt_ids, device,tokenizer):
    end_gloabl_token = 165156
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    prompt = prompt_ids.to(device)
    size = prompt.shape[1] # text+ attributes tokens
    print(prompt.shape)
    global_end_pos = size + 38
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
    # print(tokenizer.decode(generated_sequence[0])[-1])
    # Get hidden states for all generated tokens
    with TraceDict(model, HEADS, retain_input=True) as ret:
        # Run forward pass with the complete sequence
        model(generated_sequence, output_hidden_states=True)
        
    head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
    head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
    # print(head_wise_hidden_states.shape)
    # print(generated_sequence[0].shape)
    print("generated_sequence[0][global_end_pos-1]",generated_sequence[0][global_end_pos-1])
    # If you need to decode the generated sequence
    # decoded_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    # print(generated_sequence[0])
    # print(f"Generated text: {decoded_text}")
        
    # print(tokenizer.decode(prompt[0]))
    # with torch.no_grad():
    #     for _ in range(global_end_pos):  # max length for safety
    #         # Forward pass
    #         outputs = model(prompt)
    #         next_token_logits = outputs.logits[:, -1, :]
    #         probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    #         token = torch.tensor([[probs.argmax().item()]]).to(device)
    #         prompt = torch.cat((prompt, token), dim=1)
    #         # Check if this is the target token
    #         if token.item() == end_gloabl_token:
    #             print(f"Target token {end_gloabl_token} found!")
    #             print(prompt[0].shape)
    #             print(tokenizer.decode(prompt[0]))
                
                # with TraceDict(model, HEADS, retain_input=True) as ret:
                    # We need to run forward on the entire sequence that produced this token
                    # to get the accurate activations for the token
                    # print(prompt)
                    # model(prompt, output_hidden_states=True)
                # head_wise_hidden_states_global = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
                # head_wise_hidden_states_global = torch.stack(head_wise_hidden_states_global, dim=0).squeeze().numpy()
                # print(f"Collected activations of shape: {head_wise_hidden_states_global.shape}")   
            # elif token.item() == 151643 or token.item() == 151644 or token.item() == 151645:
            #     print(tokenizer.decode(prompt[0]))
            #     print(tokenizer.decode(token.cpu().numpy()[0][0]))
            #     break
    return global_end_pos, head_wise_hidden_states


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
    

    all_head_wise_activations_global = []
    all_head_wise_activations_semantic = []

    print("Getting activations")
    tokenizer = model.tokenizer
    Qwen = model.model
    count = 0
    for prompt in tqdm(prompts):
        if count == 100:
            break
        else:
            count+=1
        global_end_pos, head_wise_activations = get_sparktts_activations_v2(Qwen, prompt, device, tokenizer)
        print(head_wise_activations.shape)
        # Get activations for the last token
        head_wise_activations_global = head_wise_activations[:,global_end_pos-1,:].copy()
        head_wise_activations_semantic = head_wise_activations[:,-1,:].copy()

        # print(head_wise_activations_wanted.shape)
        # Free memory
        del head_wise_activations
        all_head_wise_activations_global.append(head_wise_activations_global)
        all_head_wise_activations_semantic.append(head_wise_activations_semantic)
        
        gc.collect()

    os.makedirs(args.save_dir, exist_ok=True)
    # print("all_head_wise_activations_global", len(all_head_wise_activations_global))
    # print("all_head_wise_activations_semantic", len(all_head_wise_activations_semantic))
    # print("all_head_wise_activations_global[0]", all_head_wise_activations_global[0].shape)
    # print("all_head_wise_activations_semantic[0]", all_head_wise_activations_semantic[0].shape)
    print("Saving head wise activations")
    np.save(os.path.join(args.save_dir, f'{args.model_name}_head_wise_global.npy'), all_head_wise_activations_global)
    np.save(os.path.join(args.save_dir, f'{args.model_name}_head_wise_semantic.npy'), all_head_wise_activations_semantic)


if __name__ == '__main__':
    main()
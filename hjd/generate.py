import os
import json
import argparse
import torch
import torch.nn.functional as F
import time
from einops import rearrange
import sys
import numpy as np
from utils import get_activations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cli.SparkTTS import SparkTTS
from hjd.get_tts_activations import tokenized_speech_dataset
import soundfile as sf
import re

# global variables to share across functions
SRC_ACTIVATIONS = None  # source activations, shape: [num_samples, num_layers, num_heads, head_dim]
TGT_ACTIVATIONS = None  # target activations, shape: [num_samples, num_layers, num_heads, head_dim]
SS_RANK = {}  # style subspace rank
SS_VH = {}  # style subspace Vh
SS_PROJ_SRC_MEAN_ACT = {}  # style subspace projection of source activations
SS_PROJ_TGT_MEAN_ACT = {}  # style subspace projection of target activations
SELECTED_HEADS_BY_LAYER = {}  # layer -> head, selected heads


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_path", type=str, required=True)
    parser.add_argument("--test_folder",type=str,required=True)
    parser.add_argument("--selected_heads_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--gender_neutral", choices=["male", "female"],default="female")
    parser.add_argument(
        "--pitch_neutral", choices=["very_low", "low", "moderate", "high", "very_high"],default="high"
    )
    parser.add_argument(
        "--speed_neutral", choices=["very_low", "low", "moderate", "high", "very_high"],default="high"
    )   
    parser.add_argument("--output_folder", type=str, default="generated_audio_method_1", help="Folder to save generated audio files")
    # parser.add_argument("--save_dir", type=str, required=True)
    # # rank selection
    rank_group = parser.add_mutually_exclusive_group()
    rank_group.add_argument("--rank", type=int)
    rank_group.add_argument("--adaRank", action="store_true")
    parser.add_argument("--var_threshold", type=float, default=0.5)
    # acceleration
    parser.add_argument("--generation_method", type=str, choices=["baseline", "fast", "faster"], required=True)
    # # KNN
    # parser.add_argument("--KNN_neighbor_num", type=int, default=None)
    return parser.parse_args()

def search_rank_BIC(X, U, s, Vh, var_threshold, constrain_range=False):
    N, D = X.shape

    # calculate explained variance
    var_explained = (s**2) / torch.sum(s**2)
    cumulative_var_explained = torch.cumsum(var_explained)
    
    # find the minimum rank value that satisfies the threshold
    r_var = torch.argmax(cumulative_var_explained >= var_threshold) + 1
    
    # define search range
    r_min = r_var if constrain_range else 1
    r_max = s.shape[0] - 1
    
    # search for the optimal rank
    best_r = r_min
    best_BIC = float('inf')
    
    for r in range(r_min, r_max + 1):
        # use the first r singular values and vectors to reconstruct the data
        X_reconstructed = U[:, :r] @ torch.diag(s[:r]) @ Vh[:r, :]
        
        # calculate the mean square error
        MSE = torch.mean((X - X_reconstructed) ** 2)
        
        # calculate the BIC
        # BIC = N * D * log(MSE) + r * (N + D + 1) * log(N * D)
        BIC = N * D * torch.log(MSE) + r * (N + D + 1) * torch.log(N * D)
        
        if BIC < best_BIC:
            best_BIC = BIC
            best_r = r

    return best_r, best_BIC

def svd_decomposition(rank=None, adaRank=False, var_threshold=None):
    # either rank or adaRank
    assert rank is not None or (adaRank is not None and var_threshold is not None)
    if adaRank:
        print("adaRank with var_threshold =", var_threshold)
    else:
        print("rank =", rank)

    global SS_RANK, SS_VH, SS_PROJ_SRC_MEAN_ACT, SS_PROJ_TGT_MEAN_ACT

    for layer_idx in SELECTED_HEADS_BY_LAYER:
        for head_idx in SELECTED_HEADS_BY_LAYER[layer_idx]:
            src_activations = SRC_ACTIVATIONS[:, layer_idx, head_idx, :]
            tgt_activations = TGT_ACTIVATIONS[:, layer_idx, head_idx, :]
            # print(src_activations.shape) # [792, 64]

            # SVD
            delta_activations = tgt_activations - src_activations
            U, s, Vh = torch.linalg.svd(delta_activations.float(), full_matrices=False)
            U, s, Vh = U.to(src_activations.dtype), s.to(src_activations.dtype), Vh.to(src_activations.dtype)

            # projection of activations in the style subspace
            proj_src_activations = torch.matmul(src_activations, Vh.T)
            proj_tgt_activations = torch.matmul(tgt_activations, Vh.T)

            if adaRank:
                rank, _ = search_rank_BIC(delta_activations, U, s, Vh, var_threshold)

            SS_RANK[(layer_idx, head_idx)] = rank
            SS_VH[(layer_idx, head_idx)] = Vh
            # print(SS_VH[(layer_idx, head_idx)].shape) # [k = 64, head dim = 64]
            SS_PROJ_SRC_MEAN_ACT[(layer_idx, head_idx)] = torch.mean(proj_src_activations, dim=0)
            SS_PROJ_TGT_MEAN_ACT[(layer_idx, head_idx)] = torch.mean(proj_tgt_activations, dim=0)
            # print(SS_PROJ_SRC_MEAN_ACT[(layer_idx, head_idx)].shape) # [64]

def get_steering_vector(layer_idx, head_idx, cur_activations, beta=0.80):  # FIXME: beta is hard-coded
    # read from global variables
    rank = SS_RANK[(layer_idx, head_idx)]
    Vh = SS_VH[(layer_idx, head_idx)][:rank, :]
    proj_src_mean_act = SS_PROJ_SRC_MEAN_ACT[(layer_idx, head_idx)][:rank]
    proj_tgt_mean_act = SS_PROJ_TGT_MEAN_ACT[(layer_idx, head_idx)][:rank]

    # base strength, determined by the dataset
    base_strength = proj_tgt_mean_act - proj_src_mean_act

    # diff strength, determined by the current activations
    proj_cur_activations = torch.matmul(Vh, cur_activations)
    diff_strength = proj_tgt_mean_act - proj_cur_activations

    # combine
    strength = base_strength * (1 + 0.5 * torch.sign(base_strength) * diff_strength)

    steering_vector = torch.matmul(Vh.T, strength)

    # apply global scaling factor
    steering_vector = beta * steering_vector

    return steering_vector

def edit_model_bias(model, cur_activations):
    """
    model: model to edit
    cur_activations: torch tensor, shape: (num_layers, seq_len, num_heads * head_dim)
    layer_head_dict: dict, key is layer_idx, value is a list of head_idx
    """
    cur_activations = rearrange(cur_activations, 'l s (h d) -> l s h d', h=model.config.num_attention_heads)

    for layer_idx, head_idx_list in SELECTED_HEADS_BY_LAYER.items():
        displacement = torch.zeros((int(model.config.num_attention_heads), int(model.config.hidden_size / model.config.num_attention_heads)),
                                   device=model.device, dtype=model.dtype)
        for head_idx in head_idx_list:
            cur_head_activations = cur_activations[layer_idx, -1, head_idx]  # vector of shape (head_dim,)
            steering_vector = get_steering_vector(layer_idx, head_idx, cur_head_activations)
            displacement[head_idx] = steering_vector

        displacement = rearrange(displacement, 'h d -> (h d)')
        bias_tobe = F.linear(displacement, model.model.layers[layer_idx].self_attn.o_proj.weight)
        model.model.layers[layer_idx].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)

def reset_model(model):
    """reset model bias to 0"""
    for layer_idx, heads in SELECTED_HEADS_BY_LAYER.items():
        zero_bias = torch.zeros(model.config.hidden_size, dtype=model.dtype, device=model.device)
        model.model.layers[layer_idx].self_attn.o_proj.bias = torch.nn.parameter.Parameter(zero_bias)

def generate(model, question_tokens, qa_prefix_tokens, max_length=600):

    tokens_without_template = question_tokens
    tokens_with_template = qa_prefix_tokens
    answer_token_ids = []

    for _ in range(max_length):
        with torch.no_grad():
            #edit
            reset_model(model)
            cur_activations, _ = get_activations(model, tokens_without_template)
            edit_model_bias(model, cur_activations)       

            # predict next token
            outputs = model(tokens_with_template)
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to(model.device)

            # update tokens
            tokens_without_template = torch.cat((tokens_without_template, token), dim=1)
            tokens_with_template = torch.cat((tokens_with_template, token), dim=1)

            # collect answer token ids
            answer_token_ids.append(token.cpu().numpy()[0][0])
            
            # 
            if token.cpu().numpy()[0][0] == 151643 or token.cpu().numpy()[0][0] == 151644 or token.cpu().numpy()[0][0] == 151645: 
                break
    print(answer_token_ids)
    return answer_token_ids

def generate_fast(model, question_tokens, qa_prefix_tokens, max_length=600):
    """Use kv cache for base forward, but not for style forward. This implementation completely follows the
    implementation of the original paper, just using kv cache for acceleration."""
    tokens_without_template = question_tokens
    tokens_with_template = qa_prefix_tokens
    answer_token_ids = []

    past_kv_base = None

    for _ in range(max_length):
        with torch.no_grad():
            # edit
            reset_model(model)
            cur_activations, past_kv_base = get_activations(model,
                                                            tokens_without_template if past_kv_base is None else token,
                                                            use_cache=True,
                                                            past_key_values=past_kv_base)
            edit_model_bias(model, cur_activations)

            # predict next token, recalculate from the first token
            outputs = model(tokens_with_template)

            # determine next token
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to(model.device)

            # update tokens
            tokens_without_template = torch.cat((tokens_without_template, token), dim=1)
            tokens_with_template = torch.cat((tokens_with_template, token), dim=1)

            # collect answer token ids
            answer_token_ids.append(token.cpu().numpy()[0][0])

            if token.cpu().numpy()[0][0] == 151643 or token.cpu().numpy()[0][0] == 151644 or token.cpu().numpy()[0][0] == 151645: # 正常生成到151645停
                break

    return answer_token_ids

def generate_faster(model, question_tokens, qa_prefix_tokens, max_length=600):
    """Use kv cache for base forward and style forward. This implementation slightly relaxed the constraint of the original paper.
    But yields similar performance with better speed."""
    tokens_without_template = question_tokens
    tokens_with_template = qa_prefix_tokens
    answer_token_ids = []

    past_kv_base = None
    past_kv_style = None

    for _ in range(max_length):
        with torch.no_grad():
            # edit
            reset_model(model)
            cur_activations, past_kv_base = get_activations(model,
                                                            tokens_without_template if past_kv_base is None else token,
                                                            use_cache=True,
                                                            past_key_values=past_kv_base)
            edit_model_bias(model, cur_activations)

            # predict next token with kv cache
            # Note that past_kv_style here is computed by model variants
            # that have been modified in previous steps, meaning the kv cache
            # of each token is calculated with a different bias!
            outputs = model(tokens_with_template if past_kv_style is None else token,
                            use_cache=True,
                            past_key_values=past_kv_style)
            past_kv_style = outputs.past_key_values

            # determine next token
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to(model.device)

            # update tokens
            tokens_without_template = torch.cat((tokens_without_template, token), dim=1)
            tokens_with_template = torch.cat((tokens_with_template, token), dim=1)

            # collect answer token ids
            answer_token_ids.append(token.cpu().numpy()[0][0])

            if token.cpu().numpy()[0][0] == 151643 or token.cpu().numpy()[0][0] == 151644 or token.cpu().numpy()[0][0] == 151645: 
                break

    return answer_token_ids

def inspect_model_biases(model):
    """
    Print the o_proj bias status and values for each layer
    """
    print("Inspecting o_proj biases across all layers:")
    for layer_idx, layer in enumerate(model.model.layers):
        # Check if bias exists
        has_bias = hasattr(layer.self_attn.o_proj, 'bias') and layer.self_attn.o_proj.bias is not None

        print(hasattr(layer.self_attn.o_proj, 'bias'))
        print(layer.self_attn.o_proj.bias is not None)

def main(args):
    # load model
    print("Loading model...")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = SparkTTS(args.model_dir, device)
    tokenizer = model.tokenizer
    audio_tokenizer = model.audio_tokenizer
    model.model.config.oproj_bias = True
    
    # print("Initial bias state:")
    # inspect_model_biases(model)

    # load testing dataset
    # find the file end with .json in the test_folder
    for file in os.listdir(args.test_folder):
        if file.endswith(".json"):
            transcript_file = os.path.join(args.test_folder, file)
            break   
    with open(transcript_file, 'r', encoding='utf-8') as f:
        print(f"Loading transcripts from {transcript_file}")
        transcript_data = json.load(f)
        
    print("Loading/Processing dataset...")

    prompts = tokenized_speech_dataset(
        gender_neutral=args.gender_neutral,
        pitch_neutral=args.pitch_neutral,
        speed_neutral=args.speed_neutral,
        transcript_data=transcript_data,
        model=model
    )
    print(f"Processed {len(prompts)} prompts")


    # load activations
    large_model = model
    model = model.model # turn the model to qwen
    global SRC_ACTIVATIONS, TGT_ACTIVATIONS
    print("Loading activations...")
    activations = np.load(args.activations_path)
    activations = torch.from_numpy(activations).to(model.device)
    source_indices = list(range(1, activations.shape[0], 2))
    target_indices = list(range(0, activations.shape[0], 2))
    source_activations = activations[source_indices]
    target_activations = activations[target_indices]
    source_activations = source_activations.to(model.device)
    SRC_ACTIVATIONS = rearrange(source_activations, 'b l (h d) -> b l h d', h=model.config.num_attention_heads)
    target_activations = target_activations.to(model.device)
    TGT_ACTIVATIONS = rearrange(target_activations, 'b l (h d) -> b l h d', h=model.config.num_attention_heads)
    print(SRC_ACTIVATIONS.shape)  # [792, 24, 14, 64]
    print(TGT_ACTIVATIONS.shape)  # [792, 24, 14, 64]


    # load selected heads
    global SELECTED_HEADS_BY_LAYER
    print("Loading selected heads...")
    selected_heads = np.load(args.selected_heads_path, allow_pickle=True)
    # If it's stored as an object array, convert it to a list of tuples
    if isinstance(selected_heads, np.ndarray):
        selected_heads = [(int(layer), int(head)) for layer, head in selected_heads]
    # group by layer for convenience and efficiency
    for layer_idx, head_idx in selected_heads:
        if layer_idx not in SELECTED_HEADS_BY_LAYER:
            SELECTED_HEADS_BY_LAYER[layer_idx] = []
        SELECTED_HEADS_BY_LAYER[layer_idx].append(head_idx)
    print(SELECTED_HEADS_BY_LAYER)

    # determine the style subspace
    print("Determining the style subspace...")
    svd_decomposition(rank=args.rank, adaRank=args.adaRank, var_threshold=args.var_threshold)

    # generate
    print("Start generating...")

    if args.generation_method == "baseline":
        generate_method = generate
    elif args.generation_method == "fast":
        generate_method = generate_fast
    elif args.generation_method == "faster":
        generate_method = generate_faster

    print(f"Generation method: {args.generation_method}")

    model.eval()

    cum_time = 0
    cum_token = 0
    answers = []

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    
    prompts_neutral = tokenized_speech_dataset(args.gender_neutral, args.pitch_neutral, args.speed_neutral, transcript_data, large_model)
    count = 0
    for index, sample_neutral in enumerate(prompts_neutral):

        #########################################################这个地方可能需要修改 #########################################################
    
        tik = time.time()
        response = generate_method(model, sample_neutral.to(model.device), sample_neutral.to(model.device))
        # use model.tokenizer to detokenize the response
        response = tokenizer.decode(response, skip_special_tokens=True)
        print(response)
        

        time_cost = time.time() - tik
        cum_time += time_cost
        cum_token += len(response)


    # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", response)])
            .long()
            .unsqueeze(0)
        )

        global_token_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", response)])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        wav = audio_tokenizer.detokenize(
            global_token_ids.to(device).squeeze(0),
            pred_semantic_ids.to(device),
        )

        
        count += 1
        # Get the filename from transcript_data
        filename = list(transcript_data.keys())[index]
        # Clean filename to make it safe for filesystem
        safe_filename = ''.join(c for c in filename if c.isalnum() or c in '._- ')
        # Save with original filename in specified output folder
        sf.write(os.path.join(args.output_folder, f"stylized_{safe_filename}"), wav, 16000)

      


    









    # print(f"Speech number: {len(prompts_stylized)}\n"
    #     f"Cumulative token number: {cum_token}\n"
    #     f"Time cost: {cum_time} s\n"
    #     f"Average generation speed: {cum_token/cum_time} token/s")

    # # save results
    # print("Saving results...")
    # output_data = []
    # for i in range(len(answers)):
    #     dict = {}
    #     dict["question"] = prompts[i]
    #     dict["daiyu_answer"] = answers[i]  # FIXME: to remove
    #     dict["model_path"] = args.model_dir  # FIXME: to remove
    #     output_data.append(dict)

    # os.makedirs(args.save_dir, exist_ok=True)

    # output_path = args.save_dir + "/results.json"
    # with open(output_path, 'w', encoding='utf-8') as file:
    #     json.dump(output_data, file, ensure_ascii=False, indent=4)

    # output_path = args.save_dir + "/speed.txt"
    # with open(output_path, 'w', encoding='utf-8') as file:
    #     file.write(f"Question number: {len(prompts)}\n"
    #     f"Cumulative token number: {cum_token}\n"
    #     f"Time cost: {cum_time} s\n"
    #     f"Average generation speed: {cum_token/cum_time} token/s")

    # print("Results saved to", args.save_dir)




if __name__ == "__main__":
    args = parse_args()
    main(args)



# python generate.py --model_dir "../pretrained_models/Spark-TTS-0.5B" --activations_path "activations/SparkTTS_head_wise.npy" --test_folder "../dataset/last100_data" --selected_heads_path "edited_weights/top_heads_64_3.0.npy" --rank 64 --generation_method "baseline" 


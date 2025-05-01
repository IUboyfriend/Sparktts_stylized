# python edit_weight.py --model_name Qwen2.5-0.5B --activation_path "activations_v2/SparkTTS_head_wise_semantic.npy" --model_dir "../pretrained_models/Spark-TTS-0.5B" --num_heads 40 --alpha 3 --save_dir "edited_weights_semantic/" --selection_method "linear_probing" 
# python edit_weight.py --model_name Qwen2.5-0.5B --activation_path "activations_v2/SparkTTS_head_wise_global.npy" --model_dir "../pretrained_models/Spark-TTS-0.5B" --num_heads 192 --alpha 3 --save_dir "edited_weights_global/" --selection_method "linear_probing" 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append('../')
from cli.SparkTTS import SparkTTS
# from utils import (
#     alt_tqa_evaluate,
#     layer_head_to_flattened_idx,
#     get_interventions_dict,
#     get_top_heads,
#     get_separated_activations,
#     get_com_directions,
#     get_top_heads_group_lasso,
#     get_top_heads_heuristic,
#     get_top_heads_heuristic_v2,
#     get_top_heads_mmd,
#     get_top_heads_lda_ratio,
#     get_top_heads_cluster
# )

from utils import (
    get_interventions_dict,
    alt_tqa_evaluate,
    layer_head_to_flattened_idx,
    get_top_heads,
    get_separated_activations,
    get_com_directions,
)

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Qwen2.5-0.5B', help='model name')
    parser.add_argument("--activation_path", type=str, default=None, help='activation path')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--save_dir", type=str, default="edited_model", help='directory to save the edited model')
    # 以上为必需参数
    parser.add_argument('--num_heads', type=int, default=96, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=5, help='alpha, intervention strength')
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--collaborative_selection', action='store_true', help='use collaborative selection', default=False)
    parser.add_argument('--selection_method', type=str, choices=['group_lasso', 'linear_probing', 'heuristic', 'heuristic_v2', 'mmd', 'lda_ratio', 'cluster'], default='linear_probing', help='head selection method')
    parser.add_argument('--l0_layer', type=int, default=0, help='start layer')
    parser.add_argument('--ln_layer', type=int, default=40, help='end layer')
    parser.add_argument('--select_svd_components', type=int, default=64, help='number of SVD components to select')
    parser.add_argument('--l1_reg', type=float, default=0.0, help='L1 regularization')
    parser.add_argument('--group_reg', type=float, default=0.05, help='group regularization')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()

    # set seeds
    print("set seeds")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    print("create model")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = SparkTTS(args.model_dir, device)
    tokenizer = model.tokenizer
    audio_tokenizer = model.audio_tokenizer
    model = model.model


    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(num_layers,num_heads)


    # load activations 
    print("load activations")
    head_wise_activations = np.load(f"{args.activation_path}")
    length = len(head_wise_activations)
    labels = []
    for i in range(length):
        labels.append(1 if i % 2 == 0 else 0)

    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads) # (2264, 24, 14, 64)
    # print(head_wise_activations.shape)
    
    dataset_len = head_wise_activations.shape[0] // 2 # 1132
    print(dataset_len) 

    # tuning dataset: no labels used, just to get std of activations along the direction
    tuning_activations = np.load(f"{args.activation_path}")
    tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)


    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations) 
    # list, each (2, num_layers, num_heads, head_dim)
    
    
    print(dataset_len)
    train_idxs = np.arange(dataset_len)
    print(train_idxs.shape)

    # pick a val set using numpy
    train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
    print(train_set_idxs.shape) #905
    print(val_set_idxs.shape) #227

    print("getting top heads")
    # get directions
    com_directions = None
    print("selection method: ", args.selection_method)
    if args.selection_method == 'group_lasso':
        top_heads, probes = get_top_heads_group_lasso(train_set_idxs, 
                                                      val_set_idxs, 
                                                      separated_head_wise_activations, 
                                                      separated_labels, 
                                                      num_layers, 
                                                      num_heads, 
                                                      args.seed, 
                                                      args.num_heads, 
                                                      args.use_random_dir, 
                                                      l0_layer=args.l0_layer, 
                                                      ln_layer=args.ln_layer,
                                                      svd_components=args.select_svd_components,
                                                      l1_reg=args.l1_reg,
                                                      group_reg=args.group_reg)
    elif args.selection_method == 'linear_probing':
        top_heads, probes = get_top_heads(train_set_idxs, 
                                          val_set_idxs, 
                                          separated_head_wise_activations, 
                                          separated_labels, 
                                          num_layers, 
                                          num_heads, 
                                          args.seed, 
                                          args.num_heads, 
                                          args.use_random_dir)
    elif args.selection_method == 'heuristic':
        top_heads, probes = get_top_heads_heuristic(train_set_idxs,
                                                    val_set_idxs,
                                                    separated_head_wise_activations,
                                                    separated_labels,
                                                    num_layers,
                                                    num_heads,
                                                    args.seed,
                                                    args.num_heads,
                                                    args.use_random_dir,
                                                    pa_threshold=0.6,
                                                    ds_percentile_threshold=0.1,
                                                    ds_metric='norm')
    elif args.selection_method == 'heuristic_v2':
        top_heads, probes = get_top_heads_heuristic_v2(train_set_idxs,
                                                        val_set_idxs,
                                                        separated_head_wise_activations,
                                                        separated_labels,
                                                        num_layers,
                                                        num_heads,
                                                        args.seed,
                                                        args.num_heads,
                                                        args.use_random_dir,
                                                        ds_percentile_threshold=0.1,
                                                        dc_threshold=0.6,
                                                        ds_metric='norm')
    elif args.selection_method == 'mmd':
        top_heads, probes = get_top_heads_mmd(train_set_idxs,
                                              val_set_idxs,
                                              separated_head_wise_activations,
                                              separated_labels,
                                              num_layers,
                                              num_heads,
                                              args.seed,
                                              args.num_heads,
                                              args.use_random_dir,
                                              use_median_heuristic=True,
                                              default_sigma=1.0,
                                              mmd_batch_size=None)
    elif args.selection_method == 'lda_ratio':
        top_heads, probes = get_top_heads_lda_ratio(train_set_idxs,
                                                    val_set_idxs,
                                                    separated_head_wise_activations,
                                                    separated_labels,
                                                    num_layers,
                                                    num_heads,
                                                    args.seed,
                                                    args.num_heads,
                                                    args.use_random_dir,
                                                    epsilon=1e-9)
    elif args.selection_method == 'cluster':
        top_heads, probes = get_top_heads_cluster(train_set_idxs,
                                                  val_set_idxs,
                                                  separated_head_wise_activations,
                                                  separated_labels,
                                                  num_layers,
                                                  num_heads,
                                                  args.seed,
                                                  args.num_heads,
                                                  args.use_random_dir,
                                                  top_k_consistency=400,
                                                  epsilon=1e-9)

    os.makedirs(args.save_dir, exist_ok=True)
    # np.save(os.path.join(args.save_dir, f'probes_{args.num_heads}_{args.alpha:.1f}.npy'),probes)
    # print(top_heads)
    np.save(os.path.join(args.save_dir, f'top_heads_{args.num_heads}_{args.alpha:.1f}.npy'),top_heads)

    # 这里的interventions中得到的intervention向量实际上没有用到, only used for recording the selected heads
    interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)
    # print(interventions)
    # print selected layers and heads
    # for intervention in sorted(interventions.keys(), key=lambda x: int(x.split('.')[2])):
    #     print(intervention)
    #     print([
    #         head_no
    #         for head_no, _, _ in interventions[intervention]
    #     ])

    activations_dict = {} # save
    for head_out_name, list_int_vec in tqdm(interventions.items()):
        layer_no = int(head_out_name.split('.')[2])
        print(layer_no)
        displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
        activations_dict[layer_no] = {} # save
        for head_no, head_vec, std in list_int_vec:

            activations = tuning_activations[:,layer_no,head_no,:]
            correct_activations = activations[::2, :]
            incorrect_activations = activations[1::2, :]
            # print(correct_activations.shape)
            # print(incorrect_activations.shape) # (1132, 64)
            correct_activations = np.mean(correct_activations, axis=0)
            incorrect_activations = np.mean(incorrect_activations, axis=0)
            # 真正用到的intervention向量在这里计算
            displacement[head_no] = args.alpha * (correct_activations - incorrect_activations)
            
            activations_dict[layer_no][head_no] = displacement[head_no] # save
      
        device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
        displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
        # FIXME: 原本的o_proj.bias中，没有被选中的heads的bias被覆盖为0; 还是说本身就为零？
        bias_tobe = F.linear(displacement.to(torch.float32), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
        model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)
    # with open(os.path.join(args.save_dir, f'activations_{args.num_heads}_{args.alpha:.1f}.pkl'), 'wb') as f:
    #     pickle.dump(activations_dict, f)

    # print("saving model with edited weights")
    # save_folder = os.path.join(args.save_dir, f'{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha:.1f}')
    # if os.path.exists(save_folder):
    #   shutil.rmtree(save_folder)
    # os.makedirs(save_folder)
    # print("saving model to", save_folder)
    # model.config.oproj_bias = True
    # model.save_pretrained(save_folder, safe_serialization=False, max_shard_size="10GB")
    # tokenizer.save_pretrained(save_folder)


if __name__ == "__main__":
    main()

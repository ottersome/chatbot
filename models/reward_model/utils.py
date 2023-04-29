import argparse
import os
import numpy as np
import random
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_training",
                        dest='do_training',
                        default=True,
                        type=bool)
    parser.add_argument("--batch_size_per_gpu",
                        dest='batch_size_per_gpu', default=1,
                        type=int)
    parser.add_argument("--gradient_accum_steps",
                        dest='gradient_accumulation_steps',
                        default=1,
                        type=int)
    parser.add_argument("--msteps_validation",
                        dest='msteps_validation',
                        default=10,
                        type=int)
    parser.add_argument("--device",
                        dest='device',
                        default='cuda',
                        type=str)
    # parser.add_argument("--ds_path",
                        # dest='dataset_path',
                        # required=True,
                        # help='Path to your dataset for fine-tuning',
                        # type=str)
    parser.add_argument("--logging_steps",
                        dest='logging_steps',
                        default=1000,
                        type=int)
    parser.add_argument("--output_dir",
                        dest='output_dir',
                        default='./output/',
                        type=str)
    parser.add_argument("--checkpoint_interval",
                        dest='checkpoint_interval',
                        default=1,
                        type=int)
    parser.add_argument("--checkpoint_path",
                        dest='checkpoint_path',
                        default="",
                        type=str)
    parser.add_argument("--model_type",
                        dest='model_type',
                        default='gptj',
                        type=str)
    parser.add_argument("--config_name",
                        dest='config_name',
                        #  default='microsoft/DialoGPT-medium',
                        default='EleutherAI/gpt-j-6B',
                        type=str)
    parser.add_argument("--tokenizer_name",
                        dest='tokenizer_name',
                        #  default='microsoft/DialoGPT-medium',
                        default='EleutherAI/gpt-j-6B',
                        type=str)
    parser.add_argument("--model_name_or_path",
                        dest='model_name_or_path',
                        #  default='microsoft/DialoGPT-medium',
                        default='EleutherAI/gpt-j-6B',
                        type=str)
    parser.add_argument("--block_size",
                        dest='block_size',
                        default=512,
                        type=int)
    parser.add_argument("--fp16_opt_level",
                        dest='fp16_opt_level',
                        default='O1',
                        type=str)
    parser.add_argument("--learning_rate",
                        dest='learning_rate',
                        default=1e-4,
                        type=float)
    parser.add_argument("--adam_epsilon",
                        dest='adam_epsilon',
                        default=1e-8,
                        type=float)
    parser.add_argument("--fp_16",
                        dest='fp_16',
                        default=False,
                        type=bool)
    parser.add_argument("--seed",
                        dest='seed',
                        default=120,
                        type=int)
    parser.add_argument("--overwrite_cached",
                        dest='overwrite_cached',
                        default=False,
                        type=bool)
    parser.add_argument("--cache_dir",
                        dest='cache_dir',
                        default='./.my_cache',
                        type=str)
    parser.add_argument("--weight_decay",
                        dest='weight_decay',
                        default=0.0,
                        type=float)
    parser.add_argument("--warmup_steps",
                        dest='warmup_steps',
                        default=0.0,
                        type=float)
    parser.add_argument("--gradient_accumulation_steps",
                        dest='gradient_accumulation_steps',
                        default=1,
                        type=int)
    parser.add_argument("--max_steps",
                        dest='max_steps',
                        default=-1,
                        type=int)
    parser.add_argument("--num_train_epochs",
                        dest='num_train_epochs',
                        default=2,
                        type=int)
    return parser.parse_args()

# For our particular dataset we want it to have all the context it has so far



         
def get_mask(batch, pad_token, eos_token):
    mask = torch.zeros_like(batch)
    for i,row in enumerate(batch):
        eos_tokens = np.where(row == eos_token)[0]
        if len(eos_tokens) == 0 :  
            print("Your eos token is :", eos_token)
            print("Eos Tokens are : ", eos_tokens)
            print("Your problem : \n",batch[row])
        assert len(eos_tokens) != 0, print('There is no eos token in the utterance')
        last_eos = eos_tokens[-1]
        mask[i,0:last_eos+1] = 1
    return mask



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if(torch.cuda.device_count() > 0):
        torch.cuda.manual_seed_all(seed)

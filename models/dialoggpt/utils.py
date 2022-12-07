import argparse
import os
import numpy as np
import pickle
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_training",
                        dest='do_training',
                        default=True,
                        type=bool)
    parser.add_argument("--batch_size",
                        dest='batch_size',
                        default=512,
                        type=int)
    parser.add_argument("--gradient_accum_steps",
                        dest='gradient_accumulation_steps',
                        default=0,
                        type=int)
    parser.add_argument("--device",
                        dest='device',
                        default='cuda',
                        type=str)
    parser.add_argument("--ds_path",
                        dest='dataset_path',
                        required=True,
                        help='Path to your dataset for fine-tuning',
                        type=str)
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
                        default=3500,
                        type=str)
    parser.add_argument("--model_type",
                        dest='model_type',
                        default='gpt2',
                        type=str)
    parser.add_argument("--config_name",
                        dest='config_name',
                        #  default='microsoft/DialoGPT-medium',
                        default='microsoft/DialoGPT-small',
                        type=str)
    parser.add_argument("--tokenizer_name",
                        dest='tokenizer_name',
                        #  default='microsoft/DialoGPT-medium',
                        default='microsoft/DialoGPT-small',
                        type=str)
    parser.add_argument("--block_size",
                        dest='block_size',
                        default=512,
                        type=int)
    parser.add_argument("--learning_rate",
                        dest='learning_rate',
                        default=5e-5,
                        type=float)
    parser.add_argument("--adam_epsilon",
                        dest='adam_epsilon',
                        default=1e-8,
                        type=float)
    parser.add_argument("--fp_16",
                        dest='fp_16',
                        default=True,
                        type=bool)
    parser.add_argument("--seed",
                        dest='seed',
                        default=420,
                        type=int)
    parser.add_argument("--model_name_or_path",
                        dest='model_name_or_path',
                        #  default='microsoft/DialoGPT-medium',
                        default='microsoft/DialoGPT-small',
                        type=str)
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
                        default=3,
                        type=int)
    return parser.parse_args()

# Construct Conversation from a row of contexts :p
def construct_conv(row,tokenizer_of_choice: PreTrainedTokenizer):
    # The Format here will be of n colums 1 of which is a response and n-1 are just context 
    # Flatten will go take a row, go through columns, go through their words, encode them and then put them all together
    convo = []
    for col in reversed(row): 
        convo.append(tokenizer_of_choice.encode(col))
        convo.append([tokenizer_of_choice.eos_token_id])
    convo.pop(-1)# Remove the last eos
    flat_convo = [item for sublist in convo for item in sublist]
    return flat_convo


# Mostly useful wfor when we want to use RandomSampler and 隨便的 store this in a cache for later use
class DialogDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, dataframe, logger, block_size=512):
        # model_max_length represents the maximum numberof tokens a model can handle(including speicla tokens)
        # TODO understand this one right here
        # block_size = block_size  - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory = args.cache_dir
        cached_features_file = os.path.join(directory, args.model_type+"chached_lm_"+str(block_size))

        if os.path.exists(cached_features_file) and not args.overwrite_cached:
            logger.info("Loading cached features from cache file %s", cached_features_file)
            with open(cached_features_file,"rb") as filo:
                self.examples = pickle.load(filo)
        else:
            logger.info("Creating cache of features from dataset at %s", cached_features_file)
            # Actually Do wome work on the dataframe
            self.examples = []
            logger.info("Formatting Data Properly...")
            for _,row in dataframe.iterrows():
                self.examples.append(construct_conv(row,tokenizer))# Single Row of DataFrame formatted for use
            logger.info("Saving Encoded Data into file at %s", cached_features_file)
            with open(cached_features_file,"wb") as filo:
                pickle.dump(self.examples, filo, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self,idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

    def __len__(self):
        return self.len()

    def len(self):
        return len(self.examples)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if(torch.cuda.device_count() > 0):
        torch.cuda.manual_seed_all(seed)

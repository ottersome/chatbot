import argparse
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pickle
import pandas as pd
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
                        default=1,
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
                        default=500,
                        type=str)
    parser.add_argument("--model_type",
                        dest='model_type',
                        default='gpt2',
                        type=str)
    parser.add_argument("--config_name",
                        dest='config_name',
                        #  default='microsoft/DialoGPT-medium',
                        default='microsoft/DialoGPT-medium',
                        type=str)
    parser.add_argument("--tokenizer_name",
                        dest='tokenizer_name',
                        #  default='microsoft/DialoGPT-medium',
                        default='microsoft/DialoGPT-medium',
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
                        default='microsoft/DialoGPT-medium',
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
def construct_dialog(row,tokenizer_of_choice: PreTrainedTokenizer):
    # The Format here will be of n colums 1 of which is a response and n-1 are just context 
    # Flatten will go take a row, go through columns, go through their words, encode them and then put them all together
    convo = []
    for col in reversed(row): 
        convo.append(tokenizer_of_choice.encode(col))
        convo.append([tokenizer_of_choice.eos_token_id])

    convo.pop(-1)# Remove the last eos
    flat_convo = [item for sublist in convo for item in sublist]
    return flat_convo

def prepare_convo_dataset(path):
    conv_df = pd.read_csv(path)

def prepare_discussion_dataset(path,article_max_length=1024):

    art_df = pd.read_csv(path+'discussion_article.csv')
    conv_df = pd.read_csv(path+'discussion_conv.csv')
    #Get Number of Conversations
    first_idx = conv_df['conversation_id'][0]
    last_idx = conv_df['conversation_id'].iloc[-1]
    num_of_convos = first_idx - last_idx + 1


    # TODO: Do Train/Validation Split
    dialogues=[]
    for i in range(first_idx, last_idx+1):
        # Get article
        dialogue = []
        convo = conv_df[conv_df['conversation_id']==i]
        article_id = convo['article_id'].iloc[0]
        print(f"At {i}, Using article id {article_id}")
        article_text =  art_df.loc[art_df['article_id'] == article_id]['text']
        # Get all utterances of this convo
        # full_convo = convo['utterance'].str.cat(sep=' ')
        dialogue.append(article_text.tolist()[0])
        dialogue = dialogue + convo['utterance'].tolist()

        dialogues.append(dialogue)
    df = pd.DataFrame.from_records(dialogues)
    return train_test_split(df, test_size= 0.1)


def process_datasets(path,context_length):
    # Load Data Set and Give it some Context
    dataframe = pd.read_csv(path)

    # contexted_rows = []
    contexted_rows = [list(reversed(dataframe['line'][j-context_length-1:j].tolist()))  for j in range(context_length+1,len(dataframe))]

    # Create New DataFrame with this structure
    columns = ['reponse']
    columns += ['context'+str(i+1) for i in range(0,context_length)]
    dataframe = pd.DataFrame.from_records(contexted_rows,columns=columns)

    trn_df, val_df = train_test_split(dataframe, test_size = 0.1)
    # TODO: This conversation is a bit flawed since subsequent lines need not be from different characters
    # It could be a line that follows a pause. Not to mention the fact the some conversations take place in different spaces and times
    # Need better Dataset

    print("Processing dataframe")
    print(dataframe.head(2))

    return trn_df, val_df
    # Split into test and train


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
class DiscussionDataset(Dataset):
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
class RnMDialogue(Dataset):
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

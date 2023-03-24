import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BinaryFeedbackDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, args, logger,is_train = 1):

        self.is_train = 0

        # Start Caching Procedure
        directory = args.cache_dir
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(directory, args.model_type+"cached_lm_")
        self.samples = []

        # Checked for Cached Dataset
        if os.path.exists(cached_features_file) and not args.overwrite_cached:
            logger.info("Loading cached features from cache file %s", cached_features_file)
            with open(cached_features_file,"rb") as filo:
                self.samples = pickle.load(filo)
        # If not exist than build it 
        else:
            logger.info("Creating cache of features from dataset at %s", cached_features_file)

            rw_df = pd.read_parquet('../../datasets/rewardmodel/binary_reward.parquet', engine='pyarrow')

            final_ds = []
            final_ds += self.load_ds(rw_df, '<EOS>',tokenizer.eos_token)
            final_ds  = pd.DataFrame(final_ds)

            logger.info("Formatting Data Properly...")
            # Training Data in one file
            for _,row in tqdm(final_ds.iterrows(), desc="Prepping Datasets"):
                interaction = self.tokenize_strings(row.to_list(),tokenizer)
                self.samples.append(interaction)

            logger.info("Saving Encoded Data into file at %s", cached_features_file)
            with open(cached_features_file,"wb") as filo:
                pickle.dump(self.samples, filo, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # Create the Split
        self.train, self.test = train_test_split(self.samples,test_size=0.1)
        self.samples =  self.train if is_train else self.test 

    def change_mode(self,is_train: bool):
        self.is_train = is_train



    def load_ds(self,df: pd.DataFrame ,their_eos_token, our_eos_token):
        # Every Conversation is a Row.
        dialogues = []
        for _,row in df.iterrows():
            dialogues.append([row['prompt'].replace(their_eos_token, our_eos_token),
                row['chosen'].replace(their_eos_token, our_eos_token),
                row['rejected'].replace(their_eos_token, our_eos_token)])
        return dialogues

    def tokenize_strings(self,strings_list,tokenizer_of_choice: PreTrainedTokenizer):
        # The Format here will be of n colums 1 of which is a response and n-1 are just context 
        # Flatten will go take a row, go through columns, go through their words, encode them and then put them all together

        return_list = []
        for element in strings_list:
            return_list.append(tokenizer_of_choice.encode(element))

        # convo.pop(-1)# Remove the last eos
        return return_list

    def __getitem__(self,idx):
        # TODO not hard code this so much 
        good = self.samples[idx][0] + self.tokenizer.encode('<|SEP|>') + self.samples[idx][1]
        bad = self.samples[idx][0] + self.tokenizer.encode('<|SEP|>') + self.samples[idx][1]

        return torch.tensor(good, dtype=torch.long),torch.tensor(bad, dtype=torch.long)

    def __len__(self):
        return self.len()

    def len(self):
        return len(self.samples)

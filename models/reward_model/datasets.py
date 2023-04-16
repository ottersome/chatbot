import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

SPECIAL_TOKENS_DICT = {
        'guesser':'<|GUESS|>'
        }

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
            ath_df = pd.read_parquet('../../datasets/ath/train.parquet', engine='pyarrow')

            final_ds = []
            final_ds += self.load_ds(rw_df, '<EOS>',tokenizer.eos_token)
            final_ds += self.load_ds(ath_df, '<|EOS|>',tokenizer.eos_token)
            final_ds  = pd.DataFrame(final_ds)

            logger.info("Formatting Data Properly...")
            # Training Data in one file<Checked>
            samples_removed = 0
            for _,row in tqdm(final_ds.iterrows(), desc="Prepping Datasets"):
                interaction = self.tokenize_strings(row.to_list(),tokenizer)
                glen = len(interaction[0]+interaction[1])
                blen = len(interaction[0]+interaction[2])
                # We are runnign out of memory for promopts bigger than 1000
                if not(glen > 1000 or blen  > 1000):
                    self.samples.append(interaction)
                else: 
                    samples_removed += 1

            logger.info("Saving Encoded Data into file at %s", cached_features_file)
            print("Removed {} samples because they were too big".format(samples_removed))
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
        for i,element in enumerate(strings_list):
            #  if i == 0:
            if True:
                return_list.append(tokenizer_of_choice.encode(element))
            #  else:
            #      return_list.append(tokenizer_of_choice.encode(element)+tokenizer_of_choice.encode(SPECIAL_TOKENS_DICT['guesser']))

        # convo.pop(-1)# Remove the last eos
        return return_list

    def __getitem__(self,idx):
        # TODO not hard code this so much 
        #  good = self.samples[idx][0] + self.tokenizer.encode('<|SEP|>') + self.samples[idx][1]
        #  bad = self.samples[idx][0] + self.tokenizer.encode('<|SEP|>') + self.samples[idx][1]
        good = self.samples[idx][0] + self.samples[idx][1]
        bad = self.samples[idx][0] + self.samples[idx][2]
        ctx_len = len(self.samples[idx][0])
        good_len = len(self.samples[idx][1])
        bad_len = len(self.samples[idx][2])

        out = {
                "gcombo" : torch.tensor(good,dtype=torch.long),
                "bcombo": torch.tensor(bad,dtype=torch.long),
                "good_type_ids" : torch.tensor([0]*ctx_len + [1]*good_len),
                "bad_type_ids" : torch.tensor([0]*ctx_len + [1]*bad_len)
                }

        return out

    def __len__(self):
        return self.len()

    def len(self):
        return len(self.samples)

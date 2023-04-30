import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
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
        self.samples = [] # ALL Samples including 
        self.active_samples = [] # Either Training or Test Samples depending on which is activated
        self.samples_lengths = [] # Lets keep an order of sizes
        self.longest_sample = None
        self.longest_sample_length = 0 

        # Checked for Cached Dataset
        if os.path.exists(cached_features_file) and not args.overwrite_cached:
            logger.info("Loading cached features from cache file %s", cached_features_file)
            with open(cached_features_file,"rb") as filo:
                loaded_pickle = pickle.load(filo)
                self.samples = loaded_pickle['samples']
                self.samples_lengths = loaded_pickle['samples_lengths']
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
                if glen <= 1000 and blen  <= 1000:
                    self.samples.append(interaction)
                else: 
                    samples_removed += 1

            logger.info("Saving Encoded Data into file at %s", cached_features_file)
            print("Removed {} samples because they were too long".format(samples_removed))
            # Organize Sample Len
            self.samples_lengths.sort(reverse=True,key=lambda el : el[1])

            with open(cached_features_file,"wb") as filo:
                pickle.dump(
                        {"samples":self.samples,
                         "samples_lengths":self.samples_lengths}
                            , filo, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # Create the Split
        self.train, self.test = train_test_split(self.samples,test_size=0.1)
        self.active_samples =  self.train if is_train else self.test 

        self.longest_sample = self.samples[self.samples_lengths[0][0]]
        good = len(self.longest_sample[0]) +  len(self.longest_sample[1])
        bad = len(self.longest_sample[0]) +  len(self.longest_sample[2])
        self.longest_sample_length = good + bad - len(self.longest_sample[0])
        print("Longest Sample length is good:", good, " bad:",bad)

    def change_mode(self,is_train: bool):
        self.is_train = is_train

    def get_longest_batch(self, batch_size):
        assert batch_size <= len(self.samples), "Batch Size longer than dataset"
        batch = []
        print("Testing longest batch with lengths:")
        goods, bads = [],[]
        gtypeids, btypeids = [],[]
        print('Longest Sample: ', self.longest_sample_length)
        print('\tBroken Down: g', len(self.longest_sample[0] + self.longest_sample[1]),
              " & b:",len(self.longest_sample[0] + self.longest_sample[2]))

        longest_example =  self.longest_sample
        #longest_example =  self.samples[self.samples_lengths[0][0]]
        # These determine the second dimension of our batches
        gbatch_width = len(longest_example[0])+len(longest_example[1])
        bbatch_width = len(longest_example[0])+len(longest_example[2])
        print('Length of longest example: gcombo ',gbatch_width, ' bcombo', bbatch_width )

        gbatch   = [] 
        bbatch   = [] 
        gtypeids = [] 
        btypeids = [] 

        #print("Batch Dimensions are gbatch:{} and bbatch:{}".format(gbatch.shape, bbatch.shape))

        ctr = 0
        # print('Size of the samples lengths is ' , len(self.samples))
        for idx, length in self.samples_lengths:
            if ctr >= batch_size: break
            # print("\tidx_{}:length_{}".format(idx, length))
            gcombo = self.samples[idx][0] + self.samples[idx][1]
            bcombo = self.samples[idx][0] + self.samples[idx][2]

            # print('gmcombo length is ', len(gcombo), 'while bcombo length is  ', len(bcombo))
            gbatch.append(torch.tensor(gcombo))
            bbatch.append(torch.tensor(bcombo))
            gtypeids.append(torch.tensor([0]*len(self.samples[idx][0]) + [1]*len(self.samples[idx][1])))
            btypeids.append(torch.tensor([0]*len(self.samples[idx][0]) + [1]*len(self.samples[idx][2])))

            ctr += 1
        gbatch = pad_sequence(gbatch, batch_first=True) 
        bbatch = pad_sequence(bbatch, batch_first=True)
        gtypeids = pad_sequence(gtypeids, batch_first=True)
        btypeids = pad_sequence(btypeids, batch_first=True)

        batch = (gbatch,bbatch,gtypeids,btypeids)

        return batch


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
        #  good = self.active_samples[idx][0] + self.tokenizer.encode('<|SEP|>') + self.active_samples[idx][1]
        #  bad = self.active_samples[idx][0] + self.tokenizer.encode('<|SEP|>') + self.active_samples[idx][1]
        good = self.active_samples[idx][0] + self.active_samples[idx][1]
        bad = self.active_samples[idx][0] + self.active_samples[idx][2]
        ctx_len = len(self.active_samples[idx][0])
        good_len = len(self.active_samples[idx][1])
        bad_len = len(self.active_samples[idx][2])

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
        return len(self.active_samples)

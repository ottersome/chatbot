import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BotDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, args, logger,is_train = 1):

        self.is_train = 0

        # Start Caching Procedure
        directory = args.cache_dir
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

            conv_df = pd.read_csv('../../datasets/fixed_ds.csv', sep='|')
            ass_df = pd.read_parquet('../../datasets/assitantprompt.parquet', engine='pyarrow')

            final_ds = []
            final_ds += self.load_ds(conv_df)
            final_ds += self.load_ds(ass_df)
            final_ds  = pd.DataFrame(final_ds)

            logger.info("Formatting Data Properly...")
            # Training Data in one file
            for _,row in tqdm(final_ds.iterrows(), desc="Prepping Datasets"):
                dialog = self.construct_convo(row,tokenizer)
                if (len(dialog) > 0):
                    self.samples.append(dialog)# Single Row of df_train formatted for use

            logger.info("Saving Encoded Data into file at %s", cached_features_file)
            with open(cached_features_file,"wb") as filo:
                pickle.dump(self.samples, filo, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # Create the Split
        self.train, self.test = train_test_split(self.samples,test_size=0.1)
        self.samples =  self.train if is_train else self.test 

    def change_mode(self,is_train: bool):
        self.is_train = is_train

    # TODO implement
    def load_txt(self, file):
        pass


    def load_ds(self,df):

        first_idx = df['conversation_id'][0]
        last_idx = df['conversation_id'].iloc[-1]
        num_of_convos = last_idx-first_idx+1

        dialogues = []
        for i in range(first_idx,last_idx+1):
            # Get all utterances for a single dialogue example
            convo = df[df['conversation_id'] == i]
            # If Empty Convo(Yeah My Mistake in Dataset for now)
            if (len(convo) == 0):
                print("Skipping Empty Convo")
                continue
            unit_convo = convo['utterance'].tolist()
            # TODO This might wanna be a feature for when we want to load longer conversations
            ctx_wn_start = 0  # Object we will use to move context window depending on size
            convo_length = 0
            for conv in unit_convo : convo_length = convo_length + len(conv.split(' '))
            #assert convo_length < 1024# Otherwise it might be too big for the transformer
            if convo_length >= 900:
                print("Skipping convo with size {}".format(convo_length))
                continue
            dialogues.append(unit_convo)

        return dialogues

    def construct_convo(self,row,tokenizer_of_choice: PreTrainedTokenizer):
        # The Format here will be of n colums 1 of which is a response and n-1 are just context 
        # Flatten will go take a row, go through columns, go through their words, encode them and then put them all together
        convo = []

        for i,col in enumerate(row):
            if col  == None: break
            #  if i == 0:
            #      convo.append(tokenizer_of_choice.encode(
            #          "<Add some prefixed string to the conversation here if needed>"))
            convo.append(tokenizer_of_choice.encode(col))
            convo.append([tokenizer_of_choice.eos_token_id])

        # convo.pop(-1)# Remove the last eos
        final_convo = [item for sublist in convo for item in sublist]
        if len(final_convo)>1023:
            print("Skipping Dialog because size is ",len(final_convo))
            final_convo = []
        return final_convo

    def __getitem__(self,idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)

    def __len__(self):
        return self.len()

    def len(self):
        return len(self.samples)

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
            print('Formatting Data Properly')
            for _,row in dataframe.iterrows():
                dialog = construct_dialog(row,tokenizer)
                if (len(dialog) > 0):
                    self.examples.append(dialog)# Single Row of DataFrame formatted for use

            logger.info("Saving Encoded Data into file at %s", cached_features_file)
            with open(cached_features_file,"wb") as filo:
                pickle.dump(self.examples, filo, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self,idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

    def __len__(self):
        return self.len()

    def len(self):
        return len(self.examples)


def prepare_discussion_dataset(path,article_max_length=1024):

    art_df = pd.read_csv(path+'discussion_article.csv')
    conv_df = pd.read_csv(path+'discussion_conv.csv')
    #Get Number of Conversations
    first_idx = conv_df['conversation_id'][0]
    last_idx = conv_df['conversation_id'].iloc[-1]
    num_of_convos = first_idx - last_idx + 1
    #


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
    # Flatten will go a row, go through columns, go through their words, encode them and then put them all together
    convo = []
    # Revered becaus en this case fow goes
    # response -> PrevStaement -> PrePrevStatement -> ...
    for col in reversed(row): 
        convo.append(tokenizer_of_choice.encode(col))
        convo.append([tokenizer_of_choice.eos_token_id])
    convo.pop(-1)# Remove the last eos
    flat_convo = [item for sublist in convo for item in sublist]
    return flat_convo
#
# Construct Conversation from a row of contexts :p

def construct_dialog(row,tokenizer_of_choice: PreTrainedTokenizer):
    # The Format here will be of n colums 1 of which is a response and n-1 are just context 
    # Flatten will go take a row, go through columns, go through their words, encode them and then put them all together
    convo = []
    print("Constructig Dialog")
    for i,col in enumerate(row):
        if col  == None: break
        if i == 0:
            convo.append(tokenizer_of_choice.encode(
                "The following is an article and a following conversation between two people discussing it:\n"))
        convo.append(tokenizer_of_choice.encode(col))
        convo.append([tokenizer_of_choice.eos_token_id])

    convo.pop(-1)# Remove the last eos
    final_convo = [item for sublist in convo for item in sublist]
    if len(final_convo)>1023:
        print("Skipping Dialog because size is ",len(final_convo))
        final_convo = []
    return final_convo
# I am not sure if I like it like this. It seems unecessary.
# def prepare_convo_dataset(path, limit_in_size):
    # conv_df = pd.read_csv(path, sep='|')
    # # Let me get the numver of 
    # first_idx = conv_df['conversation_id'][0]
    # last_idx = conv_df['conversation_id'].iloc[-1]
    # num_of_convos = last_idx-first_idx+1

    # dialogues = []
    # for i in range(first_idx,last_idx+1):
        # # Get all utterances for a single dialogue example
        # convo = conv_df[conv_df['conversation_id'] == i]
        
        # # TODO make sure it works when you actually have exmaples that are that long
        # ctx_wn_start = 0  # Object we will use to move context window depending on size
        
        # # This should inject every single tiny context up to where the conversation is so far.
        # for j in range(1,len(convo)+1):
            # unit_convo = convo['utterance'].iloc[0:j].tolist()
            # convo_length = 0

            # for conv in unit_convo : convo_length = convo_length + len(conv.split(' '))
            # while convo_length > limit_in_size:
                # ctx_wn_start += 1
                # unit_convo = convo['utterance'].iloc[ctx_wn_start:j].tolist()
                # convo_length = 0
                # for conv in unit_convo : convo_length = convo_length + len(conv.split(' '))

            # dialogues.append(unit_convo)

    # df = pd.DataFrame.from_records(dialogues)
    # return train_test_split(df,test_size=0.1)


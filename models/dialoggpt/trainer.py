import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import pandas as pd 
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange

from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


# try:
    # from torch.utils.tensorboard import SummaryWriter
# except ImportError:
    # from tensorboardX import SummaryWriter

# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# TODO we must think about changing the way we process these prompt-> reponses
# I dont t like these proces of "adding context".
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


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int,float]:

    #Start Summary Writter
    # Pad Sequence
    batch_size = 2 # Lets leave it at that for now 
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Sample from training dataset
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,sampler=train_sampler, batch_size=batch_size, collate_fn=collate, drop_last=True)

    # Start Model and resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    no_decay = ["bias","LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

    total_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Chose Adam as optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # Scheduler with WarmpUp an then linear decrease
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
    )
    #
    # TODO: Check if checkopoints exists
    
    # Actually Train
        # Show some Logging Info 
        # Check for model checkpoint
        # Start with checkpointif available
        # Start Training Itaration
    model.zero_grad() 
    train_iterator = trange(0,int(50), desc="Epoch")
    global_step = 0
    tr_loss = 0
    for _ in train_iterator:
        within_epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(within_epoch_iterator):
            inputs, labels = (batch, batch)

            # Skip Long Examples
            if inputs.shape[1] > 1024: continue

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.train()
            outputs  = model(inputs,labels=labels)
            loss = outputs[0]

            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()

        #
def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, df_trn, df_val, prefix ="") -> Dict:
    pass


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
from utils import set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


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

    # For main gpu 
    tb_writer = SummaryWriter()

    #Start Summary Writter
    # Pad Sequence
    batch_size = 2 # Lets leave it at that for now 
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Sample from training dataset
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,sampler=train_sampler, batch_size=8, collate_fn=collate, drop_last=True)

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

    #total_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    total_training_steps = len(train_dataloader) // 1 * args.num_train_epochs

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

    logger.info("***** Started training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    #logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size
        # * args.gradient_accumulation_steps
        # * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_training_steps)

    # TODO load checkpoints

    model.zero_grad() 
    train_iterator = trange(0,int(rgs.num_train_epochs),desc="Epoch")
    global_step = 0
    logging_loss, tr_loss = 0.0,0.0
    set_seed(args.seed)

    for _ in train_iterator:
        within_epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(within_epoch_iterator):
            inputs, labels = (batch, batch)

            print("At step: ", step)
            # Skip Long Examples
            if inputs.shape[1] > 1024: continue

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.train()
            outputs  = model(inputs,labels=labels)
            loss = outputs[0]

            loss.backward()

            tr_loss += loss.item()

            # Here we might use accumulation steps
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step+=1

            if global_step % args.logging_steps:
                tb_writer.add_scalar("lr", scheduler.get_lr()[0],global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss)/args.logging_steps)
                
                logging_loss = tr_loss
            if global_step % args.checkpoint_interval:
                output_dir = os.path.join(args.output_dir,"{}-{}".format("chkpnt",global_step))
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = model

                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir,"training_args.bin"))

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    tb_writer.close()
    return global_step, tr_loss /global_step



        #
def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, df_trn, df_val, prefix ="") -> Dict:
    # Create output dir
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate, drop_last=True)

    # Load the evaluation data set
    # Set teh evaluation size
    #
    # Do colation 
    # Crete Sampelr
    # Create Data Loader
    #
    # Log Some Stuff
    #
    # Set the model to evaluation mode
    #
    # Go through the batch 
    #
    # eval_loss /= evaluation_steps
    # prepelexity = torch.exp(torch.tensor(eval_loss))
    # result = {"perplexity": perplexity}

    # log_eval_file = os.path.join(e)

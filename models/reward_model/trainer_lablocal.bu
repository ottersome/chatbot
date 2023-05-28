import glob
import io
import json
import logging
import os
import pickle
import random
import sys
import re
import shutil
from typing import Dict, List, Tuple

import pandas as pd 
import numpy as np
import torch
import bitsandbytes as bnb
import matplotlib.pyplot as plt
import torch.nn.functional as F

from datasets import *
# from pytorch_memlab import LineProfiler, MemReporter
# from GPUtil import showUtilization as gpu_usage

from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad, logsigmoid
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import gc

# For Reporting to Telegram
import requests

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
from utils import set_seed,get_mask

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

def save_checkpoint(model, optimizer, args,tinfo):
    output_dir = os.path.join(args.output_dir,"{}-{}-{}".format("chkpnt",tinfo['epoch'],tinfo['global_step']))
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving model on the {}-th epoch and {}-th global step into {}".format(tinfo['epoch'],tinfo['global_step'], output_dir))

    print("Saving model on the {}-th epoch and {}-th global step into {}".format(tinfo['epoch'],tinfo['global_step'], output_dir))

    if isinstance(model,torch.nn.DataParallel):
        gval_head = model.module.gpt_w_valhead
    else:
        gval_head = model.gpt_w_valhead
    torch.save(model.module.gpt_w_valhead.state_dict(),os.path.join(output_dir,"model_state_dict.pt")) # For DataParallel
    # torch.save(model,os.path.join(output_dir,"model_state_dict.pt")) # Single Gpu
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(tinfo, os.path.join(output_dir, "tinfo.bin"))
    print("Done model on the {}-th epoch and {}-th global step into {}".format(tinfo['epoch'],tinfo['global_step'], output_dir))

def test_memory_overflow(batch, model, tokenizer, args):

    gcombo, bcombo, gtypeids, btypeids =  batch
    gmasks, bmasks = get_mask(gcombo,0,tokenizer.eos_token_id), get_mask(bcombo,0,tokenizer.eos_token_id)

    gcombo = gcombo.to(args.device)
    bcombo = bcombo.to(args.device)
    gmasks = gmasks.to(args.device)
    bmasks = bmasks.to(args.device)
    gtypeids = gtypeids.to(args.device)
    btypeids = btypeids.to(args.device)

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    model.train()
    model.zero_grad()

    logger.info("Testing memory Consumption with Largest batch")
    output = model(gcombo,bcombo,gmasks, bmasks, gtypeids,btypeids)
    print('Supposed peak memory use : '+get_mem_use(args))
    logger.info("Current Memory Consumption")
    loss = output.loss.mean()
    loss.backward()
    optimizer.step()

    del loss, gcombo, bcombo, gmasks, bmasks, gtypeids, btypeids
    logger.info("Cleared Memory Consumption")


def get_mem_use(args):
    string= ""
    for i in range(args.n_gpus):
        # string += "gpu{} is : {:.2f}%,".format(i, torch.cuda.memory_allocated('cuda:'+str(i))/torch.cuda.max_memory_allocated('cuda:{}'.format(i)))
        string += "max gpu{} is : {:.2f} GB,".format(i, torch.cuda.max_memory_allocated('cuda:{}'.format(i))/1e9 )
    return string

def get_mem_use_tinfo(args,tinfo,prefix):
    string= ""
    for i in range(args.n_gpus):
        amnt = torch.cuda.memory_allocated('cuda:'+str(i))/torch.cuda.max_memory_allocated('cuda:'+str(i))
        string += "gpu{} is : {:.2f}%,".format(i, amnt)
        #tinfo[prefix+'_mem_use_gpu'+str(i)].append(amnt)
        tinfo.setdefault(prefix+'_mem_use_gpu'+str(i),[]).append(amnt)
    return string

def write_to_tensorboard(summwritter: SummaryWriter,tinfo: Dict, epoch):
    for k,v in tinfo.items():
        if isinstance(v,list) and len(v)!= 0:
            summwritter.add_scalar(k,v[-1],epoch)

    
def send_plot(x,y,message_id):
    """
    Prepared data should be json which includes at least `chat_id` and `plot_file`
    """
    plt.plot(x,y)
    img = io.BytesIO()
    plt.title('Average Loss')
    plt.savefig(img,format='png')
    img.seek(0)

    # Update
    if message_id!='':
        url = 'https://api.telegram.org/bot'+os.environ['TTOKEN']+'/editMessageMedia'
        data = {'chat_id', int(os.environ['TUSER'])}
        message_url = 'https://api.telegram.org/bot'+os.environ['TTOKEN']+'/editMessageMedia'
        files = {'photo':('avg_loss.png',img.getvalue())}
        data = {
                "chat_id": int(os.environ['TUSER']),
                "message_id": message_id,
                "media": json.dumps({
                    "type": "photo",
                    "media": 'attach://photo',
                    "caption": 'Average Loss for Epoch '+str(len(x)),
                })
            }
    else:
        message_url = 'https://api.telegram.org/bot'+os.environ['TTOKEN']+'/sendPhoto'
        files = {'photo':('avg_loss.png',img.getvalue())}
        data = {'chat_id':int(os.environ['TUSER']),'caption':'Avg_loss'}

    response = requests.post(message_url,files=files, data=data)
    plt.clf()
    if response.status_code == 200:
        message_id = response.json()['result']['message_id']
    else:
        print('Error sending message:', response.text)

    return message_id


def train(args, dataset: BinaryFeedbackDataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int,float]:

    tel_message_id = ''

    def collate(examples: List[torch.Tensor]):
        logger.info("Size of examples here is :"+str(len(examples)))
        gcombos,bcombos = [],[]
        gtids,btids= [],[]
        for example in examples:
            gcombos.append(example['gcombo'])
            bcombos.append(example['bcombo'])
            gtids.append(example['good_type_ids'])
            btids.append(example['bad_type_ids'])


        if tokenizer._pad_token is None:
            # Will likey by this with GPTJ, so by default it will pad with 0
            g_padded = pad_sequence(gcombos, batch_first=True)
            b_padded = pad_sequence(bcombos, batch_first=True)
            gtids_padded = pad_sequence(gtids, batch_first=True)
            btids_padded = pad_sequence(btids, batch_first=True)
            return g_padded,b_padded, gtids_padded, btids_padded

        g_padded = pad_sequence(gcombos, batch_first=True, padding_value=tokenizer.pad_token_id)
        b_padded = pad_sequence(bcombos, batch_first=True, padding_value=tokenizer.pad_token_id)
        gtids_padded = pad_sequence(gtids, batch_first=True, padding_value=tokenizer.pad_token_id)
        btids_padded = pad_sequence(btids, batch_first=True, padding_value=tokenizer.pad_token_id)
        return g_padded,b_padded, gtids_padded, btids_padded
    
    # Choose Adam as optimizer
    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
           {
               "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
               "weight_decay": args.weight_decay,
           },
           {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
       ]

    # Choose Adam as optimizer
    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #  optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    tinfo = {"loss" : 0,"epoch": 1,"global_step" : 0 , "saved_step" : 0,"epoch_wise_loss" : [],"avg_loss":[], "epoch_wise_valloss" : []}
    # Add GPU Memory Usage
    # for i in range(args.n_gpus):
        # tinfo['mem_use_gpu'+str(i)] = []

    ########################################
    # Load Checkpoint
    ########################################
    #if args.checkpoint_path != "":
        # No loading of tinfo when using supervised checkpoint
    if False: 
        print("Starting with checkpoint: "+args.checkpoint_path)
        optimizer.load_state_dict(torch.load(args.checkpoint_path+'/optimizer.pt'),strict=False)
        # tinfo = torch.load(args.checkpoint_path+'/tinfo.bin')
        # tinfo['epoch'] += 1 # Add a 1. Because it remembers last epoch not next one

    # For main gpu 
    tb_writer = SummaryWriter()

    #Start Summary Writter
    # Pad Sequence
    batch_size = args.batch_size_per_gpu# Lets leave it at that for now 
    
    # Here each example is made of a good and bad combination

    # Sample from training dataset
    print(f"Size of training data set is {len(dataset)}")
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset,sampler=train_sampler, batch_size=batch_size, collate_fn=collate, drop_last=True)

    # Test for Memory Break
    model_to_test = model.module if isinstance(model, torch.nn.DataParallel)  else  model
    test_memory_overflow(dataset.get_longest_batch(batch_size), model, tokenizer,args)

    # TODO: Maybe do gradient accumulation again?
    total_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # scheduler = get_linear_schedule_with_warmup(
        # optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)

    logger.info("***** Started training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    #logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size_per_gpu
        # * args.gradient_accumulation_steps
        # * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_training_steps)


    print("Starting with epoch " ,tinfo['epoch'])
    model.zero_grad() 
    train_iterator = trange(tinfo['epoch'],int(args.num_train_epochs),desc="Epoch")
    tinfo['global_step'] = 0
    tr_loss = 0.0
    set_seed(args.seed)
    epoch_wise_valloss = tinfo['epoch_wise_valloss']

    logger.info("We will do savings per batch : {}".format(int(0.1*(dataset.len()/args.batch_size_per_gpu))))

    sent_plot = None
    for _ in train_iterator:
        within_epoch_iterator = tqdm(train_dataloader, initial=tinfo['saved_step'],desc="Iteration", leave=False)
        step = 0
        for batch in within_epoch_iterator:
            desc=""
            desc+= 'Epoch wise start memory use : '+get_mem_use(args)+'\n'
            get_mem_use_tinfo(args,tinfo, prefix="begin_loop")
            
            optimizer.zero_grad()
            gcombo, bcombo, gtypeids, btypeids =  batch

            gmasks, bmasks = get_mask(gcombo,0,tokenizer.eos_token_id), get_mask(bcombo,0,tokenizer.eos_token_id)

            gcombo = gcombo.to(args.device)
            bcombo = bcombo.to(args.device)
            gmasks = gmasks.to(args.device)
            bmasks = bmasks.to(args.device)
            gtypeids = gtypeids.to(args.device)
            btypeids = btypeids.to(args.device)

            model.train()

            logger.info("Length of gcombo: {} and bcombo {}".format(gcombo.shape, bcombo.shape) )

            output = model(gcombo,bcombo,gmasks, bmasks, gtypeids,btypeids)
            
            get_mem_use_tinfo(args,tinfo, prefix="aft_forw")
            desc+= 'After loss calculated: '+get_mem_use(args)+'\n'

            loss = output.loss
            gscore = output.logits[:,0].detach()
            bscore = output.logits[:,1].detach()

            # Our own Loss
            if args.n_gpus > 1:# For sake of clarity
                loss = loss.mean()
            
            # Calculate Gradients
            loss.backward()
            desc+= 'After loss backward use'+get_mem_use(args)+'\n'
            get_mem_use_tinfo(args,tinfo, prefix="aft_loss")

            gval_head = model.module.gpt_w_valhead
            if isinstance(model,torch.nn.DataParallel):
                gval_head = model.module.gpt_w_valhead
            transformer = gval_head.transformer
            # print(model)
            desc+="Scores are : g:{}\n\t and b:{}\n".format(gscore,bscore)
            desc+="Learning Rates is : {}\n".format(scheduler.get_lr())
            desc+="Transformer Block Weights Norm: {}\n".format(transformer.h[27].attn.q_proj.adapter[0].weight.abs().sum())
            desc+="Transformer Block Weights Grad: {}\n".format(transformer.h[27].attn.q_proj.adapter[0].weight.grad.abs().sum())
            desc+="Value  Weights Norm: {}\n".format(gval_head.val_head.weight.abs().sum())
            logger.info(desc)
            print(desc)
            #desc+="Value  Weights Norm: {}\n".format(model.module.val_head.weight.abs().sum())
            #loss = outputs[0]
            tinfo['epoch_wise_loss'].append(loss.item())

            tr_loss += loss.item()


            # Here we might use accumulation steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            desc+= 'After optimizer step: '+get_mem_use(args)+'\n'
            tinfo['global_step']+=1
            tinfo['avg_loss'].append(tr_loss/tinfo['global_step'])
            
            # Log If Weights are changing
            #  print("Embedding Weights:", model.transformer.wte.adapter[1].weight)

            logger.info(f"Loss for epoch {tinfo['epoch']}, global steo {tinfo['global_step']} is {tr_loss/tinfo['global_step']}")
            within_epoch_iterator.set_description('CSLoss: {} CBLoss: {}'.format(loss.item(),tr_loss/tinfo['global_step']))

            del gcombo,bcombo, gmasks, gtypeids,btypeids,bmasks, gscore,bscore, loss, batch
            torch.cuda.empty_cache()
            gc.collect()
            tinfo['saved_step'] +=1
            step += 1
            ### END OF BATCH ##
            ## Some Post-Processing of the Step Here:

            write_to_tensorboard(tb_writer,tinfo, step)
            # Send Data For Monitoring
            if tinfo['global_step'] % 2:
                tel_message_id = send_plot(
                        np.arange(len(tinfo['avg_loss'])),
                        np.array(tinfo['avg_loss']),
                        tel_message_id)



            # Save Checkpoint
            if tinfo['global_step'] % int(0.1*(dataset.len()/args.batch_size_per_gpu)) == 0:
                save_checkpoint(model, optimizer,args,tinfo)


        tinfo['epoch'] += 1
        scheduler.step()
        tinfo['saved_step'] =0
        ########################################
        # End of Epoch Maintenance
        ########################################
        if tinfo['epoch']  % args.checkpoint_interval:
            save_checkpoint(model, optimizer,args,tinfo)
        # Test
        dataset.change_mode(0)
        val_score = evaluate(args, model, tokenizer, dataset)
        tinfo['epoch_wise_valloss'].append(val_score)
        # TODO compare validation results to see if there is no longer any improvement
        logger.info(f"Validation loss(perplexity) for epoch {tinfo['epoch']} is {val_score}")
        if len(tinfo['epoch_wise_vallos']) > 3 :
            threshold_crossed  = (tinfo['epoch_wise_valloss'][-1]-tinfo['epoch_wise_valloss'][-2] > 0 ) \
                    and (tinfo['epoch_wise_valloss'][-2]-tinfo['epoch_wise_valloss'][-3] > 0 )
            if threshold_crossed:
                output_dir = os.path.join(args.output_dir,"{}-{}".format("early_stop",tinfo['global_step']))
                os.makedirs(output_dir,exist_ok=True)
                logger.info('We have detected an increase in validation loss in two consecutive epochs.')
                logger.info('We will now save this model and stop trianing ')

                save_checkpoint(model, optimizer,args, tinfo)
                # torch.save(tinfo['epoch_wise_valloss'],os.path.join(output_dir,'valloss.pt'))
                # torch.save(tinfo['epoch_wise_loss'],os.path.join(output_dir,'loss.pt'))
                # torch.save(optimizer.state_dict(),os.path.join(output_dir,'optimizer.pt'))
                print(f'We have hit early stopping at epoch {tinfo["epoch"]} saving and breaking now...')
                
                break
        dataset.change_mode(1)

    train_iterator.close()
    tb_writer.close()
    avg_loss = tr_loss /tinfo['global_step']
    print(f"Finished with training with global_step:{global_step} and average loss {avg_loss}")
    return global_step, avg_loss



def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix ="") -> Dict:

    # Create output dir
    #  eval_output_dir = args.output_dir
    #  os.makedirs(eval_output_dir, exist_ok=True)
    def collate(examples: List[torch.Tensor]):
        logger.info("Size of examples here is :"+str(len(examples)))
        gcombos,bcombos = [],[]
        gtids,btids= [],[]
        for example in examples:
            gcombos.append(example['gcombo'])
            bcombos.append(example['bcombo'])
            gtids.append(example['good_type_ids'])
            btids.append(example['bad_type_ids'])


        if tokenizer._pad_token is None:
            # Will likey by this with GPTJ, so by default it will pad with 0
            g_padded = pad_sequence(gcombos, batch_first=True)
            b_padded = pad_sequence(bcombos, batch_first=True)
            gtids_padded = pad_sequence(gtids, batch_first=True)
            btids_padded = pad_sequence(btids, batch_first=True)
            return g_padded,b_padded, gtids_padded, btids_padded

        g_padded = pad_sequence(gcombos, batch_first=True, padding_value=tokenizer.pad_token_id)
        b_padded = pad_sequence(bcombos, batch_first=True, padding_value=tokenizer.pad_token_id)
        gtids_padded = pad_sequence(gtids, batch_first=True, padding_value=tokenizer.pad_token_id)
        btids_padded = pad_sequence(btids, batch_first=True, padding_value=tokenizer.pad_token_id)
        return g_padded,b_padded, gtids_padded, btids_padded
    batch_size = args.batch_size_per_gpu#* max(1, args.n_gpu)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, collate_fn=collate, drop_last=True)

    # TODO set model for multi gpu environment
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(eval_dataloader):
        inputs, labels = (batch,batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    return perplexity

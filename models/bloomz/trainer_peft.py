import os 
import torch
import logging
from datetime import date
from datasets import *
from utils import *
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer)
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
from peft import (
        prepare_model_for_int8_training,
        LoraConfig,
        get_peft_model,
        TaskType)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        filename='training.log',
                        level=logging.DEBUG
                        )
    ########################################
    # Load Model
    ########################################
    print('Loading Model...')
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map="auto")
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir = args.cache_dir)

    print('Model Loaded.')
    tparams,allparams = trainable_params(model)
    print(
            f"Originally: Trainable params: {tparams} || all params: {allparams} || trainable%: {100 * tparams / allparams}"
    )

    print('Configuring Peft...')
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
            r=16,lora_alpha=32, target_modules=["query_key_value"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )

    model = get_peft_model(model,lora_config)
    print(
            f"After Peft: Trainable params: {tparams} || all params: {allparams} || trainable%: {100 * tparams / allparams}"
    )

    ########################################
    # Load Data
    ########################################
    print('Loading Data...')
    train_dataset = BotDataset(tokenizer,args,logger)

    ########################################
    # Training
    ########################################
    
    def collator(examples: List[Dict[str,torch.Tensor]]):
        batch = {}
        first = examples[0]
        # Collect Differently
        for k in first.keys():
            batch[k] = [e[k] for e in examples]
        # Now Just Pad it as we know how to 
        padder = 0.0
        if tokenizer._pad_token != None:
            paddier= tokenizer._pad_token
        for k,v in batch.items():
            batch[k] = pad_sequence(v, batch_first=True, padding_value=tokenizer.pad_token_id)

        return batch

    training_args = TrainingArguments(
            "temp",
            evaluation_strategy="epoch",
            learning_rate=1e-5,
            gradient_accumulation_steps=1,
            auto_find_batch_size=True,
            num_train_epochs=1,
            warmup_steps=100,
            logging_dir="runs",
            logging_steps=5,
            save_strategy="steps",
            #report_to='tensorboard',
            save_steps=500,
            save_total_limit=2
            )

    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset)
    model.config.use_cache = False

    trainer.train()
    print('Training Finished')

    # TODO Evaluate the model
    # today = date.today()
    # datefmt="%m-%d-%Y_%H-%M-%S"
    # datestr = today.strftime(datefmt)
    # checkpoint_name = f"{args.model_name_or_path}_{datestr}.pt"


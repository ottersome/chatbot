import os 
import torch
import logging
from datetime import date
from datasets import *
from utils import *
from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer)
from peft import (
        prepare_model_for_int8_training,
        LoraConfig,
        get_peft_model,
        TaskType)

if __name__ == '__main__':

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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    print('Model Loaded.')
    tparams,allparams = trainable_params(model)
    print(
            f"Originally: Trainable params: {tparams} || all params: {allparams} || trainable%: {100 * tparams / allparams}"
    )

    print('Configuring Peft...')
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
            r=16,lora_alpha=32, target_modules=["q","v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
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
    
    training_args = TrainingArguments(
            "temp",
            evaluation_strategy="epoch",
            learning_rate=1e-3,
            gradient_accumulation_steps=1,
            auto_find_batch_size=True,
            num_train_epochs=1,
            output_dir="output",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=8
            )

    trainer = Trainer(
        model=model,
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


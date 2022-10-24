from transformers import ElectraModel, ElectraConfig, ElectraTokenizerFast, ElectraForPretraining
from AR_ElectraDialogModel import *

c = MyConfig({
    'device':'cuda:0',
    'start':0,
    'end': 10,

    'pretrained_checkpoint': 'vanilla_11081_100.0%pth',
    'seeds':None,

    'weight_decay': 0,
    'adam_bias_correction': False,
    'xavier_reinited_outlayer': True,
    'schedule': 'original_linear',
    'original_lr_layer_decays': True,
    'double_unordered': True,
    
    # whether to do finetune or test
    'do_finetune': True, # True -> do finetune ; False -> do test
    # finetuning checkpoint for testing. These will become "ckp_dir/{task}_{group_name}_{th_run}.pth"
    'th_run': { 'qqp': 7, 'qnli': 5,
               'mrpc': 7, 'mnli': 2, 'ax': 2,
               'sst2': 3, 'rte': 7,  'wnli': 0, 
               'cola': 1, 'stsb': 8,  
               },

    'size': 'small',
    'wsc_trick': False,

    'num_workers': 3,
    'my_model': False, # True only for my personal research
    'logger': 'wandb',
    'group_name': None, # the name of represents these runs
    # None: use name of checkpoint.

    # Fine Tuning
    'hidden_size': 64

    })

def get_model():
    discriminator = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
    return discriminator,tokenizer

def get_corpus(location,interaction_maxlen):
    print("Loading Corpus at ",location)
    pairs = []
    with open(location, encoding="utf-8") as f:
        it = iter(f)
        for prompt in it:
            # Check if the size is right
            prep_prompt = "[CLS ]"+prompt+" [SEP]",
            ans = it.next()
            if len(prep_prompt+ans) <= interaction_maxlen:
                continue
            pairs.append([prep_prompt,ans])
        
    # Create Pairs
    print("AMount of Setences")
    return pairs

if __name__ == "__main__":
    # Load the Base Model up
    discriminator,tokenizer = get_model()

    # Feed Model to our Wrapper
    AR_DialogModel = AutoRegressiveDialogModel(discriminator)

    # Load Corpus
    corpus





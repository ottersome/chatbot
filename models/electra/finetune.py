from transformers import ElectraModel, ElectraConfig, ElectraTokenizerFast, ElectraForPreTraining
from AR_ElectraDialogModel import *
import argparse
import yaml

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

def yaml_config(config_yaml : str) -> {}:
    with open(config_yaml) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def cmd_config(parser: argparse.ArgumentParser()) -> {}:
    parser.add_argument("--hiddensize",
            help="Hidden Vector Size",
            dest="hiddensize",
            default=64,
            type=int,
            )
    return vars(parser)


if __name__ == "__main__":
    # Load the Base Model up
    discriminator, tokenizer = get_model()

    # Parse and get Config
    parser = argparse.ArgumentParser("Electra for AutoRegressive Dialog")
    config = {'yaml_path':'./configs/finetune.yaml'}
    config.update(yaml_config(config['yaml_path']))
    config.update(cmd_config(parser))
    print(config)
    exit

    # Feed Model to our Wrapper
    AR_DialogModel = AutoRegressiveDialogModel(discriminator)

    # Load Corpus
    corpus = get_corpus()

    # Fine Tune
    


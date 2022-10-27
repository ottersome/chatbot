from transformers import ElectraForPreTraining, ElectraTokenizerFast
import argparse
import configs
import torch


#  print(torch.cuda.is_available())
#  dev = "cuda:0"
#  print(torch.zeros(1).cuda())
#  exit()

def load_base_components():
    #  device = torch.device
    discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
    #  discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator",device=device)
    # Inheritance goes: Electra <- Bert <- Wordpiece
    # So basically wordpiece
    #  tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

    return discriminator, tokenizer

def test_discriminator(discriminator,tokenizer, sentence_pairs):

    #  sentence = sentence_pairs['true']
    fake_sentence = sentence_pairs['false']

    #Tokenize the Sentences
    #  fake_tokens = tokenizer.tokenize(fake_sentence)
    #  fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
    fake_inputs = tokenizer(fake_sentence, return_tensors="pt")# Gives you ids
    decoded_input = tokenizer.decode(fake_inputs['input_ids'][0])
    #  fake_inputs.to(device)
    print("Our Vocab-Tokenized Sentences is :\n\t",decoded_input)
    discriminator_outputs = discriminator(fake_inputs['input_ids'])
    print("Debugging Station")

    predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

    #  [print("%7s" % token, end="") for token in decoded_input]

    [print("%7s" % float(prediction), end="") for prediction in predictions.squeeze().tolist()]

if __name__ == "__main__":

    sentence_pairs = {
            "true":"The quick brown fox jumps over the lazy dog",
            "false":"The quick brown fox fake over the lazy dog"
            }
    discriminator, tokenizer = load_base_components()
    test_discriminator(discriminator,tokenizer,sentence_pairs)



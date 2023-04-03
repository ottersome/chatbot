import sys
import torch
from GPT_RM import *

from torch.nn.utils.rnn import pad_sequence
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    pipeline,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
#pipe = pipeline(model='EleutherAI/gpt-j-6B',model_kwargs={'device_map':"auto","load_in_8_bits":True})
name = 'hivemind/gpt-j-6B-8bit'
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')

if len(sys.argv) > 1: 
    #  model = AutoModelWithLMHead.from_pretrained('output/dialoggpt-medium-epoch-20')
    model = GPTJForCausalLMWithValueHead.from_pretrained(name, low_cpu_mem_usage=True)
    add_adapters(model)
    # model.load_state_dict(torch.load(sys.argv[1]+'/model_state_dict.pt'))
    model=torch.load(sys.argv[1]+'/model_state_dict.pt')
    device = torch.device('cuda')
    model.to(device)
else:
    print("Using online model")

    print('Setting Up Tokenizers and (Possibly) PreTrained Models')
    config = AutoConfig.from_pretrained(name, cache_dir='./.my_cache')
    model = AutoModelWithLMHead.from_pretrained(
            name, 
            from_tf=False,
            config=config,
            cache_dir='./.my_cache/')


print("All good hre , we are about to start")
# Let's chat for 5 lines

# This one will be a bit different as we are only using this for debugging and  not displaying it in some random endpoint.
cur_length = 0
# Get The Prompt First. 
print("Prompt:")
prompt=""
while('ENDO' not in prompt):
    prompt += input("") 
app_idx = prompt.find('ENDO')
prompt = prompt[:app_idx] 
prompt.replace('<EOS>', tokenizer.eos_token),

# Get the Mask
tkd_prompt = tokenizer.encode(prompt + tokenizer.eos_token)
print("Shape of tokenized prompt: {}".format(len(tkd_prompt)))
mask_0s = torch.zeros(len(tkd_prompt))# Lets keep the shape simple for now
# Set the Model to not store gradietns
model.eval()
# We will be getting multiple reponses and returning their 
while True:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_response =""
    while('ENDO' not in new_response):
        new_response += input("") 
    app_idx = new_response.find('ENDO')
    new_response = new_response[:app_idx] 
    tkd_response = tokenizer.encode(new_response + tokenizer.eos_token)
    # Similarly make a mask of 1s
    mask_1s = torch.ones(len(tkd_response))

    # We will be doing single element batches so I wont care about padding:
    # print("Response was : "+new_response)
    tkd_interaction = torch.tensor(tkd_prompt+ tkd_response).type(torch.long).view(1,-1).to(device)
    interaction_partition_mask = torch.cat([mask_0s,mask_1s], dim=-1 ).type(torch.int).view(1,-1).to(device)
    # print("Tokened Interaction {}".format(tkd_interaction))
    # print("Interaction partition mask: {}".format(interaction_partition_mask))

    # This is ready for the model
    score = model(input_ids=tkd_interaction, token_type_ids=interaction_partition_mask)
    print("Score for your reponse is : ", score.logits.item())

    # append the new user input tokens to the chat history
    # bot_input_ids = torch.cat([nth_output, new_user_input_ids['input_ids']], dim=-1) if cur_length > 0 else new_user_input_ids['input_ids']
    #cur_length = bot_input_ids.shape[-1]

    # generated a response while limiting the total chat history to 1000 tokens, 
    # nth_output = model.generate(input_ids=bot_input_ids,max_length=2048, do_sample=True)
    
    # pretty print last ouput tokens from bot
    # print("botinput ids: ", bot_input_ids)
    # print("Nth output:\n\t", nth_output)

    del new_response



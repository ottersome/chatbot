import sys
import torch
import logging
from datetime import datetime

from torch.nn.utils.rnn import pad_sequence
from transformers import utils
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
from peft import (
        prepare_model_for_int8_training,
        LoraConfig,
        PeftConfig,
        get_peft_model,
        TaskType)

name = 'bigscience/bloomz-7b1'
#pipe = pipeline(model='EleutherAI/gpt-j-6B',model_kwargs={'device_map':"auto","load_in_8_bits":True})
#tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")

config = AutoConfig.from_pretrained('bigscience/bloomz-7b1')
#model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-7b1", device_map="auto")

#model = AutoModelForCausalLM.from_pretrained(name,config=config,path=name,load_in_8_bits=True,device_map="auto",local_files_only=True)
if len(sys.argv) > 1:
    print('We assume your argument {} means checkpoint'.format(sys.argv[1]))
    chkpnt = sys.argv[1]
    model = AutoModelForCausalLM.from_pretrained(name,device_map="auto",load_in_8bit=True)
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(inference_mode=True,
            r=16,lora_alpha=32, target_modules=["query_key_value"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )
    model = get_peft_model(model,lora_config)
    model.state_dict = torch.load(chkpnt+'/pytorch_model.bin')

model.eval()
cur_length = 0
# print('Conversation Starts:')
with torch.no_grad():
    while True:
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        user_input = input("User: ").strip()
        #user_input = "Who was the president of Mexico in 2014?"
        print('You said:\n',user_input)
        new_user_input_ids = tokenizer.encode(user_input , return_tensors='pt').to('cuda')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([nth_output.to('cuda'), new_user_input_ids], dim=-1).to('cuda') if cur_length > 0 else new_user_input_ids
        cur_length = bot_input_ids.shape[-1]

        # generated a response while limiting the total chat history to 1000 tokens, 
        #nth_output = model.generate(input_ids=bot_input_ids, do_sample=True)
        nth_output = model.generate(input_ids=bot_input_ids,max_length=2048, do_sample=True, top_k=0, temperature=0.7)
        
        decoded_output = tokenizer.decode(nth_output[0][cur_length:]).replace(tokenizer.eos_token,'')

        print("Bot: {}\n".format(decoded_output))
        sys.stdout.flush()



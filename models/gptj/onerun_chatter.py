import sys
import torch

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
name = 'EleutherAI/gpt-j-6B'
tokenizer = AutoTokenizer.from_pretrained(name)

if len(sys.argv) > 1: 
    print("Using online model")
    print(sys.argv[1])

    print('Setting Up Tokenizers and (Possibly) PreTrained Models')
    config = AutoConfig.from_pretrained(sys.argv[1], cache_dir='./.my_cache')
    model = AutoModelWithLMHead.from_pretrained(
            sys.argv[1], 
            from_tf=False,
            config=config,
            cache_dir='./.my_cache/')
else:
    #  model = AutoModelWithLMHead.from_pretrained('output/dialoggpt-medium-epoch-20')
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto",load_in_8bit=True)

params = {
  "layers": 28,
  "d_model": 4096,
  "n_heads": 16,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64,

  "seq": 2048,
  "cores_per_replica": 8,
  "per_replica_batch": 1,
}

device = 'cpu'
# Let's chat for 5 lines
while True:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    tokens = tokenizer(input("User: "), return_tensors='pt' )
    prompt = {key:value.to(device) for key,value in tokens.items()}

    # generated a response while limiting the total chat history to 1000 tokens, 
    out = model.generate(**prompt,min_length=10, max_length=128,do_sample=True
        #pad_token_id=tokenizer.eos_token_id,
        #top_p=0.92,
        #top_k = 30
        #  top_k = 50
    )
    
    # pretty print last ouput tokens from bot
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    sys.stdout.flush()



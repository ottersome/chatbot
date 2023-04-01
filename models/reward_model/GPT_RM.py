import transformers

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Mapping, Any

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from transformers.modeling_outputs import CausalLMOutputWithPast

from tqdm.auto import tqdm

class EloLoss(nn.Module):
    def __init__(self):
        super(EloLoss, self).__init__()
    def forward(self, output, target):
        pass
        # We wil llikely receive batches here


class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias
 
    def forward(self, input):
        # Apply the Old Weigths
        output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias)
        # Sum the product with new weights:
        if self.adapter:
            output += self.adapter(input)
        return output
 
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
 
 
class DequantizeAndLinear(torch.autograd.Function): 
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias).clone()
 
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias
 
 
# Remeber, only a single layer of these in the entire stack. 
class FrozenBNBEmbedding(torch.nn.modules.sparse.Embedding):
    def __init__(self,num_embeddings, embedding_dim, weight, absmax, code):
        super().__init__(num_embeddings, embedding_dim)
        delattr(self,'weight')
        print('We are creating our Embedding from init')
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
 
    def forward(self, inputs, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(inputs, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(inputs)
        return output 
 
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        print('We are not using this function because we are downloading a model\
              thats already quantized')
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"
 
 
def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        # Chunk Out ourinput matrix:
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)
 
    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)
 
 
def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name != 'val_head':# Lm_head is replaced here so its not needed to be that way
                # Weights will be loaded later
                setattr(
                    module,
                    name,#Name refers to the specific module, say val_head
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                        FrozenBNBEmbedding(
                            child.num_embeddings, child.embedding_dim,
                            weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                            absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                            code=torch.zeros(256),
                        )
                    )

# I think the model we use for pretraining has already ran this 
# on the original GPT-J 6B
class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


# Doesnt Seem to be getting used
# -> THis is because this boy is for MOdels without the Linear Matrix Head
# -> (The one that does class classification)
class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


        
class GPTJForCausalLMWithValueHead(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.embedding_dim = config.n_embd
        self.val_head = nn.Linear(config.n_embd, 1)
        self.val_head.weight.data.normal_(mean=0.0,std=0.2)
        self.val_head.bias.data.zero_()
        self.dropout = nn.Dropout(0.1)
        convert_to_int8(self)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # This might be every single hiddden vector the transformer_output
        # outputs for every word in the given sentence.
        # information 
        hidden_states = transformer_outputs[0]

        # Get Each Index of Last EOS
        # TODO soft code the EOS Token
        if False:
            hits = (input_ids == 50256).nonzero(as_tuple=True)
            idxs = []
            offset = 0
            tensors = torch.zeros((hidden_states.shape[0],1,hidden_states.shape[2]))
            for i in range(input_ids.shape[0]):
                count_amnt = (hits[0]==i).sum()# Shoudl Give us an Index
                idxs.append(hits[1][offset+count_amnt-1])
                #idxs.append(offset+count_amnt-1)
                offset += count_amnt
                tensors[i,0,:] = hidden_states[i,idxs[-1],:]
                #  print("Shape of input_ids:",input_ids.shape[1]," and of hidden:",hidden_states.shape[1],
                #        "our idx is:",idxs[-1])
                #  print("We have selected token corresponding to: ",input_ids[i,idxs[-1]])
            # Set device for model parallelism
            last_hstates= torch.tensor(tensors).to(torch.float32).to(self.val_head.weight.device)
        else:
            last_hstates = hidden_states[:,-1,:]

        # Do some drop out
        dropped_hstates = self.dropout(last_hstates)


        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.val_head.weight.device)

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        #lm_logits = torch.flatten(self.val_head(dropped_hstates).to(torch.float32))
        lm_logits = self.val_head(dropped_hstates)
        # This gives memy scalar 

        loss = None
        assert labels == None, print('We are not yet working with labels')

        if not return_dict:     
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        # Quickly remove head 
        delattr(self.val_head)
        super().load_state_dict(state_dict, strict)
        setattr(self, 'val_head', nn.Linear(self.embedding_dim, 1))
        self.val_head = nn.Linear(config.n_embd, 1)
        self.val_head.weight.data.normal_(mean=0.0,std=0.2)
        self.val_head.bias.data.zero_()

# class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):

    # def __init__(self, config):
        # super().__init__(config)
        # self.lm_head = nn.Linear(config.n_embd, 1,bias=False)
        # convert_to_int8(self)
        # #self.lm_head = nn.Linear(self.lm_head.in_features, 1,bias=False)
        
        # # Replace Category logits to scalar output
        # print("Last Module is :")
        # print(self.lm_head)
        # print("With In features : ")
        # print(self.lm_head.in_features)
        # # We will train this as the Scalar Ranker
        # print('Done with this')

def add_adapters(model, adapter_dim=16):
    assert adapter_dim > 0

    # After we have converted to 8 bits we then fill in extra layers for the Frozen Layers
    for module in model.modules():
        if isinstance(module, FrozenBNBLinear):
            module.adapter = nn.Sequential(
                # A and B matrices
                nn.Linear(module.in_features, adapter_dim, bias=False), 
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)
        elif isinstance(module, FrozenBNBEmbedding):
            module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)




transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J

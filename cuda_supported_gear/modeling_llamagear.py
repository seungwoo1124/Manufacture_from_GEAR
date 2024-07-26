import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from quant.new_pack import triton_quantize_and_pack_along_last_dim,triton_quantize_and_pack_along_last_dim_witherror, headwise_lrap
from quant.matmul import cuda_bmm_fA_qB_outer

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

_CONFIG_FOR_DOC = "LlamaConfig"
def if_not_then_None(variable):
    try:
        variable
    except NameError:
        variable = None
    return variable
def key_compression(key_full,compress_config):
    bsz, num_head, seq_len, head_dim = key_full.shape
    if "gearl" in compress_config["compress_method"] or "gearsl" in compress_config["compress_method"]:
        key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new, error = triton_quantize_and_pack_along_last_dim_witherror(key_full, 
                                                                                                                            compress_config["group_size"], 
                                                                                                                            compress_config["quantize_bit"])
        error = error.reshape(bsz, num_head, head_dim, seq_len).transpose(2,3)

        key_states_p,key_states_q = headwise_lrap(error,compress_config["rank"],compress_config["loop"])
    else:
        key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_full, 
                                                                                                                            compress_config["group_size"], 
                                                                                                                            compress_config["quantize_bit"])
        key_states_p,key_states_q = None, None
    return key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new, key_states_p,key_states_q

def value_compression(value_full,compress_config):
    bsz, num_head, seq_len, head_dim = value_full.shape
    if "gearl" in compress_config["compress_method"] or "gearsl" in compress_config["compress_method"]:
        value_states_quant_new, scale, mn, error = triton_quantize_and_pack_along_last_dim_witherror(value_full,
                                                                                                                            compress_config["group_size"], 
                                                                                                                            compress_config["quantize_bit"])
        error = error.reshape(bsz, num_head, seq_len, head_dim)

        value_states_p,value_states_q = headwise_lrap(error, compress_config["rankv"],compress_config["loop"])
    else:
        value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_full, 
                                                                                                                            compress_config["group_size"], 
                                                                                                                            compress_config["quantize_bit"])
        value_states_p,value_states_q = None, None
    return value_states_quant_new, scale, mn, value_states_p,value_states_q
def matmul_withlrap(group_size,a,b,scale,mn,bits,pbase,qbase, type = "key"):
    # print("qbase_shape",qbase.shape)
    # print("pbase_shape",pbase.shape)
    # print("a_size",a.shape)
    # print("b_size",b.shape)
    result1 = cuda_bmm_fA_qB_outer(group_size, a,b,scale,mn,bits)

    if pbase[0] == None:
        return result1
    if len(pbase) == 1:
        result2 = a @ qbase[0]
        result3 = result2 @ pbase[0].transpose(2,3)
        result1 = result1 + result3
    else:
        
        
        
        if type == "key":
            result2 = a @ qbase[0]
            result3 = result2 @ pbase[0].transpose(2,3)
            prefill_length = pbase[0].shape[-2]
            result1[:,:,:,:prefill_length] = result1[:,:,:,:prefill_length] + result3
            # result_1 shape [bsz, num_head, seq_len_q, seq_len_k]
            # result 2 shape [bsz, num_head, seq_len_q, seq_len_k_prefill]
            # pbase[1] shape [buffer_num, bsz ,num_head, seq_len_buffer, rank]
            # qbase[1] shape [buffer_num, bsz ,num_head,head_dim, rank ]
            result4 = a.unsqueeze(0) @ qbase[1] @ pbase[1].transpose(3,4)
            # result4 shape [buffer_num, bsz, num_head, head_dim, seq_len_buffer]
            # get dim and dim 4 together
            buffer_num, bsz, num_head, head_dim, seq_len_buffer = result4.shape
            result4 = result4.permute(1,2,3,0,4).reshape(bsz, num_head, head_dim, buffer_num*seq_len_buffer)
            result1[:,:,:,prefill_length:] = result1[:,:,:,prefill_length:] + result4
        else:
            prefill_length = qbase[0].shape[-2]
            buffer_length = qbase[1].shape[-2]
            real_length = result1.shape[-1]
            result2 = a[:,:,:,:prefill_length] @ qbase[0]
            result3 = result2 @ pbase[0].transpose(2,3)
            ### need to be specific here
            
            result1 = result1 + result3
            # pbase[1] shape [buffer_num, bsz ,num_head, headim, rank]
            # qbase[1] shape [buffer_num, bsz ,num_head, seq_len_buffer, rank]
            generated_a = a[:,:,:,prefill_length:]
            bsz, num_head, q_len, generate_length = generated_a.shape
            generated_a = generated_a.reshape(bsz, num_head, q_len,-1, buffer_length)
            generated_a = generated_a.permute(3,0,1,2,4)
            result4 = generated_a @ qbase[1] @ pbase[1].transpose(3,4)
            result4 = result4.sum(dim = 0).squeeze(0)

            # result4 shape [buffer_num, bsz, num_head, q, head_dim]
            # b bsz, num_head, q, head_dim = result4.shape


            result1 = result1 + result4


    return result1

class LlamaAttention_GEAR(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, layer_idx, config: LlamaConfig, compress_config=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.compress_config = compress_config
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_size = config.group_size
        self.residual_length = compress_config["residual"]

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[8]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len, position_ids=position_ids) # add position_ids arguments
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        assert self.num_key_value_groups == 1
        # [bsz, nh, t, hd]
        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]
            ### New added
            key_states_p = past_key_value[9]
            key_states_q = past_key_value[10]
            key_states_index = past_key_value[11]
            key_states_value = past_key_value[12]
            value_states_p = past_key_value[13]
            value_states_q = past_key_value[14]
            value_states_index = past_key_value[15]
            value_states_value = past_key_value[16]
            #### Notion
            # key_states_p batch,num_head,seqlen/buffer,rank
            # key_states_q batch,num_head,head_dim,rank
            # value_p batch,num_head,head_dim,rank
            # value_q batch,num_head,seqlen/buffer,rank
            # if len(key_states_p) > 1:
            #     print("key_p_shape",key_states_p[1].shape)
            #     print("key_q_shape",key_states_q[1].shape)
            #     print("value_p",value_states_p[1].shape)
            #     print("value_q",value_states_q[1].shape)
            if key_states_quant_trans is not None:
                # att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans, 
                #                 key_scale_trans, key_mn_trans, self.k_bits)

                att_qkquant = matmul_withlrap(self.compress_config["group_size"],
                                              query_states,
                                              key_states_quant_trans,
                                              key_scale_trans,
                                              key_mn_trans,
                                              self.compress_config["quantize_bit"],
                                              key_states_p,
                                              key_states_q,
                                              type = "key")
 
                
            else:
                #### it seeems that we will not be here
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new, key_states_p_new,key_states_q_new = key_compression(
                    key_states_full.transpose(2, 3).contiguous(), self.compress_config
                )
                
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                    if key_states_p_new is not None:
                        if len(key_states_p) == 1:
                            key_states_p_new = key_states_p_new.unsqueeze(0)
                            key_states_q_new = key_states_q_new.unsqueeze(0)
                            key_states_p.append(key_states_p_new)
                            key_states_q.append(key_states_q_new)
                        else:
                            key_states_p_new = key_states_p_new.unsqueeze(0)
                            key_states_q_new = key_states_q_new.unsqueeze(0)
                            key_states_p[1] = torch.concat([key_states_p[1],key_states_p_new],dim = 0)
                            key_states_q[1] = torch.concat([key_states_q[1],key_states_q_new],dim = 0)

                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new
                    key_states_p = [key_states_p_new]
                    key_states_q = [key_states_q_new]
            

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if value_states_full is not None:
                value_states_full = torch.cat([value_states_full, value_states], dim=2)
            else:
                value_states_full = value_states
            # value_full_length = if value_full_length == None 0 else value_states_full.shape[-2]
            if value_states_full == None:
                value_full_length = 0
            else:
                value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, value_states_full)
            else:
                # attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
                #                                 value_scale, value_mn, self.v_bits)
                # prindsdt("layerid  ",self.layer_idx,"attn_weights",attn_weights[:, :, :, :-value_full_length].shape,"value_states_quant",value_states_quant.shape,"value_P",value_states_p.shape,"value_Q",value_states_q.shape)
                attn_output = matmul_withlrap(self.compress_config["group_size"],attn_weights[:, :, :, :-value_full_length],value_states_quant,
                                            value_scale,
                                            value_mn, self.compress_config["quantize_bit"],value_states_p,value_states_q,
                                            type = "value")
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full)
            
            if value_full_length == self.residual_length:
                # print("here?")
                # print(value_full_length)
                # assert value_full_length == self.residual_length + 1
                # value_states_quant_new, scale, mn, value_states_p,value_states_q = value_compression(
                #     value_states_full[:, :, :1, :].contiguous(),
                #     self.compress_config
                # )

                

                # value_states_full = value_states_full[:, :, 1:, :].contiguous()
                # if value_states_quant is not None:
                #     value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                #     value_scale = torch.cat([value_scale, scale], dim=2)
                #     value_mn = torch.cat([value_mn, mn], dim=2)
                # else:
                #     value_states_quant = value_states_quant_new
                #     value_scale = scale
                #     value_mn = mn
                value_states_quant_new, scale, mn, value_states_p_new,value_states_q_new = value_compression(
                    value_states_full.contiguous(),
                    self.compress_config
                )
                

                

                value_states_full = None
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                    if value_states_p_new is not None:
                        if len(value_states_p) == 1:
                            value_states_p_new = value_states_p_new.unsqueeze(0)
                            value_states_q_new = value_states_q_new.unsqueeze(0)
                            value_states_p.append(value_states_p_new)
                            value_states_q.append(value_states_q_new)
                        else:
                            value_states_p_new = value_states_p_new.unsqueeze(0)
                            value_states_q_new = value_states_q_new.unsqueeze(0)
                            value_states_p[1] = torch.concat([value_states_p[1],value_states_p_new],dim = 0)
                            value_states_q[1] = torch.concat([value_states_q[1],value_states_q_new],dim = 0)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn
                    value_states_p = [value_states_p_new]
                    value_states_q = [value_states_q_new]

        else:
            attn_weights = torch.matmul(query_states, 
                                        key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                #### initial quantization
                key_states_quant_trans, key_scale_trans, key_mn_trans,key_states_p, key_states_q = key_compression(
                    key_states_quant.transpose(2, 3).contiguous(),
                    self.compress_config
                )
                key_states_p = [key_states_p]
                key_states_q = [key_states_q]
                
               
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
                key_states_p, key_states_q = None, None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
                value_states_p, value_states_q = None, None
            else:
                #### initial quantization
                residual = value_states.shape[-2] % self.residual_length
                quant_legnth = value_states.shape[-2] - residual
                value_states_quant = value_states[:, :, :quant_legnth, :].contiguous()
                value_states_full = value_states[:, :, quant_legnth:, :].contiguous()
                
                value_states_quant, value_scale, value_mn, value_states_p, value_states_q = value_compression(
                    value_states_quant,
                    self.compress_config
                )
                value_states_p =[value_states_p]
                value_states_q = [value_states_q]
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states) 
        
        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len,
                          key_states_p,
                          key_states_q,
                          None,
                          None,
                          value_states_p,
                          value_states_q,
                          None,
                          None) if use_cache else None
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value
    

class LlamaDecoderLayer_GEAR(nn.Module):
    def __init__(self,layer_idx, config: LlamaConfig, compress_config=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention_GEAR(layer_idx,config=config, compress_config=compress_config)
            # if not getattr(config, "_flash_attn_2_enabled", False)
            # else LlamaFlashAttention2(config=config)
        )
        self.layer_idx = layer_idx
        self.compress_config = compress_config
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModel_GEAR(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, compress_config=None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer_GEAR(_,config,compress_config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.compress_config = compress_config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][8]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_GEARKIVI(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, compress_config=None):
        super().__init__(config)
        self.model = LlamaModel_GEAR(config,compress_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.compress_config = compress_config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print(f"last hidden state shape : {outputs.last_hidden_state.shape}")
        # print(f"hidden states shape : {outputs.hidden_states.shape}")
        # print(f"attentions shape : {outputs.attentions.shape}")

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            
            # debug
            # for layer_idx in range(1):
            #     (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len,
            #               key_states_p, 
            #               key_states_q, 
            #               _,
            #               _,
            #               value_states_p,
            #               value_states_q,
            #               _,
            #               _) = past_key_values[layer_idx]
            #     print(f"===== LAYER {layer_idx} =====")
            #     print(f"key_states_quant_trans shape : {key_states_quant_trans.shape}")
            #     print(f"key_states_full shape : {key_states_full.shape if key_states_full is not None else None}")
            #     print(f"key_scale_trans shape : {key_scale_trans.shape}")
            #     print(f"key_mn_trans shape : {key_mn_trans.shape}")
            #     print(f"value_states_quant shape : {value_states_quant.shape}")
            #     print(f"value_states_full shape : {value_states_full.shape if value_states_full is not None else None}")
            #     print(f"value_scale shape : {value_scale.shape}")
            #     print(f"value_mn shape : {value_mn.shape}")
            #     print(f"kv_seq_len : {kv_seq_len}")
            #     print(f"key_states_p shape : {key_states_p[0].shape}")
            #     print(f"key_states_q shape : {key_states_q[0].shape}")
            #     print(f"value_states_p shape : {value_states_p[0].shape}")
            #     print(f"value_states_q shape : {value_states_q[0].shape}")
            # end debug

            past_length = past_key_values[0][8]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
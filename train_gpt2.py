"""
Reference code for GPT-2 training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Example launches to only benchmark the speed of bfloat16 compiled GPU training:
1 GPU:
python train_gpt2.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 --dtype=bfloat16
you can also turn on flash-attention by appending --flash=1
4 GPU:
torchrun --standalone --nproc_per_node=4 train_gpt2.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 --dtype=bfloat16
"""

import os
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model
####GPT-2原始仓库的 gelu实现 https://github.com/openai/gpt-2/blob/master/src/model.py
class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        ## batch维度的  qkv 所有的head 都对应一个 qkv 的矩阵，维度变化
        # (B, T, C) -> (B, T, 3*C) 
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #做一次线性投影
        #(B, T, C) -> (B, T, C) 
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        # 注意力头的数量
        self.n_head = config.n_head
        # 编码的词向量维度
        self.n_embd = config.n_embd
        #生成一个mask矩阵，用于attention 左下角为1 右上角为0
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)  ## (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)## qkv拆分tensor
        ### 将C维度拆分，变成n_head个head，View形状为(B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if FLASH:
            #直接调用pytorch的flashattention
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            # q和k 做 attention  得到 （B, nh, T, T) attention矩阵 做了 K size的归一化
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            #这个Python代码的作用是使用masked_fill函数将att张量中满足self.bias[:,:,:T,:T] == 0条件的元素填充为负无穷大（float('-inf')）。
            # 具体来说，它通过将self.bias张量的前T个元素与att张量进行比较，将满足条件的位置置为负无穷大，从而实现对att张量的修改。
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # 计算softmax 
            att = F.softmax(att, dim=-1)
            #将 attention 矩阵和v矩阵相乘
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # (B, nh, T, hs) -> (B, T, nh,hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection 
        #(B,T,C) ->(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """
    MLP forward B,T,C -> B,T,4C  ->act -> B,T,4C -> B,T,C
    线性层-> gelu -> 线性层
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """
    Layernorm -> Attention -> LayerNorm -> MLP
    注意 中间的 残差链接
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model
# ----------------------------------------------------------------------------- 
# gpt2的配置表
#该代码定义了一个名为GPTConfig的类，用于配置GPT模型的超参数。类中有五个属性，分别是：
# block_size：一个整数，表示每个输入序列的长度，默认值为1024。
# vocab_size：一个整数，表示词汇表的大小，默认值为50257。
# n_layer：一个整数，表示模型的层数，默认值为12。
# n_head：一个整数，表示多头注意力的头数，默认值为12。
# n_embd：一个整数，表示词嵌入的维度，默认值为768。
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            ## 词嵌入 词向量编码
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            ## 位置嵌入 位置变量编码 
            wpe = nn.Embedding(config.block_size, config.n_embd),
            ## transformer的堆叠block 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ##最后一个层归一化
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        #  最后一个 线性层，用于分类用
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        # 词向量和 最后一个head 线性层的权重共享
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        ## 随机数种子
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        # 权重初始化
        self.apply(self._init_weights)

    # 该函数用于初始化神经网络模型的权重。具体来说，它会对模型的不同组件进行不同方式的权重初始化。
    # 如果组件是nn.Linear类型的：
    # 根据GPT-2论文中的特殊缩放初始化方法，设定权重的标准差(std)为0.02。如果组件具有'LLMC_RESIDUAL_SCALE_FLAG'属性，则标准差会除以2 * self.config.n_layer的平方根。
    # 如果组件不具有'LLMC_SKIP_INIT'属性，使用均值为0.0，标准差为计算得到的std的高斯分布对权重进行初始化。
    # 如果该组件有偏置项，则将偏置项初始化为零向量。
    # 如果组件是nn.Embedding类型的：
    # 使用均值为0.0，标准差为0.02的高斯分布对权重进行初始化。
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None, return_logits=True):
        device = idx.device
        b, t = idx.size()# batch size, sequence length
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # 生成位置变量
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        # 词嵌入
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # 位置嵌入
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # 词嵌入和位置嵌入相加，融合
        x = tok_emb + pos_emb

        ## 多层block 堆叠 抽取语义
        for block in self.transformer.h:
            x = block(x)
        ## 最后一个层归一化
        x = self.transformer.ln_f(x)

        if targets is not None:## training 计算loss
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            ## 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            ## 该函数的功能是从给定的输入张量 x 中获取最后一个时间步的输出，并将其作为 logits 返回。
            # 具体来说，x[:, [-1], :] 表示选取 x 中所有行的最后一个时间步（由 [-1] 指定），并保持时间维度不变。
            # 然后，self.lm_head 被用来对选取的部分进行处理，并返回结果作为 logits。
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        从Hugging Face加载预训练的GPT-2模型权重。
        
        参数:
        model_type (str): 模型类型，支持'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'。
        
        返回:
        model: 初始化的GPT模型。
        """
        # 确保model_type是支持的类型之一
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # 导入GPT2LMHeadModel类
        from transformers import GPT2LMHeadModel
        # 打印加载的模型类型
        print("loading weights from pretrained gpt: %s" % model_type)

        # 根据模型类型设置配置参数
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M参数
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M参数
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M参数
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M参数
        }[model_type]
        # GPT模型词汇表大小固定为50257
        config_args['vocab_size'] = 50257 
        # GPT模型的块大小固定为1024
        config_args['block_size'] = 1024 
        # 创建一个从头开始初始化的minGPT模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        # 获取模型的状态字典
        sd = model.state_dict()
        sd_keys = sd.keys()
        # 过滤掉不需要的参数
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 初始化一个Hugging Face的GPT2LMHeadModel模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 确保两个模型的参数名称和形状都对齐
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        # 需要转置的权重参数
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 确保两个模型的参数数量匹配
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # 复制参数
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 直接复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # 返回加载了预训练权重的模型
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        """
        配置优化器参数。

        根据参数属性，如是否需要梯度、参数维度等，将模型参数分组并配置相应的优化器。
        
        参数:
            weight_decay (float): 权重衰减（L2正则化）系数。
            learning_rate (float): 学习率。
            betas (tuple): Adam优化器的beta参数。
            device_type (str): 设备类型，如'cuda'或'cpu'。
            zero_stage (int): Zero冗余优化器的阶段。

        返回:
            optimizer: 配置好的优化器实例。
        """
        # 从模型中获取所有参数及其名称
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤出需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 根据参数维度分组，2D及以上参数使用权重衰减
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # 构建优化器参数组
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 统计各组参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # 打印参数统计信息
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 根据设备类型选择是否使用融合版AdamW优化器
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        # 根据zero_stage选择优化器类型
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        根据给定的条件序列idx（形状为(b,t)的LongTensor），通过模型生成新的token序列。
        该函数将依据模型当前的状态，依次生成max_new_tokens个新的token，并将其反馈回模型以继续生成。
        在使用此功能时，通常需要确保模型处于eval()模式。
        
        参数:
        - idx: 长整型张量，形状为(b,t)，表示条件序列。
        - max_new_tokens: 整数，表示要生成的最大新token数量。
        - temperature: 浮点数，表示生成过程的温度参数，默认为1.0。
        - top_k: 整数或None，表示在生成时考虑的最高k个候选，默认为None。
        
        返回:
        - idx: 长整型张量，形状为(b, t+max_new_tokens)，表示生成的完整序列。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文长度超过模型的block_size，需要裁剪
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 将裁剪后的序列送入模型，获取序列下一位的logits
            logits, _ = self(idx_cond)
            # 对logits进行温度调整
            logits = logits[:, -1, :] / temperature
            # 如果指定了top_k，对logits进行裁剪，只保留最可能的top_k个
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用softmax函数将logits转换为概率
            probs = F.softmax(logits, dim=-1)
            # 从概率分布中采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将采样的token添加到序列中，并继续下一轮生成
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    """
    读取数据文件的头部信息。

    该函数用于检查二进制数据文件的格式是否正确，主要通过读取文件的头部信息来实现。
    它首先检查文件的魔数（一个特定的数字序列），以确认文件格式，然后读取其他头部信息，
    如数据版本和声称的令牌数量。目前，该函数仅返回令牌数量。

    参数:
    - filename: 要检查的二进制数据文件的名称。

    返回值:
    - ntok: 文件头部声称的令牌数量。

    注意:
    - 如果魔数不匹配或版本号不是1，函数将打印错误信息并退出。
    - 如果文件格式正确，该函数为数据处理或验证提供了一个快速的预检查机制。
    """
    # 以二进制模式打开文件，用于读取原始字节数据
    with open(filename, "rb") as f:
        # 首先读取头部，这里是256个int32整数（每个4字节）
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    
    # 检查魔数是否正确，这是确认文件格式的关键
    if header[0] != 20240520:
        # 如果魔数不匹配，打印错误提示信息并退出
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    
    # 确认版本号为1，其他版本目前不支持
    assert header[1] == 1, "unsupported version"
    
    # ntok是文件头部声称的令牌数量
    ntok = header[2] 
    
    # 目前仅返回令牌数量
    return ntok 

def _load_data_shard(filename):
    """
    从给定的二进制文件中加载数据片段。

    此函数用于从包含数据和元数据的二进制文件中加载数据。它首先读取文件的头部，
    检查魔法数字和版本号，然后读取实际的数据令牌。

    参数:
    - filename: 要加载数据的文件名。

    返回:
    - tokens: 从文件中读取的数据令牌数组。
    """
    # 打开文件以进行二进制读取
    with open(filename, "rb") as f:
        # 首先读取头部，其中包含256个int32整数（每个4字节）
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        # 检查魔法数字是否匹配，以验证文件格式
        assert header[0] == 20240520, "魔法数字在数据.bin文件中不匹配"
        # 检查版本号，目前只支持版本1
        assert header[1] == 1, "不支持的版本"
        ntok = header[2]  # 数据令牌的数量（声明的数量）
        # 剩余部分是令牌，存储为uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    # 确保读取的令牌数量与头部声明的数量一致
    assert len(tokens) == ntok, "读取的令牌数量与头部不匹配？"
    # 返回读取的令牌数组
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        """
        初始化函数，用于在数据加载器中设置各种参数和加载数据文件。

        参数:
        - filename_pattern: str，文件名模式，用于匹配数据文件。
        - B: int，批量大小。
        - T: int，序列长度。
        - process_rank: int，当前进程的排名。
        - num_processes: int，总进程数。

        该函数会根据提供的文件名模式搜索文件，验证数据分片，并计算总令牌数。
        """
        # 初始化类属性
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # 搜索符合模式的文件并排序
        self.files = sorted(glob.glob(filename_pattern))
        # 确保找到了匹配的文件
        assert len(self.files) > 0, f"未找到任何匹配模式 {filename_pattern} 的文件"

        # 加载和验证所有数据分片，统计总令牌数
        ntok_total = 0
        for fname in self.files:
            # 预览数据分片并获取令牌数
            shard_ntok = _peek_data_shard(fname)
            # 确保每个分片的令牌数至少满足最小要求
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        # 打印总令牌数和文件数
        print0(f"DataLoader: 总令牌数: {ntok_total:,}，跨越 {len(self.files)} 个文件")

        # 准备开始处理数据
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        # 加载第一个分片的数据  
        # 更新位置
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        ## 读取下一个分片数据
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        # 读取一个batch的数据
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        # 转tensor
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        # 生成训练数据，目标预测下一个词 ，形状为  (B, T)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # 移动读取数据的指针
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        # 如果加载下一个batch的数据超出范围，则加载下一个分片
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files
##TODO nedd to be read and comprehansion
## 保存权重以float32的形式
def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

## 保存权重以bfloat16的形式
def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.write(b)

## 保存gpt2 的权重参数
def write_tensors(model_tensors, L, file, dtype):
    # writes the GPT-2 model's weights to a binary file
    assert dtype in {"float32", "bfloat16"}
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    # 保存 词嵌入权重
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    #保存 位置编码权重
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    ## 保存 transformer的block的权重 ln1 attn ln2 mlp
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    ## last  ln func 权重
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
def pad_vocab(tensor, multiple=128, value=0):
    """
    为了提高GPU上的矩阵运算效率，对GPT-2的词汇表大小维度进行填充。
    GPT-2的词汇表大小为50,257，这个数字对于很多GPU上的矩阵操作来说不够“友好”。
    因此，我们将它填充到最接近的“友好”倍数，例如，当multiple=128时填充到50,304。
    这在算法上是一个无操作（NOOP），只是为了使张量操作更加高效。
    
    参数:
    tensor: 需要填充的二维张量，代表GPT-2的词汇表。
    multiple: 填充到的最接近的倍数，默认为128。
    value: 用于填充的数值，默认为0。
    
    返回:
    padded: 填充后的张量，其行数是multiple的倍数。
    """
    # 确保输入张量是二维的
    assert tensor.ndim == 2
    V, C = tensor.shape
    # 确保输入张量的行数是GPT-2词汇表大小
    assert V == 50257, "just being defensive here"
    # 计算填充后的词汇表大小，通过向上取整到最近的multiple倍数
    Vp = ((V + multiple - 1) // multiple) * multiple
    # 计算需要填充的行数
    pad_rows = Vp - V
    # 根据需要填充的行数，对张量进行填充，如果pad_rows为0，则不进行填充
    padded = tensor if pad_rows == 0 else F.pad(tensor, (0, 0, 0, pad_rows), value=value)
    # 确保填充后的张量形状正确
    assert padded.shape == (Vp, C)
    return padded

def write_model(model, filename, dtype):
    """
    将模型参数写入二进制文件。

    参数：
    model: 模型实例。
    filename: 保存模型的文件名。
    dtype: 模型参数的数据类型，支持"float32"和"bfloat16"。

    返回：
    无。
    """
    # 确保数据类型是支持的
    assert dtype in {"float32", "bfloat16"}  # float16 可能以后支持
    # 根据数据类型设置版本号
    version = {
        "float32": 3,  # 3: 所有张量都是fp32，填充过的词汇表
        "bfloat16": 5,  # 5: 所有张量都是bf16，填充过的词汇表
    }[dtype]
    # 初始化头部信息，包含模型配置的元数据
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326  # 魔法数字
    header[1] = version  # 检查点版本
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    # 获取模型的所有参数
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # 将词汇表填充到128的倍数，以提高C中的效率
    wte = params["transformer.wte.weight"]  # (V, C)
    wte_padded = pad_vocab(wte)  # (Vp, C)
    params["transformer.wte.weight"] = wte_padded  # (Vp, C)
    print(f"词汇表大小从 {wte.size(0)} 填充到 {wte_padded.size(0)}")
    header[7] = wte_padded.size(0)  # 填充后的词汇表大小存储在头部
    # 写入文件
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())  # 写入头部
        write_tensors(params, model.config.n_layer, file, dtype)  # 写入参数
    print(f"已写入 {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240327 # magic
    header[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    header[2] = x.size(0) # batch size of the batch, B
    header[3] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, value=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    print(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

def write_tokenizer(enc, filename):
    """
    将编码器信息保存到文件中，用于分词。

    参数:
    enc: 编码器对象，包含分词所需的映射信息。
    filename: 保存编码器信息的文件名。

    返回:
    无返回值，但会在指定路径创建一个用于分词的文件。
    """
    # 计算需要支持的最大token数量
    n = enc.max_token_value + 1
    # 创建一个头部数组，用于存储编码器的元数据信息
    header = torch.zeros(256, dtype=torch.int32)
    # 设置一个魔法数字，用于标识文件的类型
    header[0] = 20240328
    # 设置编码器版本为2，表示包括了EOT（End of Text）标记
    header[1] = 2
    # 存储token的数量
    header[2] = n
    # 存储EOT标记的token值
    header[3] = enc.eot_token
    # 打开指定文件，准备写入编码器信息
    with open(filename, "wb") as file:
        # 写入头部信息
        file.write(header.numpy().tobytes())
        # 遍历每个token，将其信息写入文件
        for i in range(n):
            # 解码单个token为字节序列
            b = enc.decode_bytes([i])
            # 检查字节序列长度，确保不超过255字节
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            # 写入字节序列的长度
            file.write(struct.pack("<B", length))
            # 写入实际的字节序列
            file.write(b)
    # 输出文件写入完成的信息
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    import time
    import argparse
    import tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=int, default=1, help="write tensors to disk")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    # 分布式多机多卡训练
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        # 确保CUDA可用，目前我们认为DDP（分布式数据并行）需要CUDA支持
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"

        # 初始化进程组，使用NCCL作为后端
        init_process_group(backend='nccl') 

        # 获取当前进程的全局排名
        ddp_rank = int(os.environ['RANK']) 

        # 获取当前进程的本地排名
        ddp_local_rank = int(os.environ['LOCAL_RANK'])

        # 获取整个分布式系统中的进程总数
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        # 根据本地排名选择对应的CUDA设备
        device = f'cuda:{ddp_local_rank}'

        # 设置当前进程使用的CUDA设备
        torch.cuda.set_device(device)

        # 判断当前进程是否为主进程，主进程将负责日志记录和模型保存等工作
        master_process = ddp_rank == 0 

        # 所有进程使用完全相同的随机种子
        seed_offset = 0 

        # 根据命令行参数设置ZeRO阶段
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # select the device
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    ## 总进程的读取tokens数量
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    # 梯度累积次数，数值够了执行反向传播
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    #精度设置训练
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')## tensorcore 高精度计算

    # turn on/off flash attention 是否开启flash attention
    assert args.flash in {0, 1}
    FLASH = args.flash

    # init (and write) the tokenizer
    # 拿到 tokenizer 编码器
    enc = tiktoken.get_encoding("gpt2")
    if master_process and args.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    if args.model[0] == "d":
        # from scratch (random weights)
        # 基本配置，得到随机初始化的权重
        model_config = {
            "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
            "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
            "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
            "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
        }[args.model]
        model = GPT(model_config)
    else:
        # load the GPT-2 model weights，加载预训练权重
        model = GPT.from_pretrained(args.model)
    model.train()### 设置训练模式
    model.to(device) ## 将模型参数移动到GPU
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)## 图优化

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens 加载tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    # 做一次测试 生成C测试数据
    if master_process and args.write_tensors and (not args.inference_only):
        x, y = train_loader.next_batch()## 获取batch数据
        x, y = x.to(device), y.to(device)## move to gpu
        logits, loss = model(x, y) # model forward pass
        loss.backward() #       # backward pass
        # save model params, in both float32 and bfloat16
        model_to_size = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_to_size.update({f"d{d}": f"d{d}" for d in [12, 24, 36, 48]})
        model_size_str = model_to_size[args.model] # e.g. "124M", or "d12"
        write_model(model, f"gpt2_{model_size_str}.bin", dtype="float32")
        write_model(model, f"gpt2_{model_size_str}_bf16.bin", dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, f"gpt2_{model_size_str}_debug_state.bin")
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    ## warp ddp 会让模型的字典多了 module 的 字符串
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup) 带预热的余弦退火学习率
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)

    # create the logging directory if it does not exist
    ## 打log
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        ### 做一次val
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        # 推理一次
        if (args.sample_every > 0 \
            and (step % args.sample_every == 0 or last_step)) \
            and master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            start_ids = [enc.eot_token]
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps### 梯度累加除以它
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            if not args.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.
        ###训练状态的 输出
        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()

import pandas as pd
import numpy as np
import torch
import flax.linen as nn
import jax.numpy as jnp
import math

from typing import Any, Callable, Optional, Tuple, Type

"""
순서
어텐션
MLP(피드포워드)
인코더 블록
인코더
"""
"""
나머지 주석은 나중에
"""

class MultiheadSelfAttention(nn.Module):
  def __init__(self, num_heads: int):
    self.num_heads = num_heads # 헤드 수
    super().__init__()
  
  @nn.compact # flax에서 모델을 사용자 정의할려면 @nn.compact을 붙여야 한다.
  def __call__(self,  inputs_q, inputs_k=None, inputs_v=None, mask=None):
    
    """
    입력 받는 텐선 inputs_q는 [batch_sizes..., length, features]의 형태로 배치 사이즈, 길이(N+1), D차원(768)으로 이루어져있음. 
    """

    if inputs_k is None:
      inputs_k = inputs_q # K나 V에 해당하는 텐서를 받지 못하였을 때, Q텐서로 K, V텐서를 만들겠다는 의미로 같은 토큰에서 Q, K, V를 뽑아내는 것
    if inputs_v is None:
      inputs_v = inputs_k
    d_model = inputs_q.shape[-1]  # D차원을 d_model에 저장 
    head_dim = d_model // self.num_heads  # d차원을 헤드의 수로 나누어서 각 헤드의 차원을 구함. 이를 위해 D차원은 num_heads로 나누어 떨어져야 함

    query = nn.DenseGeneral(features=(self.num_heads, head_dim), dtype=self.dtype, kernel_init=self.kernel_init, axis=-1, name="query")(inputs_q)
    # 텐서 Q에 대해서 마지막 차원에 대해 변환을 하여 [배치 사이즈, 길이, 헤드 수, 헤드 차원]으로 변환함
    # 이유는 멀티 헤드 셀프 어텐션을 위해 헤드의 차원을 나누어서 헤드별로 어텐션이 될 수 있게 만듬
    # DenseGeneral를 쓰는 이유는 결국 이게 가중치 학습(업데이트)이 가능한 임베딩 벡터의 효과를 주기 위해서
    key = nn.DenseGeneral(features=(self.num_heads, head_dim), dtype=self.dtype, kernel_init=self.kernel_init, axis=-1, name="key")(inputs_k)
    value = nn.DenseGeneral(features=(self.num_heads, head_dim), dtype=self.dtype, kernel_init=self.kernel_init, axis=-1, name="value")(inputs_v)

    atten = jnp.einsum("bqhd,bkhd->bhqk", query, key)
    # Q와 K차원이 [배치사이즈, 길이, 헤드수, 헤드차원]으로 되어 있는데 이를 Q와 K^T의 내적을 배치와 헤드는 고정시켜두고
    # q,d 와 k, d에 대해 d에 대해 내적하여 [배치사이즈, 헤드수, q길이, k길이] 형태로 만드는 것
    # 이는 Q에 대한 K의 유사도가 됨
    atten /= jnp.sqrt(head_dim)
    # d_k 차원을 어텐션 스코어를 루트 d_k로 나누어 스케일 내적을 마무리 

    if mask is not None:
      atten = jnp.where(mask == 0, -1e9, atten)
    # mask가 1 or 0인거에 따라서 0이면 -1e9(-10^9)을 넣어주고 아니면 원래 값을 넣어줌
    # 이후 소프트맥스할 때 마스킹된 부분이 0언저리가 됨.

    atten_weights = nn.softmax(atten, axis=-1)
    # 이후 어텐션 스코어에 소프트맥스를 해서 어텐션 분포를 얻음

    atten_output = jnp.einsum("bhqk,bkhd->bqhd", atten_weights, value)
    # 이번에는 위에서 구한 [배치사이즈, 헤드수, q길이, k길이]형태의 어텐션 분포와 [배치 사이즈, k길이, 헤드 수, 헤드 차원]형태의 value 텐서를 곱하는데
    # k길이에 대해 내적하여 [배치 사이즈, q길이, 헤드수, 헤드 차원] 형태로 만듬
    # 이는 Q에 대한 K의 유사도에 맞추어 K에 대한 실제 값인 헤드 차원 부분이 내적되면서 벨류의 텐서에 어텐션 분포가 반영이 된다

    atten_output = atten_output.reshape(inputs_q.shape[0], inputs_q.shape[1], self.num_heads * head_dim)
    # 다시 [배치사이즈, 길이, D차원] 형태로 변환함. 다시 합치는 Concat부
    output = nn.DenseGeneral(features=d_model, axis=-1, name="output")(atten_output)
    # 변환한 후 가중치 행렬 W^O를 취하기 위해서 학습 가능한 가중치를 레이어로 부여함

    return output

class MlpBlock(nn.Module):
  def __init__(self, mlp_dim):
    self.mlp_dim = mlp_dim
    super().__init__()

  @nn.compact
  def __call__(self, x):
    output_dim = x.shape[-1]
    x = nn.Dence(features=self.mlp_dim)(x)
    x = nn.gelu(x)
    y = nn.Dence(features=output_dim)(x)

    return y

class EncoderBlock(nn.Module):
  """
    해야할 거: LN갈기고, MSA갈기고 skip-connection갈기기
  """
  def __init__(self, num_heads: int, mlp_dim: int):
    self.num_heads = num_heads
    self.mlp_dim = mlp_dim
    super().__init__()

  @nn.compact
  def __call__(self, inputs):
    z = nn.LayerNorm()(inputs) # LN(레이어 정규화)적용
    z = MultiheadSelfAttention(num_heads=self.num_heads)(z, z) # MSA(멀티헤드셀프어텐션)적용
    z = z + inputs # 스킵커넥션 적용

    mlp = nn.LayerNorm()(z) # LN적용
    mlp = MlpBlock(mlp_dim=self.mlp_dim)(mlp) # MSA적용
    # 이후 스킵커넥션을 적용한 값을 반환
    return z+mlp

class Encoder(nn.Module):
  """
  해야할 거: 포지셔널 임베딩 한 거를 L개의 레이어에 넣기
  """
  def __init__(self, num_layers: int, num_heads: int, mlp_dim: int):
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.mlp_dim = mlp_dim
    super().__init__()

  @nn.compact
  def __call__(self, x, *, train):
    for layer in range(self.num_layers):
      x=EncoderBlock(num_heads=self.num_heads, mlp_dim=self.mlp_dim)
    encoder = nn.LayerNorm(x)
    return encoder

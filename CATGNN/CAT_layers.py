import numpy as np

import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

import torch.nn as nn
from torch_geometric.nn.conv  import MessagePassing
import torch.nn.functional as F
import torch_scatter
	
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
	
from e3nn import o3
from typing import Union, Optional, Dict

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

def find_activation(name: str) -> nn.Module:
	"""Return an activation function using name."""
	try:
		return {
			"relu": nn.ReLU,
			"silu": nn.SiLU,
			"gelu": nn.GELU,
			"softplus": nn.Softplus,
			"sigmoid": nn.Sigmoid,
			"tanh": nn.Tanh,
				}[name.lower()]()
	except KeyError as exc:
		raise NotImplementedError from exc


class MLP_layers(nn.Module):
	def __init__(self, input_dim: int, output_dim: int, hidden_dims: Union[int, list[int]], activation: str='softplus', dropout=0, bias: bool = True):
		super().__init__()
		if len(hidden_dims)==0 or hidden_dims is None:
			layers=nn.Linear(input_dim, output_dim, bias=bias)
			
		elif isinstance(hidden_dims, int):
			layers = [nn.Linear(input_dim, hidden_dims, bias=bias), nn.Dropout(dropout), find_activation(activation), nn.Linear(hidden_dims, output_dim, bias=bias)]
		
		elif isinstance(hidden_dims, list):
				#ayers = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0], bias=bias)])
			layers=[nn.Linear(input_dim, hidden_dims[0], bias=bias)]
			for h_in, h_out in zip(hidden_dims[:-1], hidden_dims[1:]):
				layers.append(nn.Linear(h_in, h_out))
			layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=bias))
				
		else:
			raise TypeError(
				f"{hidden_dims=} must be an integer, a list of integers, or None."
			)
		#print(type(layers), len(layers))
		#self.layers=nn.ModuleList(layers)
#		self.layers=layers
		self.layers=nn.Sequential(*layers)
	def	forward(self, x):
#		for layer in self.layers:
#			x=layer(x)
#		return x
		return self.layers(x)

	
	
class MHA_CAT(MessagePassing):
	def __init__(self, input_dim: int, output_dim: int, edge_dim: int, hidden_dims: Union[int, list[int]]=None, heads: int = 8, bias=True, concat=False, GAT_implement: bool=True):
		super().__init__(aggr='add',flow='target_to_source', )
		self.input_dim=input_dim
		self.output_dim=output_dim
		self.edge_dim=edge_dim
		self.heads=heads
			
		self.edge_weight_i = nn.Linear(edge_dim, edge_dim)
		self.edge_weight_j = nn.Linear(edge_dim, edge_dim)

		self.W_i = nn.Parameter(torch.Tensor(input_dim+edge_dim, heads*output_dim))
		self.W_j = nn.Parameter(torch.Tensor(input_dim+edge_dim, heads*output_dim))			
			
		self.bn1 = nn.BatchNorm1d(heads)
			
		#self.softmax=nn.Softmax()
		self.att = nn.Parameter(torch.Tensor(1,heads,2*output_dim))
		self.att_weight = nn.Parameter(torch.Tensor(heads, heads))
			
		self.concat=concat
		self.GAT_implement=GAT_implement
			
		#self.scale_edge_attr=nn.Linear(edge_dim, edge_dim)
		if bias and concat: self.bias = nn.Parameter(torch.Tensor(heads * output_dim))
		elif bias and not concat: self.bias = nn.Parameter(torch.Tensor(output_dim))
		else: self.register_parameter('bias', None)
		self.reset_parameters()
			
	def reset_parameters(self):
		glorot(self.W_i)
		glorot(self.W_j)
		glorot(self.att)
		glorot(self.att_weight)
		zeros(self.bias)

	def forward(self, edge_index, node_feats, edge_length_embedded):
			#print(edge_index)
		return self.propagate(edge_index, x=node_feats, edge_len_basis=edge_length_embedded)
		
	def message(self, edge_index_i, x_i, x_j, edge_len_basis):
		edge_len_i=self.edge_weight_i(edge_len_basis)
		edge_len_j=self.edge_weight_j(edge_len_basis)

		x_i = torch.cat([x_i, edge_len_i],dim=-1)
		x_j = torch.cat([x_j, edge_len_j],dim=-1)


#		x_i = F.softplus(torch.matmul(x_i, self.W))
#		x_j = F.softplus(torch.matmul(x_j, self.W))
		x_i = F.softplus(torch.matmul(x_i, self.W_i))
		x_j = F.softplus(torch.matmul(x_j, self.W_j))
			
		x_i   = x_i.view(-1, self.heads, self.output_dim)
		x_j   = x_j.view(-1, self.heads, self.output_dim)

		if self.GAT_implement:
			alpha = F.softplus((torch.cat([x_i, x_j], dim=-1)*self.att).sum(dim=-1))
			#print(alpha)
			#print('alpha shape:', alpha.shape)
		else:
			alpha=F.softplus(torch.cat([x_i, x_j], dim=-1)).sum(dim=-1)
			#print(alpha)
			alpha=torch.matmul(alpha, self.att_weight)		
			#print(alpha)
	
		alpha = F.softplus(self.bn1(alpha))
		#print(x_j.shape)

		alpha = softmax(alpha,edge_index_i)
#		alpha = softmax(alpha,edge_index_j)
		#print(alpha)
		x_j   = (x_j * alpha.view(-1, self.heads, 1)).transpose(0,1)
#		x_i   = (x_i * alpha.view(-1, self.heads, 1)).transpose(0,1)
		
		return x_j
#		return x_i
		
	def update(self, aggr_out,x):
		if self.concat is True:    aggr_out = aggr_out.view(-1, self.heads * self.output_dim)
		else:                      aggr_out = aggr_out.mean(dim=0);# print(aggr_out.shape); exit()
		if self.bias is not None:  aggr_out = aggr_out + self.bias
		return aggr_out


class Elements_Attention(torch.nn.Module):
	def __init__(self,neurons):
		'''
		Global-Attention Mechanism based on the crystal's elemental composition
		'''
		super(Elements_Attention, self).__init__()
		self._cgcnn_embed=nn.Linear(103, 103)
		self.softplus1 = nn.Softplus()
		self.node_layer1 = nn.Linear(neurons+103,32)
		self.bn1 = nn.BatchNorm1d(32)

		self.atten_layer = nn.Linear(32,1)
		self.bn2 = nn.BatchNorm1d(32)

	def forward(self,x,batch,global_feat):
		# pass num_graphs as argument instead of batch
#		counts = torch.unique(batch,return_counts=True)[-1]
		#print(x.shape, global_feat.shape)
		global_feat=self._cgcnn_embed(global_feat)
		global_feat=self.softplus1(global_feat)
		graph_embed = global_feat
#		graph_embed = torch.repeat_interleave(graph_embed, counts, dim=0)
		chunk = torch.cat([x,graph_embed],dim=-1)

		x = self.bn1(F.softplus(self.node_layer1(chunk)))
		x = self.atten_layer(x)
		weights = softmax(x,batch)
		return weights


import numpy as np

import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

import torch.nn as nn

import torch.nn.functional as F
import torch_scatter
	
#from e3nn import o3
from typing import Union, Optional, Dict

from CATGNN.CAT_layers import MHA_CAT, Elements_Attention, find_activation
from CATGNN.conv_layer import Network


default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


			
class PeriodicNetwork(Network):
	def __init__(self, in_dim, out_dim, edge_dim, n_GAT_layers=3, nonlinear_post_scatter: bool=True, **kwargs):                        
		super().__init__(**kwargs)

		# embed the mass-weighted one-hot encoding
		self.emx = nn.Linear(in_dim, out_dim)
		self.emz = nn.Linear(in_dim, out_dim)
			
		# embed the gaussian basis of bond length
		self.edge_em = nn.Linear(edge_dim, out_dim)
		#self.edge_em = nn.Linear(edge_dim, edge_dim)
			
		self._activation = find_activation('softplus') #torch.nn.SiLU()
		#self._activation = torch.nn.GELU()
		#self._activation = torch.nn.PReLU()
		#self._activation = torch.nn.Softplus()

		self.GAT=nn.ModuleList([MHA_CAT(input_dim=out_dim, output_dim=out_dim, edge_dim=out_dim, heads=8, GAT_implement=False) for i in range(n_GAT_layers)])
		self.batch_norm = nn.ModuleList([nn.BatchNorm1d(out_dim) for i in range(n_GAT_layers)])

		#self.node_att = nn.ModuleList([MHA_CAT(input_dim=out_dim, output_dim=out_dim, edge_dim=edge_dim, heads=8) for i in range(nl)])
		#self.batch_norm     = nn.ModuleList([nn.BatchNorm1d(n_h) for i in range(nl)])	
			
		self.E_Atten = Elements_Attention(out_dim)
			
		self.nonlinear_post_scatter=nonlinear_post_scatter
			
		self.linear1 = nn.Linear(out_dim, out_dim)
		self.linear2 = nn.Linear(out_dim, out_dim)
		self.out=nn.Linear(out_dim, 1)

	def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
		num_graphs=data["ptr"].numel() - 1
		#data.x = F.relu(self.em(data.x))
		#data.z = F.relu(self.em(data.z))
		cgcnn_feats=data.x
		data.x = self._activation(self.emx(data.x))
		data.z = self._activation(self.emz(data.z))

#		if hasattr(data, 'z'):
#			data.z = self._activation(self.emz(data.z))
#			print('data has z')
#		else:
#			print('data does not have z')

		output = super().forward(data)
		#print(output)
		
		output = self._activation(output)
		#output = torch.relu(output)
		#output = F.softplus(output)

		edge_length_embedded = self._activation(self.edge_em(data.edge_length_embedded))
		
		pre_output=output
		for a_idx in range(len(self.GAT)):
			output=self.GAT[a_idx](data.edge_index, output, edge_length_embedded = edge_length_embedded)
			output = self.batch_norm[a_idx](output)
			output = F.softplus(output)
			output = torch.add(output, pre_output)
			pre_output=output
		"""
		output=self.GAT(data.edge_index, output, edge_length_embedded = edge_length_embedded)
		output=self.batch_norm(output)

		print(output)
		print(output.shape)
		"""
	
#		ag = self.E_Atten(output, data.batch, data.comp_feats)
		ag = self.E_Atten(output, data.batch, cgcnn_feats)

#		output=(output)*ag
		output = torch.add(output, (output)*ag)
		#print(output.shape, ag.shape)
		#print(output, ag)
		#exit()
		y = torch_scatter.scatter_mean(src=output, index=data.batch, dim=0,)# dim_size=num_graphs)
		#output = torch_scatter.scatter_sum(output, data.batch, dim=0)
		#energy = scatter_sum(src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs)
			
		#print(y.shape)
		#exit()
		#y = global_add_pool(x,data.batch)
		if self.nonlinear_post_scatter:
			#y = self._activation(y)
			y=self.linear1(y)
			y = self._activation(y)
			y=self.linear2(y)
			y = self._activation(y)
#		else:
#		y=self.linear2(y)

		y=self.out(y).squeeze()
		return y

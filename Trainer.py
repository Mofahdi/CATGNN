import os
import numpy as np
import math
import time

import pandas as pd 
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch


bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)



def loglinspace(rate, step, end=None):
	t = 0
	while end is None or t <= end:
		yield t
		t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

def validate(model, dataloader, loss_fn, loss_fn_mae, device):
	model.eval()
	loss_cumulative = 0.
	loss_cumulative_mae = 0.
	start_time = time.time()
	with torch.no_grad():
		for batch, data in enumerate(dataloader):
			data.to(device); #d.to(device)
			output = model(data)
			loss = loss_fn(output, data.prop).cpu()
			loss_mae = loss_fn_mae(output, data.prop).cpu()
			loss_cumulative = loss_cumulative + loss.detach().item()
			loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
			#print(loss_cumulative)
	return loss_cumulative/len(dataloader), loss_cumulative_mae/len(dataloader)

#train(model, optimizer, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, run_name, max_iter=101, scheduler=None, device="cpu"):
#	model.to(device)

def train(model, optimizer, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, run_name, max_iter=101, scheduler=None, device="cpu", output_dir='results', patience=10):
	model.to(device)

	checkpoint_generator = loglinspace(0.3, 1)
	checkpoint = next(checkpoint_generator)
	start_time = time.time();
	min_valid_avg_loss = np.inf
	trigger_times = 0
	
	if not os.path.isdir(os.path.join(os.getcwd(), output_dir)):
		os.mkdir(output_dir)
	
	try: model.load_state_dict(torch.load(os.path.join(output_dir, run_name + '.torch'))['state'])
	except:
		results = {}
		history = []
		s0 = 0
	else:
		results = torch.load(os.path.join(output_dir, run_name + '.torch'))
		history = results['history']
		s0 = history[-1]['step'] + 1
		min_valid_avg_loss=history[-1]['valid']['mean_abs']
	#print(min_valid_avg_loss)

	for step in range(max_iter):
		model.train()
		loss_cumulative = 0.
		loss_cumulative_mae = 0.
        
		for batch, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train), bar_format=bar_format):
			data.to(device); #d.to(device)
			output = model(data)
			loss = loss_fn(output, data.prop).cpu()
			loss_mae = loss_fn_mae(output, data.prop).cpu()
			loss_cumulative = loss_cumulative + loss.detach().item()
			loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()

			optimizer.zero_grad()
			loss_mae.backward()
			optimizer.step()

		end_time = time.time()
		wall = end_time - start_time

		if step == checkpoint:
			checkpoint = next(checkpoint_generator)
			assert checkpoint > step

			valid_avg_loss = validate(model, dataloader_valid, loss_fn, loss_fn_mae, device)
			train_avg_loss = validate(model, dataloader_train, loss_fn, loss_fn_mae, device)
			
			
			history.append({
				'step': s0 + step,
				'wall': wall,
				'batch': {
					'loss': loss.item(),
					'mean_abs': loss_mae.item(),
				},
				'valid': {
					'loss': valid_avg_loss[0],
					'mean_abs': valid_avg_loss[1],
				},
				'train': {
					'loss': train_avg_loss[0],
					'mean_abs': train_avg_loss[1],
				},
			})

			results = {
				'history': history,
				'state': model.state_dict()
			}

			print(f"Iteration {step+1:4d}   " +
					f"train loss = {train_avg_loss[1]:8.4f}   " +
					f"valid loss = {valid_avg_loss[1]:8.4f}   " +
					f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")


			if min_valid_avg_loss > valid_avg_loss[1]: # MAE validation loss
				print(f'Validation Loss Decreased({min_valid_avg_loss:.6f}--->{valid_avg_loss[1]:.6f}) \t Saving The Model\n')
				min_valid_avg_loss=valid_avg_loss[1]
				torch.save(model.state_dict(), os.path.join(output_dir, run_name +'_best_model.pth'))
				trigger_times = 0
				#best_model=model
			else:
				trigger_times += 1
				if trigger_times>=patience:
					print('Early stopping! start testing process')
					return

			with open(os.path.join(output_dir, run_name + '.torch'), 'wb') as f:
				torch.save(results, f)

		if scheduler is not None:
			scheduler.step(); #print('scheduler lr', scheduler.get_last_lr())
	end_time = time.time(); 
	total=end_time - start_time
	print('total time taken:', {time.strftime('%H:%M:%S', time.gmtime(total))})
	#return best_model
	

def train_val_loss_plot(output_dir, device, run_name, filename='history.jpg', dpi=600):
	history = torch.load(os.path.join(output_dir, run_name + '.torch'), map_location=device)['history']
	steps = [d['step'] + 1 for d in history]
	#loss_train = [d['train']['loss'] for d in history]
	#loss_valid = [d['valid']['loss'] for d in history]
	loss_train = [d['train']['mean_abs'] for d in history]
	loss_valid = [d['valid']['mean_abs'] for d in history]

	fig, ax = plt.subplots(figsize=(6,5))
	ax.plot(steps[::5], loss_train[::5], '-', label="Training",) #color=colors['train'])
	ax.plot(steps[::5], loss_valid[::5], '-', label="Validation",) #color=colors['valid'])
	ax.set_xlabel('epochs')
	ax.set_ylabel('loss')
	ax.legend(frameon=False);
	#ax.tight_layout()
	fig.tight_layout()
	
	output_dir=os.path.join(os.getcwd(), output_dir)
	if os.path.isdir(output_dir):
		fig.savefig(os.path.join(output_dir, filename), dpi=dpi)
#	else:
#		os.mkdir(output_dir)
#		fig.savefig(os.path.join(output_dir, filename), dpi=dpi)
	print('model history plot created')
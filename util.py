import torch
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt
import numpy as np

def get_device(cuda):
	"""
	returns the device used in the system (cpu/cuda)
	"""
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device : " + str(device))
	return device

def gen_vectors(dim):
    dict_vectors = {}
    start = [1.0, -1.0, 1.0, -1.0]
    multiplier = [1.0, -1.0, -1.0, 1.0]
    for index, val in enumerate(start):
        list_tmp = []
        current_val = val
        dim_tmp = dim[1]
        list_tmp.append(current_val)
        while dim_tmp > 1:
            current_val *= multiplier[index]
            list_tmp.append(current_val)
            dim_tmp -= 1
        dict_vectors[index] = list_tmp

    return dict_vectors


def target_conversion(dict_vectors, list_targets):
    return [dict_vectors[x] for x in list_targets]


def eval_model(enc, disc, dataset, device, folder):
	"""
	Used to evaluate a model after training.
	"""

	batch_size = 32

	# Loss Functions
	DiscLoss = nn.CrossEntropyLoss().to(device)

	enc.eval()
	disc.eval()

	# get the data loader
	dataloader = torch.utils.data.DataLoader(dataset[folder], batch_size=batch_size)

	loss_epoch = 0.
	correct_epoch = 0

	for data, target in dataloader:

		# data.shape = [n,3,l,b]
		# target.shape = [n]

		# put datasets on the device
		data = data.to(device)
		target = target.to(device)
		#print(data.shape)

		# get output of discriminator
		hidden = enc(data).view(data.shape[0], -1)
		out = disc(hidden)

		# calculate loss and update params
		loss = DiscLoss(out, target)

		# get accuracy and loss_epoch
		correct = torch.sum(target == torch.argmax(out, 1))

		loss_epoch += len(data)*loss
		correct_epoch += correct

	loss_epoch = loss_epoch/len(dataset[folder])
	accuracy = correct_epoch*100.0/float(len(dataset[folder]))
	result = {'loss_epoch' : loss_epoch.item(), 'accuracy' : accuracy.item()}
	# Pretty Printing
	print("[%s] Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
		(folder, loss_epoch, correct_epoch, len(dataset[folder]), accuracy))
	return result
	
def save_model(model, name):
	if "models" not in os.listdir():
		os.mkdir("models")

	if model is not None:
		torch.save(model.state_dict(), "models/"+name)

def test_model(enc, disc, dataset, device, folder):
	"""
	Used to evaluate a model after training.
	"""

	batch_size = 32

	# Loss Functions
	DiscLoss = nn.CrossEntropyLoss().to(device)

	enc.eval()
	disc.eval()

	# get the data loader
	dataloader = torch.utils.data.DataLoader(dataset[folder], batch_size=batch_size)

	loss_epoch = 0.
	correct_epoch = 0
	batch_probs = []
	labels = []
	preds_list = []
	prob1_list = []

	for data, target in dataloader:

		# data.shape = [n,3,l,b]
		# target.shape = [n]

		# put datasets on the device
		data = data.to(device)
		target = target.to(device)

		# get output of discriminator
		hidden = enc(data).view(data.shape[0], -1)
		out = disc(hidden)   #predicted output probs for two classes

		preds = out.detach().cpu()  #copy in preds
		probs = out    #take only the probs of class1
		#print(probs)
		#probs = probs.detach().cpu().numpy()
		#print("probs shape = ",probs.shape)
		#probs = probs.tolist()
		#print("probs shape = ",len(probs))
		#batch_probs.extend(probs)
		label = target.detach().cpu().numpy().tolist()
		labels+=label

		#preds = torch.round(preds.detach().cpu())
		preds = torch.max(preds,1).indices
		preds = preds.tolist()
		preds_list += preds 
		m=nn.Softmax()
		for prob in probs:
			prob=m(prob)
			prob=prob.cpu().detach().numpy().tolist()
			prob1_list.append(prob)
	#print(preds_list)
	#return batch_probs,preds_list,labels



				# calculate loss and update params
		loss = DiscLoss(out, target)

		# get accuracy and loss_epoch
		correct = torch.sum(target == torch.argmax(out, 1))

		loss_epoch += len(data)*loss
		correct_epoch += correct
	#batch_p = np.array(batch_probs)
	#print("shape of batch_p ", batch_p.shape)

	loss_epoch = loss_epoch/len(dataset[folder])
	accuracy = correct_epoch*100.0/float(len(dataset[folder]))
	result = {'loss_epoch' : loss_epoch.item(), 'accuracy' : accuracy.item()}
	# Pretty Printing
	print("[%s] Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
		(folder, loss_epoch, correct_epoch, len(dataset[folder]), accuracy))
	#print("probs list = ",batch_probs)
	return result, prob1_list , preds_list , labels

def visualize_embedding(enc, dataset, device, folder, directory, weighted, binary, w_n, thresh, beta):
	"""
	To visualize the embeddings of the encoder
	"""

	variance = 0.1
	batch_size = 32
	num_graphs = 5
	enc.eval()

	# get the data loader
	dataloader = torch.utils.data.DataLoader(dataset[folder], batch_size=batch_size, shuffle=False)

	H = []
	T = []
	W = []

	for data, target in dataloader:

		# data.shape = [n,3,l,b]
		# target.shape = [n]

		# put datasets on the device
		data = data.to(device)
		target = target.to(device)

		# get output of discriminator
		hidden = enc(data).view(data.shape[0], -1)

		h1 = np.array(hidden.detach().cpu())
		generated_vectors = gen_vectors(h1.shape)
		target_mapping = target_conversion(generated_vectors,target.cpu().tolist())
		target_mapping = torch.tensor(target_mapping).to(device)

		H.append(hidden)
		T.append(target)

		# getting weights
		weights = None # placeholder
		if weighted is True and binary is True:

			weights = hidden - target_mapping
			weights = beta*weights*weights
			weights = torch.sum(weights, 1)
			weights = torch.exp(-1*weights)

			if w_n is True:
				weights = weights / torch.sum(weights)

			weights = (weights > thresh).long()

			W.append(weights)


	H = torch.cat(H,0)
	T = torch.cat(T,0)

	h1 = np.array(H.detach().cpu())
	#print(h1.shape)
	generated_vectors = gen_vectors(h1.shape)
	target_mapping = target_conversion(generated_vectors,T.cpu().tolist())
	target_mapping = torch.tensor(target_mapping).to(device)
	#print(target_mapping.shape)

	if weighted is True and binary is True:
		W = torch.cat(W,0)

	prior = (torch.randn(H.shape)*variance).to(device)
	#print(prior.shape)
	prior = prior + target_mapping	 # just add the class index to mean

	H = H.detach().cpu().numpy()
	T = T.detach().cpu().numpy()
	prior = prior.detach().cpu().numpy()

	print(H,len(H))
	print(T,len(T))
	print(prior)

	if "plots" not in os.listdir():
		os.mkdir("plots")

	if directory not in os.listdir("plots"):
		os.mkdir("plots/"+directory)

	for idx in range(num_graphs):

		idxs = random.sample(range(H.shape[1]), k=4)
		print()
		print("Using Indices "+str(idxs))

		# The actual plotting
		# RED : hidden 0
		# GREEN : prior 0
		# BLUE : hidden 1
		# YELLOW : prior 1

		if weighted is False or binary is False:
			for i in range(len(H)):
				if T[i] == 0:
					plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]] ,'ro', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]], 'go', markersize=2)
				elif T[i] == 1:
					plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]], 'co', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]],prior[i,idxs[2]],prior[i,idxs[3]], 'mo', markersize=2)
				elif T[i] == 2:
					plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'ko', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]],prior[i,idxs[2]],prior[i,idxs[3]], 'wo', markersize=2)
				else:
					plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'bo', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]],prior[i,idxs[2]],prior[i,idxs[3]], 'yo', markersize=2)
		else:
			for i in range(len(H)):
				if T[i] == 0:
					if W[i] == 1:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]], 'ro', markersize=2)
					else:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]], 'r+', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]],prior[i,idxs[2]],prior[i,idxs[3]], 'go', markersize=2)

				elif T[i] == 1:
					if W[i] == 1:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]], 'co', markersize=2)
					else:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]], 'c+', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]],prior[i,idxs[2]],prior[i,idxs[3]], 'mo', markersize=2)

				elif T[i] == 2:
					if W[i] == 1:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]], 'wo', markersize=2)
					else:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]], 'w+', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]],prior[i,idxs[2]],prior[i,idxs[3]], 'ko', markersize=2)

				else:
					if W[i] == 1:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]], 'bo', markersize=2)
					else:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], H[i,idxs[2]], H[i,idxs[3]], 'b+', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]],prior[i,idxs[2]],prior[i,idxs[3]], 'yo', markersize=2)

		plt.savefig("plots/%s/%d_(%d_%d).jpg"%(directory,idx+1,idxs[0], idxs[1]))
		plt.show()

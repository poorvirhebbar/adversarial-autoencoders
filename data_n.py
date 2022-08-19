from PIL import Image
import os
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
import random


#---------------------- DUMMY-DATASET ----------------------#

class DummyDataset(Dataset):

	def __init__(self, size, shape, num_classes=2):
		"""
		size is the number of samples per class
		shape is the shape of each sample

		size =:= 100
		shape =:= [3,256,256]
		num_classes =:= 4

		"""
		self.size = size
		self.shape = shape
		self.num_classes = num_classes

	def __len__(self):
		return self.num_classes*self.size

	def __getitem__(self, idx):

		target = idx % self.num_classes
		# normalize dataset externally; don't worry about it now

		data = torch.randn(self.shape)*0.1 + target

		return data, torch.tensor(target)

def getDummyDataset():

	size = [500, 100]
	shape = [3,128,128]
	num_classes = 2

	# size = [5,1]
	# shape = [3]
	# num_classes = 2

	return {"train":DummyDataset(size[0], shape, num_classes), "eval":DummyDataset(size[1], shape, num_classes)}

#---------------------- BREAKHIS-DATASET ----------------------#
def make_weights_for_balanced_classes(dataset_train, nclasses):                        
    count = [0] * nclasses                                                                                                               
    for item in dataset_train:
        count[item[1]] += 1  
    print("count = ",count)
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count)) 
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(dataset_train) 
    for idx, val in enumerate(dataset_train): 
        weight[idx] = weight_per_class[val[1]]
    return weight



class BreakhisDataset(Dataset):
	
    def __init__(self, folder, noise=False, p=0):

        super(BreakhisDataset, self).__init__()

        self.shape = [3,384,512]
        self.num_classes = 4

        self.base = "../noise9_dataset"

        self.ben = os.listdir(self.base +"/" + folder + "/Benign/" ) # 0
        self.ins = os.listdir(self.base+"/" + folder + "/InSitu/" ) # 1
        self.inv = os.listdir(self.base+"/"+ folder + "/invasive/" ) # 2
        self.nor = os.listdir(self.base+"/"+ folder + "/Normal/" ) # 3

        self.ben = [(self.base+"/"+folder+"/Benign/"+file, 0) for file in self.ben]
        self.ins = [(self.base+"/"+folder+"/InSitu/"+file, 1) for file in self.ins]
        self.inv = [(self.base+"/"+folder+"/invasive/"+file, 2) for file in self.inv]
        self.nor = [(self.base+"/"+folder+"/Normal/"+file, 3) for file in self.nor]

        self.all = self.ben + self.ins + self.inv + self.nor
        
        p = (p/100.0)*len(self.all)

        if noise is True:
            choices = random.sample(range(len(self.all)), k=int(p))
            for idx in choices:
                self.all[idx] = (self.all[idx][0], (self.all[idx][1]+1)%2)

        print("About Dataset [%s]"%(folder))
        print("Benign: %d"%(len(self.ben)))
        print("InSitu: %d"%(len(self.ins)))
        print("Invasive: %d"%(len(self.inv)))
        print("Normal: %d"%(len(self.nor)))
        print("Total: %d"%(len(self.all)))
        print("Noise: %04f"%((p*100.0)/len(self.all)))
        print("Shape: "+str(self.shape))
        print()

    def __len__(self):
        return len(self.all)

    def __getitem__(self, idx):

        img = Image.open(self.all[idx][0])
        target = self.all[idx][1]

        img = img.resize(self.shape[1:][::-1])
        #t2 = T.Resize((400,400))
        #r = t2(img)
        trans = T.ToTensor()
        r = (2.*trans(img))-1.
        
        
        return (r, torch.tensor(target).long())


def getBreakhisDataset():

	return{ "train":BreakhisDataset("train",False,0.0),
			#"eval":BreakhisDataset("eval")}
			"test":BreakhisDataset("test")}


# For some dummy testing

# data = getDummyDataset()
# for i in range(len(data["train"])):
# 	print(data["train"][i])

# data = getBreakhisDataset()
# print(data["train"][0][0].shape)
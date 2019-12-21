import os
from glob import glob

base_dir = '/local/zjf/tiny-imagenet-200'
train_dir = os.path.join(base_dir, 'train')
train_subfolders = glob(train_dir + '/*')

with open('train.txt','w') as f:
	label_dict = dict()
	for idx, subfoler in enumerate(train_subfolders):
		label_dict[subfoler.rstrip().split('/')[-1]] = idx
		images = glob(os.path.join(subfoler, 'images') +'/*JPEG')
		for image in images:
			line = image + ' ' + str(idx) +'\n'
			f.write(line)

val_dir = os.path.join(base_dir, 'val')
val_ann_path = os.path.join(val_dir, 'val_annotations.txt')
with open(val_ann_path,'r') as f:
	lines = f.readlines()
with open('val.txt','w') as f:
	for line in lines:
		image_name = line.split('\t')[0]
		label = label_dict[line.split('\t')[1]]
		image = os.path.join(val_dir,'images',image_name)
		line = image + ' ' +str(label) +'\n'
		f.write(line)
		
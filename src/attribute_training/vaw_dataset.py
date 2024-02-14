from torch.utils.data import Dataset
import pickle 
import json 
import os 
import numpy as np
"""
annotation dir: directory with VAW annotations. Files should be train,val,test. Each annotation file is a dictonary from instance id to entry, not the original VAW annotations.
feature dir: directory with features of cropped images
sample_count_file: file that contains number of attributes in the dataset 
index_to_attribute: file for index to attribute 
attribute_to_index: file for attribute to idnex 
"""
class VAW(Dataset):
    def __init__(self,index_to_attribute,attribute_to_index,annotation_dir,feature_dir,sample_count_file,sample_neg_count_file,split='train',class_index=-1):
        self.id_to_attribute = self.open_file(index_to_attribute)
        self.attribute_to_id = self.open_file(attribute_to_index)
        annotation_name = os.path.join(annotation_dir,split+'.json')
        self.annotations = self.open_file(annotation_name)
        self.feature_files = os.listdir(os.path.join(feature_dir,split))
        self.feature_dir = feature_dir 
        self.class_index = class_index 
        self.split = split 
        self.num_pos_samples_per_attribute = self.open_file(sample_count_file)
        self.num_neg_samples_per_attribute = self.open_file(sample_neg_count_file)
    def __len__(self):
        # not all annotations have segmentations 
        return len(self.feature_files)
    def __getitem__(self,idx):
        feature_name = self.feature_files[idx]
        ext = os.path.splitext(feature_name)[1]
        feature_id = feature_name.replace(ext,'')
        features = self.open_file(os.path.join(self.feature_dir,self.split,feature_name),use_json=False)
        full_annotation = self.annotations[feature_id]
        positive_labels,negative_labels,no_attribute_labels = self.construct_labels(full_annotation['positive_attributes'],full_annotation['negative_attributes'])
        if self.class_index == -1:
            return {'image':features,'positive':positive_labels,'negative':negative_labels,"unknown":no_attribute_labels}
        else:
            # get index of class 
            return {'image':features,'positive':positive_labels[self.class_index],'negative':negative_labels[self.class_index],'unknown':no_attribute_labels[self.class_index]}
    def construct_labels(self,positive_attributes,negative_attributes):
        positive_labels = np.zeros(620).astype(int)
        negative_labels = np.zeros(620).astype(int)
        no_attribute_labels = np.ones(620).astype(int)
        for p in positive_attributes:
            positive_labels[self.attribute_to_id[p]] = 1 
        for n in negative_attributes:
            negative_labels[self.attribute_to_id[n]] =1 
        for i in range(620):
            if positive_labels[i] !=1 and negative_labels[i]!=1:
                no_attribute_labels[i] = 0 
        #pnegative_labels = np.ones(620).astype(int)-positive_labels
        return positive_labels,negative_labels,no_attribute_labels
    def open_file(self,filename,use_json=True):
        if use_json:
            with open(filename) as fopen:
                contents = json.load(fopen)
        else:
            with open(filename,'rb') as fopen:
                contents = pickle.load(fopen)
        return contents 


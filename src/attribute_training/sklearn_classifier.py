from sklearn.linear_model import LogisticRegression
import pickle
import os 
import sys 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import os
import numpy as np
from tqdm import tqdm
import argparse 
import torch 
import clip 
import logging 
import gc
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import json 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def open_file(filename,use_json=True):
        if use_json:
            with open(filename) as fopen:
                contents = json.load(fopen)
        else:
            with open(filename,'rb') as fopen:
                contents = pickle.load(fopen)
        return contents 
def save_file(filename,contents):
    with open(filename,'w+') as fwrite:
        json.dump(contents,fwrite)

def initialize_classifiers(attribute_name, clip_model, args):
    logit_scale = torch.ones([])*np.log(1/.07)
    attribute_vec = clip.tokenize([attribute_name])
    weight = clip_model.encode_text(attribute_vec).float()
    weight /=weight.norm(dim=-1,keepdim=True)
    weight*=logit_scale.exp()

    logger.info('Initialized clip weights')
    return weight.numpy()


def load_features_for_class(args,attribute_name,annotations,feature_dir):
    features = []
    # 0s or 1s
    labels = []
    entries = []
    for i,instance_id in enumerate(tqdm(annotations.keys())):
        entry = annotations[instance_id]
        attribute_found = False 
        positive = False
        negative = False 
        for attribute in entry['positive_attributes']:
            if attribute == attribute_name:
                attribute_found = True 
                positive = True
        for attribute in entry['negative_attributes']:
            if attribute == attribute_name:
                attribute_found = True 
                negative = True
        if attribute_found:
            if os.path.exists(os.path.join(feature_dir,instance_id+'.pkl')):
                feature_file = open_file(os.path.join(feature_dir,instance_id+'.pkl'),use_json=False).squeeze(0)
                feature_file/=np.linalg.norm(feature_file)
                if positive:
                    label = 1 
                else:
                    assert negative == True 
                    label = 0

                features.append(feature_file)
                labels.append(label)
                entries.append(i)
    print(entries)
    return {'features':np.stack(features),'labels':np.stack(labels)}

def train_classifier(args,train_feature_dict,initial_features):
    classifier = LogisticRegression(verbose=1,max_iter=args.iterations,n_jobs=1,C=0.001,class_weight='balanced')
    classifier.coef_ = initial_features
    classifier.fit(train_feature_dict['features'],train_feature_dict['labels'])
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path,f'classifier_{args.class_id}.pkl'),'wb') as f:
        pickle.dump(classifier,f)

def eval_classifier(args,test_feature_dict):
    with open(os.path.join(args.save_path,f'classifier_{args.class_id}.pkl'),'rb') as f:
        classifier = pickle.load(f)
    classifier.coef_ = classifier.coef_.astype(np.float32)
    classifier.intercept_ = classifier.intercept_.astype(np.float32)

    roc_auc = roc_auc_score(test_feature_dict['labels'],classifier.decision_function(test_feature_dict['features']))
    ap_score = average_precision_score(test_feature_dict['labels'],classifier.decision_function(test_feature_dict['features']))
    print(f'ROC AUC:{roc_auc}')
    print(f'AP SCORE:{ap_score}')
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    with open(os.path.join(args.results_dir,f'results_class_{args.class_id}.json'),'w+') as fwrite:
        json.dump({'roc_auc':roc_auc,'ap_score':ap_score},fwrite)
    
def train_and_evaluate(args):
    id_to_attribute = open_file(os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/index_to_attribute.json'))
    #train_annotations = open_file(os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/train.json'))
    print('Loading train features')
    attribute_name = id_to_attribute[str(args.class_id)]
    # train_feature_dict = load_features_for_class(args,attribute_name,train_annotations,os.path.join(args.feature_dir,'train'))
    # feature_shape = train_feature_dict['features'].shape
    # print(f'Train Feature shape:{feature_shape}')
    # clip_model = clip.load("ViT-L/14", jit=False, device='cpu')[0].eval().requires_grad_(False)
    # initial_features = initialize_classifiers(attribute_name, clip_model, args)
    # train_classifier(args,train_feature_dict,initial_features)
    test_annotations = open_file(os.path.join(args.data_dir,'visual_genome/vaw_dataset/data/test.json'))
    test_feature_dict = load_features_for_class(args,attribute_name,test_annotations,os.path.join(args.feature_dir,'test'))
    # test_feature_shape = test_feature_dict['features'].shape
    # print(f'Test Feature shape:{test_feature_shape}')
    
    # eval_classifier(args,test_feature_dict)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/sklearn_classifiers")
    parser.add_argument("--data_dir", type=str, default='/scratch/bcgp/datasets')
    parser.add_argument("--class_id",type=int,help="value from 0 to 619",default=0)
    parser.add_argument("--feature_dir",type=str,default="/scratch/bcgp/datasets/vaw_cropped/features")
    parser.add_argument("--results_dir",type=str,default="/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/sklearn_results_sag")
    parser.add_argument('--iterations',
    default=1000,
    help='Number of iterations to run log regression')
    args = parser.parse_args()
    train_and_evaluate(args)
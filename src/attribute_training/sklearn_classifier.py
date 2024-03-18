from sklearn.linear_model import LogisticRegression
import pickle

from sklearn.neural_network import MLPClassifier
import os 
import sys 
import torch 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import os
import numpy as np
from tqdm import tqdm
import argparse 
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

def load_features_for_multi_class(args,saved_feature_dict):
    feature_dict = open_file(saved_feature_dict,use_json=False)
    features = feature_dict['features']
    labels = feature_dict['labels']
    return {'features':features,'labels':labels}
def load_features_for_class(args,saved_feature_dict,class_id,test=False):
    feature_dict = open_file(saved_feature_dict,use_json=False)
    features = feature_dict['features']
    print(class_id,'class id')
    print(len(feature_dict['labels']))
    print(feature_dict['labels'].shape)
    labels = feature_dict['labels'][:,class_id]
    if test:
        # only care about -1 and 1s 
        mask = labels!=0
        updated_labels = labels[mask]
        updated_labels[updated_labels==-1]=0
        updated_features = features[mask]
    else:
        updated_features = features 
        updated_labels = labels 
        updated_labels[updated_labels==-1] = 0
    
            
    print(len([a for a in labels if a==1]),f'num positive for test=={test}')
    return {'features':updated_features,'labels':updated_labels}

def train_classifier(args,train_feature_dict,initial_features=None):
    classifier = LogisticRegression(verbose=1,max_iter=args.iterations,n_jobs=1,multi_class='ovr',class_weight='balanced',C=.5)
    
    # if initial_features != None:
    #     classifier.coef_ = initial_features
    classifier.fit(train_feature_dict['features'],train_feature_dict['labels'])
    print(classifier.classes_)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path,f'classifier_{args.class_id}.pkl'),'wb') as f:
        pickle.dump(classifier,f)
def train_multi_class(args,train_feature_dict,initial_features=None):
    if args.mlp:
        classifier = MLPClassifier(hidden_layer_sizes=1000,verbose=True,random_state=1, max_iter=args.iterations)
    else:
        classifier = LogisticRegression(verbose=1,max_iter=args.iterations,n_jobs=1,multi_class='multinomial',solver='sag')
    classifier.fit(train_feature_dict['features'],train_feature_dict['labels'])
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path,f'classifier_{args.class_id}.pkl'),'wb') as f:
        pickle.dump(classifier,f)

def eval_multi_class(args,test_feature_dict):
    with open(os.path.join(args.save_path,f'classifier_{args.class_id}.pkl'),'rb') as f:
        classifier = pickle.load(f)
    predictions = classifier.predict(test_feature_dict['features'])
    accuracy = np.mean((test_feature_dict['labels'] == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")
    ap_score = average_precision_score(test_feature_dict['labels'],classifier.decision_function(test_feature_dict['features']))
    print(f'AP Scores (average):{ap_score}')
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    with open(os.path.join(args.results_dir,f'results_class_{args.class_id}.json'),'w+') as fwrite:
        json.dump({'acc':accuracy.item(),'ap_score':ap_score.item()},fwrite)
    
def eval_classifier(args,test_feature_dict):
    with open(os.path.join(args.save_path,f'classifier_{args.class_id}.pkl'),'rb') as f:
        classifier = pickle.load(f)
    classifier.coef_ = classifier.coef_.astype(np.float32)
    classifier.intercept_ = classifier.intercept_.astype(np.float32)
    print(test_feature_dict['labels'].shape,'shape')
    #print(np.nonzero(test_feature_dict['labels']).any())
    roc_auc = roc_auc_score(test_feature_dict['labels'],classifier.decision_function(test_feature_dict['features']))
    ap_score = average_precision_score(test_feature_dict['labels'],classifier.decision_function(test_feature_dict['features']))
    print(f'ROC AUC:{roc_auc} for class {args.class_id}')
    print(f'AP SCORE:{ap_score} for class {args.class_id}')

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    with open(os.path.join(args.results_dir,f'results_class_{args.class_id}.json'),'w+') as fwrite:
        json.dump({'roc_auc':roc_auc,'ap_score':ap_score},fwrite)
    
def train_and_evaluate(args):
    # saved_feature_dict,class_id,test=False
    id_to_attribute = open_file(os.path.join(args.data_dir,'id_to_class.json'))
    train_annotations = open_file(os.path.join(args.data_dir,'train.json'))
    print('Loading train features')
    attribute_name = id_to_attribute[str(args.class_id)]
    if args.multi_class:
        if not args.eval_only:
            train_feature_dict = load_features_for_multi_class(args,os.path.join(args.feature_dir,'train.pkl'))
            train_multi_class(args,train_feature_dict,initial_features=None)
        test_feature_dict = load_features_for_multi_class(args,os.path.join(args.feature_dir,'test.pkl'))
        eval_multi_class(args,test_feature_dict)
    else:
        if not args.eval_only:
            train_feature_dict = load_features_for_class(args,os.path.join(args.feature_dir,'train.pkl'),int(args.class_id))
            train_classifier(args,train_feature_dict)
        test_feature_dict = load_features_for_class(args,os.path.join(args.feature_dir,'test.pkl'),int(args.class_id),test=True)
        eval_classifier(args,test_feature_dict)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/dinov2_features")
    parser.add_argument("--data_dir", type=str, default='/scratch/bcgp/datasets')
    parser.add_argument("--class_id",type=int,help="value from 0 to 63",default=0)
    parser.add_argument("--feature_dir",type=str,default="/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/color_features_single_classes_early_layers")
    parser.add_argument("--results_dir",type=str,default="/scratch/bcgp/michal5/ecole_mo9_demo/src/attribute_training/dinov2_multi_class_features_single_class_early_layers_results/color")
    parser.add_argument('--iterations',
    type=int,
    default=1000,
    help='Number of iterations to run log regression')
    parser.add_argument('--multi_class',action='store_true',help='use multi class classifier or binary classifier')
    parser.add_argument('--mlp',action='store_true',help='use a mlp or linear layer')
    parser.add_argument('--eval_only',action='store_true')
    args = parser.parse_args()
    train_and_evaluate(args)
import os
import torch
from tqdm import tqdm 
import argparse
import numpy as np
import sklearn.covariance
from loguru import logger
from lib.data_loader import get_data_loader
from lib.training.common import test_acc
from lib.models.networks import get_model, get_tokenizer
from lib.inference.base import get_knn_score,  get_out_score
from lib.metrics import get_metrics
from lib.exp import get_num_labels
from torch.nn import CosineSimilarity
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import pandas as pd

# inter-layer pooling
def pooling_features(features, pooling='last'):
    num_layers = features.shape[0]
    if pooling == 'last':
        return features[-1, :, :]
    elif pooling == 'avg':
        return np.mean(features[1:], axis=0)
    else:
        raise NotImplementedError
    
def sample_estimator(features, labels):
    labels = labels.reshape(-1)
    num_classes = np.unique(labels).shape[0]
    #group_lasso = EmpiricalCovariance(assume_centered=False)
    #group_lasso =  MinCovDet(assume_centered=False, random_state=42, support_fraction=1.0)
    # ShurunkCovariance is more stable and robust where the condition number is large
    group_lasso = sklearn.covariance.ShrunkCovariance()
    sample_class_mean = []
    for c in range(num_classes):
        current_class_mean = np.mean(features[labels == c, :], axis=0)
        sample_class_mean.append(current_class_mean)
    X = [features[labels == c, :] - sample_class_mean[c] for c in range(num_classes)]
    X = np.concatenate(X, axis=0)
    group_lasso.fit(X)
    precision = group_lasso.precision_

    return sample_class_mean, precision


def load_and_pool_features(input_dir, dataset_patterns, token_pooling, layer_pooling):
    pooled_features_dict = {}

    for pattern in dataset_patterns:
        pooled_features_list = []
        i = 1
        while True:
            filepath = f"{input_dir}/{token_pooling}_{pattern}{i}.npy"
            if not os.path.exists(filepath):
                break
            features = np.load(filepath, allow_pickle=True)
            pooled_features = pooling_features(features, layer_pooling)
            pooled_features_list.append(pooled_features)
            i += 1
        
        pooled_features_dict[pattern] = pooled_features_list
    return pooled_features_dict


def average_scores(feature_sets, ind_train_features, knn):
    num_feature_sets = len(feature_sets)
    num_samples = feature_sets[0].shape[0]  
    knn = NearestNeighbors(n_neighbors=10, algorithm='brute')
    knn.fit(ind_train_features)   
    scores_sum = np.zeros(num_samples)

    for features in feature_sets:
        normalized_features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-10)
        scores = 1 - np.max(knn.kneighbors(normalized_features)[0], axis=1)
        scores_sum += scores
    average_scores = scores_sum / num_feature_sets
    return average_scores

def prepare_wiki_knn(model, wiki_loader):
        model.eval()
        hidden = []
        with torch.no_grad():
            for input_ids, labels, attention_masks in tqdm(wiki_loader):
                outputs = model(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True)
                hidden += outputs.hidden_states[-1][:, 0, :].cpu().numpy().tolist()
        hidden = np.array(hidden)
        hidden = hidden / \
            (np.linalg.norm(hidden, axis=-1, keepdims=True)+ 1e-10)
        knn = NearestNeighbors(n_neighbors=10, algorithm='brute')
        knn.fit(hidden)
        return knn
    
def wiki_average_scores(model,feature_sets, wiki_loader, knn):
    model.eval()
    knn_out = prepare_wiki_knn(model, wiki_loader)
    num_feature_sets = len(feature_sets)
    num_samples = feature_sets[0].shape[0]  
    scores_sum = np.zeros(num_samples)

    for features in feature_sets:
        normalized_features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-10)
        scores = 1 - np.max(knn_out.kneighbors(normalized_features)[0], axis=1)
        scores_sum += scores
    average_scores = scores_sum / num_feature_sets
    return average_scores
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ood_datasets',
                        type=str, required=False)
    parser.add_argument('--layer_pooling', type=str, default='last')
    parser.add_argument('--token_pooling', type=str, default='avg',
                        help='token pooling way', choices=['cls', 'avg', 'max'])
    parser.add_argument('--distance_metric', type=str, default='maha',
                        help='distance metric')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--base_dir',required=False,type=str)
    parser.add_argument('--seed',required=False,type=str)
    parser.add_argument('--input_dir', default='./log/embeddings/roberta-base/sst-2/seed13',
        type=str, required=False, help='save directory')
    parser.add_argument('--batch_size', default=32, type=int,
                        required=False, help='batch size')
    parser.add_argument('--model', default='roberta-base',
                        help='pretrained model type')
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the checkpoint to load')
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())
    
    num_labels = get_num_labels(args.dataset)
    args.num_labels = num_labels
    model = get_model(args)
    logger.info("{} model loaded".format(args.model))
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
        logger.info("model loaded from {}".format(args.pretrained_model))
    tokenizer = get_tokenizer(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("{} tokenizer loaded".format(args.model))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    input_dir = args.input_dir
    base_dir = args.base_dir
    token_pooling = args.token_pooling
    layer_pooling = args.layer_pooling
    ood_dataset = args.ood_datasets
    dataset_patterns = ['ood_combined', 'ood_oracles', 'oracle_combined', 'test_combined', 'oracle_combined_test','test_oracles']   

    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(base_dir, token_pooling))
    ind_train_labels = np.load(
        '{}/{}_ind_train_labels.npy'.format(base_dir, token_pooling))

    ind_train_features = pooling_features(ind_train_features, layer_pooling)
    ind_train_features = ind_train_features / \
            np.linalg.norm(ind_train_features, axis=-1, keepdims=True) + 1e-10
            

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(base_dir, token_pooling))
    ind_test_labels = np.load( '{}/{}_ind_test_labels.npy'.format(base_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features, layer_pooling)
    ind_test_features = ind_test_features / \
        np.linalg.norm(ind_test_features, axis=-1, keepdims=True) + 1e-10
        
    knn = NearestNeighbors(n_neighbors=10, algorithm='brute')
    knn.fit(ind_train_features)    
    
    ind_scores = 1 - np.max(knn.kneighbors(ind_test_features)[0], axis=1)  
    ood_features = np.load(
        '{}/{}_ood_features_{}.npy'.format(base_dir, token_pooling, ood_dataset))
    ood_pairs = np.load(
            '{}/{}_ood_pairs_{}.npy'.format(base_dir, token_pooling, ood_dataset),allow_pickle=True)
    ood_features = pooling_features(ood_features, layer_pooling)
    ood_features = ood_features / \
        np.linalg.norm(ood_features, axis=-1, keepdims=True) + 1e-10
    ood_scores = 1 - np.max(knn.kneighbors(ood_features)[0], axis=1)
    
    
    wiki_loader = get_data_loader('wiki', 'test', tokenizer, args.batch_size)
    knn_out = prepare_wiki_knn(model,wiki_loader)
    
    ood_features = np.load(
        '{}/{}_ood_features_{}.npy'.format(base_dir, token_pooling, ood_dataset))
    ood_features = pooling_features(ood_features, layer_pooling)
    ood_features = ood_features / \
        np.linalg.norm(ood_features, axis=-1, keepdims=True) + 1e-10 
    ood_scores = 1 - np.max(knn.kneighbors(ood_features)[0], axis=1)
    ood_scores_wiki = 1 - np.max(knn_out.kneighbors(ood_features)[0], axis=1)
    ood_scores = ood_scores - 0.5*ood_scores_wiki
    
    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(base_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features, layer_pooling)
    ind_test_features = ind_test_features / \
            np.linalg.norm(ind_test_features, axis=-1, keepdims=True) + 1e-10
    test_scores = 1 - np.max(knn.kneighbors(ind_test_features)[0], axis=1)
    test_scores_wiki =1 - np.max(knn_out.kneighbors(ind_test_features)[0], axis=1)
    test_scores = test_scores - 0.5*test_scores_wiki
    
    print("flats")
    metrics = get_metrics(test_scores, ood_scores)
    logger.info('ood dataset: {}'.format(ood_dataset))
    logger.info('metrics: {}'.format(metrics))
    
    dataset_patterns = ['ood_combined', 'ood_oracles', 'oracle_combined','test_combined', 'oracle_combined_test','test_oracles'] 
    pooled_features_dict = load_and_pool_features(input_dir, dataset_patterns, token_pooling, layer_pooling)
    
    ood_oracles = pooled_features_dict['ood_oracles']
    ood_oracles_scores2 = average_scores(ood_oracles,ind_train_features,knn)
    
    test_oracles = pooled_features_dict['test_oracles']
    test_oracles_scores2 = average_scores(test_oracles,ind_train_features,knn)

    ood_combined = pooled_features_dict['ood_combined']
    ood_combined_scores2 = average_scores(ood_combined,ind_train_features,knn)
    
    test_combined = pooled_features_dict['test_combined']
    test_combined_scores2 = average_scores(test_combined,ind_train_features,knn)

    oracle_combined = pooled_features_dict['oracle_combined']
    oracle_combined_scores2 =average_scores(oracle_combined,ind_train_features,knn)

    oracle_combined_test = pooled_features_dict['oracle_combined_test']
    oracle_combined_test_scores2 = average_scores(oracle_combined_test,ind_train_features,knn)

    
    ind_scores = - 0.7*test_oracles_scores2 + test_scores + 0.2*(test_combined_scores2 - oracle_combined_test_scores2)
    ood_scores = - 0.7*ood_oracles_scores2 + ood_scores + 0.2*(ood_combined_scores2 -  oracle_combined_scores2)
    metrics = get_metrics(ind_scores, ood_scores)    
    print("flats+CED score")
    logger.info('metrics: {}'.format(metrics))

        
if __name__ == '__main__':
    main()

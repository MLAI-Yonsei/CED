import os
import torch
import argparse
import numpy as np
import sklearn.covariance
from loguru import logger
from lib.metrics import get_metrics
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
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

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
    
    ind_scores = 1 - np.mean(knn.kneighbors(ind_test_features)[0], axis=1)  
    
    ood_features = np.load(
        '{}/{}_ood_features_{}.npy'.format(base_dir, token_pooling, ood_dataset))
    ood_pairs = np.load(
            '{}/{}_ood_pairs_{}.npy'.format(base_dir, token_pooling, ood_dataset),allow_pickle=True)
    ood_features = pooling_features(ood_features, layer_pooling)
    ood_features = ood_features / \
        np.linalg.norm(ood_features, axis=-1, keepdims=True) + 1e-10
    ood_scores = 1 - np.max(knn.kneighbors(ood_features)[0], axis=1)
    metrics = get_metrics(ind_scores, ood_scores)
    print("KNN score")
    logger.info('ood dataset: {}'.format(ood_dataset))
    logger.info('metrics: {}'.format(metrics))
    
    
    dataset_patterns = ['ood_combined', 'ood_oracles', 'oracle_combined','test_combined', 'oracle_combined_test','test_oracles'] 
    pooled_features_dict = load_and_pool_features(input_dir, dataset_patterns, token_pooling, layer_pooling)
    
    ood_features = np.load(
        '{}/{}_ood_features_{}.npy'.format(base_dir, token_pooling, ood_dataset))
    ood_features = pooling_features(ood_features, layer_pooling)
    ood_features = ood_features / \
        np.linalg.norm(ood_features, axis=-1, keepdims=True) + 1e-10
    ood_scores = 1 - np.max(knn.kneighbors(ood_features)[0], axis=1)
    
    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(base_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features, layer_pooling)
    ind_test_features = ind_test_features / \
            np.linalg.norm(ind_test_features, axis=-1, keepdims=True) + 1e-10
    test_scores = 1 - np.mean(knn.kneighbors(ind_test_features)[0], axis=1)
    
      
    ood_oracles_scores = average_scores(pooled_features_dict['ood_oracles'], ind_train_features, knn)
    test_oracles_scores = average_scores(pooled_features_dict['test_oracles'], ind_train_features, knn)
    ood_combined_scores = average_scores(pooled_features_dict['ood_combined'], ind_train_features, knn)
    test_combined_scores = average_scores(pooled_features_dict['test_combined'], ind_train_features, knn)
    oracle_combined_scores = average_scores(pooled_features_dict['oracle_combined'], ind_train_features, knn)
    oracle_combined_test_scores = average_scores(pooled_features_dict['oracle_combined_test'], ind_train_features, knn)

    ind_scores = - 0.7*test_oracles_scores + test_scores + 0.2*(test_combined_scores - oracle_combined_test_scores)
    ood_scores = - 0.7*ood_oracles_scores + ood_scores + 0.2*(ood_combined_scores -  oracle_combined_scores)
    metrics = get_metrics(ind_scores, ood_scores)    
    print("KNN+CED score")
    logger.info('metrics: {}'.format(metrics))
        
if __name__ == '__main__':
    main()

import os
import torch
import argparse
import numpy as np
import sklearn.covariance
from loguru import logger
from lib.metrics import get_metrics
from torch.nn import CosineSimilarity
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

def get_distance_score(class_mean, precision, features, measure='maha'):
    num_classes = len(class_mean)
    num_samples = len(features)
    class_mean = [torch.from_numpy(m).float() for m in class_mean]
    precision = torch.from_numpy(precision).float()
    features = torch.from_numpy(features).float()
    scores = []
    for c in range(num_classes):
        centered_features = features.data - class_mean[c]
        if measure == 'maha':
            score = -1.0 * \
                torch.mm(torch.mm(centered_features, precision),
                    centered_features.t()).diag()
        elif measure == 'euclid':
            score = -1.0*torch.mm(centered_features,
                centered_features.t()).diag()
        elif measure == 'cosine':
            score = torch.tensor([CosineSimilarity()(features[i].reshape(
                1, -1), class_mean[c].reshape(1, -1)) for i in range(num_samples)])
        else:
            raise ValueError("Unknown distance measure")
        scores.append(score.reshape(-1, 1))
    scores = torch.cat(scores, dim=1)  # num_samples, num_classes
    scores, _ = torch.max(scores, dim=1)  # num_samples
    scores = scores.cpu().numpy()
    return scores

def get_auroc(ind_train_features, ind_train_labels, ind_test_features, ood_test_features, layer_pooling):
    ind_train_features = pooling_features(ind_train_features, layer_pooling)
    sample_class_mean, precision = sample_estimator(
        ind_train_features, ind_train_labels)
    ind_test_features = pooling_features(ind_test_features, layer_pooling)
    ood_test_features = pooling_features(ood_test_features, layer_pooling)
    ind_scores = get_distance_score(
        sample_class_mean, precision, ind_test_features)
    ood_scores = get_distance_score(
        sample_class_mean, precision, ood_test_features)
    metrics = get_metrics(ind_scores, ood_scores)
    auroc = metrics['AUROC']
    return auroc

def get_closest_class_index(class_mean, precision, features, measure='maha'):
    num_classes = len(class_mean)
    class_mean = [torch.from_numpy(m).float() for m in class_mean]
    precision = torch.from_numpy(precision).float()
    features = torch.from_numpy(features).float()
    scores = []

    for c in range(num_classes):
        centered_features = features - class_mean[c]
        if measure == 'maha':
            score = -1.0 * torch.mm(torch.mm(centered_features, precision), centered_features.t()).diag()
        elif measure == 'euclid':
            score = -1.0 * torch.mm(centered_features, centered_features.t()).diag()
        else:
            raise ValueError("Unknown distance measure")
        scores.append(score.reshape(-1, 1))

    scores = torch.cat(scores, dim=1) 
    _, closest_class_indices = torch.max(scores, dim=1) 
    
    return closest_class_indices.cpu().numpy()

def find_closest_class_sample(ood_features, closest_class_indices, ind_train_features, ind_train_pairs, K):
    closest_idxs = []  
    train_labels = [pair[1] for pair in ind_train_pairs]  

    for i, ood_feature in enumerate(ood_features):
        class_idx = closest_class_indices[i]
        class_sample_idxs = np.where(np.array(train_labels) == class_idx)[0]
        class_samples = ind_train_features[class_sample_idxs]
        
        distances = np.linalg.norm(class_samples - ood_feature, axis=1)
        closest_inclass_idxs = np.argsort(distances)[:K]
        
        closest_idxs.append(class_sample_idxs[closest_inclass_idxs])
        
    return np.array(closest_idxs)

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

def class_distance_score(class_mean, precision, features, target_class_indices, measure='maha'):
    num_samples = len(features)
    class_mean = [torch.from_numpy(m).float() for m in class_mean]
    precision = torch.from_numpy(precision).float()
    features = torch.from_numpy(features).float()
    scores = torch.zeros(num_samples)

    for i in range(num_samples):
        c = target_class_indices[i]  
        centered_features = features[i] - class_mean[c] 
        if measure == 'maha':
            distance_squared = (-1) * torch.mm(torch.mm(centered_features.unsqueeze(0), precision), centered_features.unsqueeze(1))
            score = distance_squared.squeeze()
        scores[i] = score

    return scores.cpu().numpy()

def average_class_distance_scores(class_means, precision, feature_sets, target_class_indices, measure='maha'):
    num_features_sets = len(feature_sets)
    num_samples = len(feature_sets[0])  
    scores_sum = np.zeros(num_samples)
    
    for features in feature_sets:
        scores = class_distance_score(class_means, precision, features, target_class_indices, measure)
        scores_sum += scores
    average_scores = scores_sum / num_features_sets
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
    ind_train_pairs = np.load(
        '{}/{}_ind_train_pairs.npy'.format(base_dir, token_pooling), allow_pickle=True)

    ind_train_features = pooling_features(ind_train_features, layer_pooling)
    sample_class_mean, precision = sample_estimator(
        ind_train_features, ind_train_labels)

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(base_dir, token_pooling))
    ind_test_labels = np.load( '{}/{}_ind_test_labels.npy'.format(base_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features, layer_pooling)
    ind_scores = get_distance_score(
        sample_class_mean, precision, ind_test_features, args.distance_metric)
    
    ood_features = np.load(
        '{}/{}_ood_features_{}.npy'.format(base_dir, token_pooling, ood_dataset))
    ood_pairs = np.load(
            '{}/{}_ood_pairs_{}.npy'.format(base_dir, token_pooling, ood_dataset),allow_pickle=True)
    ood_features = pooling_features(ood_features, layer_pooling)
    ood_scores = get_distance_score(
        sample_class_mean, precision, ood_features, args.distance_metric)
    metrics = get_metrics(ind_scores, ood_scores)
    print("MD score")
    logger.info('ood dataset: {}'.format(ood_dataset))
    logger.info('metrics: {}'.format(metrics))
    
    measure = 'maha'
    closest_ood_indices = get_closest_class_index(sample_class_mean, precision, ood_features, measure)
    closest_test_indices = get_closest_class_index(sample_class_mean, precision, ind_test_features, measure)
    
    dataset_patterns = ['ood_combined', 'ood_oracles', 'oracle_combined','test_combined', 'oracle_combined_test','test_oracles'] 
    pooled_features_dict = load_and_pool_features(input_dir, dataset_patterns, token_pooling, layer_pooling)
    
    ood_scores = class_distance_score(sample_class_mean,precision, ood_features, closest_ood_indices,'maha')
    test_scores = class_distance_score(sample_class_mean,precision, ind_test_features, closest_test_indices,'maha')

    datasets_to_score = [
    ('ood_oracles', closest_ood_indices),
    ('test_oracles', closest_test_indices),
    ('ood_combined', closest_ood_indices),
    ('test_combined', closest_test_indices),
    ('oracle_combined', closest_ood_indices),
    ('oracle_combined_test', closest_test_indices)
    ]
    scores = {}
    for dataset_name, closest_indices in datasets_to_score:
        features = pooled_features_dict[dataset_name]
        scores[f"{dataset_name}_scores"] = average_class_distance_scores(sample_class_mean, precision, features, closest_indices, measure='maha')

    ood_oracles_scores = scores['ood_oracles_scores']
    test_oracles_scores = scores['test_oracles_scores']
    ood_combined_scores = scores['ood_combined_scores']
    test_combined_scores = scores['test_combined_scores']
    oracle_combined_scores = scores['oracle_combined_scores']
    oracle_combined_test_scores = scores['oracle_combined_test_scores']

    ind_scores = -0.7*test_oracles_scores + test_scores + 0.2*(test_combined_scores - oracle_combined_test_scores )
    ood_scores = -0.7*ood_oracles_scores + ood_scores + 0.2*(ood_combined_scores - oracle_combined_scores )
    metrics = get_metrics(ind_scores, ood_scores)    
    print("MD+CED score")
    logger.info('metrics: {}'.format(metrics))

        
if __name__ == '__main__':
    main()

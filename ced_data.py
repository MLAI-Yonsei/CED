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
        else:
            raise ValueError("Unknown distance measure")
        scores.append(score.reshape(-1, 1))
    scores = torch.cat(scores, dim=1)
    return torch.max(scores, dim=1).values.cpu().numpy()


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

def select_auxiliary_samples(ood_features, oracle_features, ind_train_features, bound=30, M=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ood_features = torch.tensor(ood_features, dtype=torch.float, device=device)
    oracle_features = torch.tensor(oracle_features, dtype=torch.float, device=device)
    ind_train_features = torch.tensor(ind_train_features, dtype=torch.float, device=device)
    
    selected_indices = []
    for ood_feature, oracle_feature in zip(ood_features, oracle_features):
        direction_vector = ood_feature - oracle_feature
        diff_vectors = ood_feature.unsqueeze(0) - ind_train_features
        similarities = F.cosine_similarity(diff_vectors, direction_vector.unsqueeze(0), dim=1)
        _, least_similar_idxs = torch.topk(similarities, bound, largest=False)
        
        distances = torch.norm(diff_vectors[least_similar_idxs], dim=1, p=2)
        sorted_idxs = torch.argsort(distances, descending=True)
        selected_idxs = least_similar_idxs[sorted_idxs][:M]
        selected_indices.append(selected_idxs.cpu().tolist())

    return np.array(selected_indices)

def oracle_combined_datasets(ood_oracles, selected_auxiliary):
    combined_datasets = []
    oracle_datasets=[]

    for i in range(ood_oracles.shape[0]):
        oracle_set = []
        combined_set = []
        for j in range(ood_oracles[i].shape[0]):
            oracle_text, oracle_label = ood_oracles[i, j]
            auxiliaries = selected_auxiliary[j, i]  

            combined_examples = []
            oracle_examples = []
            for auxiliary_text, _ in auxiliaries:
                combined_sentence = f"{oracle_text}. {auxiliary_text}"
                combined_examples.append([combined_sentence, oracle_label])
                oracle_examples.append([oracle_text, oracle_label])
              
            combined_set.append(combined_examples)  
            oracle_set.append(oracle_examples)
        combined_datasets.append(combined_set)  
        oracle_datasets.append(oracle_set)

    return combined_datasets, oracle_datasets

def ood_combined_datasets(ood_pairs, selected_auxiliary):
    new_data = []

    for i in range(len(ood_pairs)):
        oracle_text, oracle_label = ood_pairs[i][0], ood_pairs[i][1]
        combined_datasets_for_oracle = []  
        for group_index in range(selected_auxiliary.shape[0]):
            auxiliaries_for_oracle = selected_auxiliary[group_index, i]  

            for auxiliary in auxiliaries_for_oracle:
                auxiliary_text, _ = auxiliary
                combined_sentence = f"{oracle_text}. {auxiliary_text}"
                combined_datasets_for_oracle.append([combined_sentence, oracle_label])

        new_data.append(combined_datasets_for_oracle)
    return new_data


def extract_dataset(list_of_lists):
    num_datasets = len(list_of_lists[0])
    datasets = {f"{i+1}": [item[i] for item in list_of_lists] for i in range(num_datasets)}
    return datasets

def extract_dataset_flat(list_of_lists):
    num_datasets = len(list_of_lists[0][0]) * len(list_of_lists[0])
    flat_list = [item for sublist in list_of_lists for inner_list in sublist for item in inner_list]
    
    datasets = {f"{i+1}": [flat_list[j] for j in range(i, len(flat_list), num_datasets)] for i in range(num_datasets)}
    return datasets

def save_datasets(datasets_dict, path,prefix):
    if not os.path.exists(path):
        os.makedirs(path)
        
    for key, dataset in datasets_dict.items():
        file_name = f"{path}/{prefix}{key}.npy"
        np.save(file_name, dataset)
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ood_datasets',
                        type=str, required=False)
    parser.add_argument('--layer_pooling', type=str, default='last')
    parser.add_argument('--token_pooling', type=str, default='cls',
                        help='token pooling way', choices=['cls', 'avg', 'max'])
    parser.add_argument('--distance_metric', type=str, default='maha',
                        help='distance metric')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--input_dir', default='./log/embeddings/roberta-base/sst-2/seed13',
        type=str, required=False, help='save directory')
    parser.add_argument('--output_dir',type=str,required=False, help='save directory')
    parser.add_argument('--oracle_num',type=int, default='5')
    parser.add_argument('--auxiliary_num',type=int, default='3')
    parser.add_argument('--bound',type=int, default='30')
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    input_dir = args.input_dir
    token_pooling = args.token_pooling
    layer_pooling = args.layer_pooling
    ood_dataset = args.ood_datasets

    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
    ind_train_labels = np.load(
        '{}/{}_ind_train_labels.npy'.format(input_dir, token_pooling))
    ind_train_pairs = np.load(
        '{}/{}_ind_train_pairs.npy'.format(input_dir, token_pooling), allow_pickle=True)

    ind_train_features = pooling_features(ind_train_features, layer_pooling)
    sample_class_mean, precision = sample_estimator(
        ind_train_features, ind_train_labels)

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_pairs = np.load( '{}/{}_ind_test_pairs.npy'.format(input_dir, token_pooling),allow_pickle=True)
    ind_test_features = pooling_features(ind_test_features, layer_pooling)
    
    ood_features = np.load(
        '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
    ood_pairs = np.load(
            '{}/{}_ood_pairs_{}.npy'.format(input_dir, token_pooling, ood_dataset),allow_pickle=True)
    ood_features = pooling_features(ood_features, layer_pooling)
    
    measure = 'maha'
    K = args.oracle_num
    M = args.auxiliary_num
    bound = args.bound
    
    closest_ood_indices = get_closest_class_index(sample_class_mean, precision, ood_features, measure)
    closest_test_indices = get_closest_class_index(sample_class_mean, precision, ind_test_features, measure)
    
    ood_indices = find_closest_class_sample(ood_features, closest_ood_indices, ind_train_features, ind_train_pairs, K)
    test_indices = find_closest_class_sample(ind_test_features, closest_test_indices, ind_train_features, ind_train_pairs, K)

    ood_oracles = ind_train_pairs[ood_indices]
    test_oracles = ind_train_pairs[test_indices]

    # selected_auxiliary_ood_indices = []
    # selected_auxiliary_test_indices = []

    # for col_idx in range(ood_indices.shape[1]): 
    #     current_index_set = ood_indices[:, col_idx] 
    #     close_ood_oracle_feature = ind_train_features[current_index_set]
    #     auxiliary_indices = select_auxiliary_samples(ood_features, close_ood_oracle_feature, ind_train_features, bound, M)
    #     selected_auxiliary_ood_indices.append(auxiliary_indices)
        
    # for col_idx in range(test_indices.shape[1]):
    #     current_index_set = test_indices[:, col_idx]
    #     close_test_oracle_feature = ind_train_features[current_index_set]
    #     auxiliary_indices = select_auxiliary_samples(ind_test_features, close_test_oracle_feature, ind_train_features, bound,M)
    #     selected_auxiliary_test_indices.append(auxiliary_indices)

    # selected_auxiliary_ood = ind_train_pairs[selected_auxiliary_ood_indices]
    # selected_auxiliary_test = ind_train_pairs[selected_auxiliary_test_indices]
    
    selected_auxiliary_ood = [
        select_auxiliary_samples(ood_features, ind_train_features[ood_indices[:, col_idx]], ind_train_features, args.bound, args.auxiliary_num)
        for col_idx in range(ood_indices.shape[1])
    ]
    
    selected_auxiliary_test = [
        select_auxiliary_samples(ind_test_features, ind_train_features[test_indices[:, col_idx]], ind_train_features, args.bound, args.auxiliary_num)
        for col_idx in range(test_indices.shape[1])
    ]

    selected_auxiliary_ood = ind_train_pairs[selected_auxiliary_ood]
    selected_auxiliary_test = ind_train_pairs[selected_auxiliary_test]
    
    
    oracle_combined ,ood_oracles = oracle_combined_datasets(ood_oracles, selected_auxiliary_ood)
    oracle_combined_test,test_oracles = oracle_combined_datasets(test_oracles, selected_auxiliary_test)
    ood_combined = ood_combined_datasets(ood_pairs, selected_auxiliary_ood)
    test_combined = ood_combined_datasets(ind_test_pairs, selected_auxiliary_test)
    

    test_combined = extract_dataset(test_combined) 
    ood_combined = extract_dataset(ood_combined)     
    ood_oracles = extract_dataset_flat(ood_oracles)
    test_oracles = extract_dataset_flat(test_oracles)
    oracle_combined = extract_dataset_flat(oracle_combined) 
    oracle_combined_test = extract_dataset_flat(oracle_combined_test)

    save_datasets(ood_oracles, args.output_dir,'ood_oracles')
    save_datasets(test_oracles, args.output_dir, 'test_oracles')
    save_datasets(ood_combined, args.output_dir,'ood_combined')
    save_datasets(test_combined, args.output_dir,'test_combined')
    save_datasets(oracle_combined, args.output_dir,'oracle_combined')
    save_datasets(oracle_combined_test, args.output_dir,'oracle_combined_test')

if __name__ == '__main__':
    main()

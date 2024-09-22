import os
import torch
import argparse
import numpy as np
from loguru import logger

from lib.data_loader import get_raw_data , data_loader, get_data_loader 
from lib.models.networks import get_model, get_tokenizer
from lib.inference.draw import get_full_features
from lib.exp import get_num_labels
from transformers import AutoModelForSequenceClassification

def create_dataloaders(output_dir, dataset_patterns, tokenizer, batch_size):
    data_loaders = {}

    for pattern in dataset_patterns:
        i = 1
        while True:
            filepath = f'{output_dir}/{pattern}{i}.npy'
            if not os.path.exists(filepath):
                break
            data = np.load(filepath, allow_pickle=True)
            texts = [item[0] for item in data]
            labels = [int(item[1]) for item in data]

            loader = data_loader(texts, labels, 'test', tokenizer, batch_size)
            data_loaders[f"{pattern}{i}"] = loader
            i += 1

    return data_loaders
            

def extract_and_save_features_for_all_datasets(data_loaders, model, pooling_types, output_dir, feature_pos):
    for key, loader in data_loaders.items():
        for p in pooling_types:
            logger.info(f'token pooling: {p}')
            
            features, _, _ = get_full_features(model, loader, p, pos=feature_pos)
            feature_file = f'{output_dir}/features/{p}_{key}.npy'
            os.makedirs(os.path.dirname(feature_file), exist_ok=True)  
            np.save(feature_file, features)
            
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ood_datasets',
                        type=str, required=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        required=False, help='batch size')
    parser.add_argument(
        '--model', default='roberta-base', help='pretrained model type')
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the model to load')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--output_dir', default='saved_model/',
                        type=str, required=False, help='save directory')
    parser.add_argument('--feature_pos', type=str,
                        default='after', help='feature position')
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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

    dataset_patterns = ['ood_combined', 'ood_oracles', 'oracle_combined', 'test_combined', 'oracle_combined_test','test_oracles'] 
    data_loaders = create_dataloaders(output_dir, dataset_patterns, tokenizer, args.batch_size)

    
    pooling = ['cls','avg']
    with torch.no_grad():
        for p in pooling:
            extract_and_save_features_for_all_datasets(data_loaders, model, pooling, output_dir, args.feature_pos)
        

if __name__ == '__main__':
    main()

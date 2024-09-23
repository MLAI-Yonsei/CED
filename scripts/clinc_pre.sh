CUDA_VISIBLE_DEVICES=0 python extract_full_features.py \
    --dataset clinc \
    --ood_datasets clinc_ood \
    --output_dir output/clinc/pre \
    --model roberta-base 

CUDA_VISIBLE_DEVICES=0 python ced_data.py \
    --dataset clinc \
    --ood_datasets clinc_ood \
    --input_dir output/clinc/pre \
    --output_dir output/clinc/pre/CED\
    --oracle_num 3\
    --auxiliary_num 5\
    --bound 100\
    --layer_pooling last \
    --token_pooling cls

CUDA_VISIBLE_DEVICES=0 python ced_features.py \
    --dataset clinc \
    --ood_datasets clinc_ood \
    --output_dir output/clinc/pre/CED \
    --model roberta-base 

CUDA_VISIBLE_DEVICES=0 python md_eval.py \
    --dataset clinc \
    --ood_datasets clinc_ood \
    --base_dir output/clinc/pre \
    --input_dir output/clinc/pre/CED/features \
    --layer_pooling last \
    --token_pooling cls \
    --seed pre
    
CUDA_VISIBLE_DEVICES=0 python knn_eval.py \
        --dataset clinc \
        --ood_datasets clinc_ood \
        --base_dir output/clinc/pre \
        --input_dir output/clinc/pre/CED/features \
        --layer_pooling last \
        --token_pooling cls \
        --seed pre

CUDA_VISIBLE_DEVICES=0 python flats_eval.py \
    --dataset clinc \
    --ood_datasets clinc_ood \
    --base_dir output/clinc/pre \
    --input_dir output/clinc/pre/CED/features \
    --layer_pooling last \
    --token_pooling cls \
    --model roberta-base \
    --seed pre

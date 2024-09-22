CUDA_VISIBLE_DEVICES=0 python extract_full_features.py \
    --dataset snips \
    --ood_datasets snips_ood \
    --output_dir output/snips/pre \
    --model roberta-base 

CUDA_VISIBLE_DEVICES=0 python ced_data.py \
    --dataset snips \
    --ood_datasets snips_ood \
    --input_dir output/snips/pre \
    --output_dir output/snips/pre/CED\
    --oracle_num 3\
    --auxiliary_num 5\
    --bound 100\
    --layer_pooling last \
    --token_pooling cls

CUDA_VISIBLE_DEVICES=0 python ced_features.py \
    --dataset snips \
    --ood_datasets snips_ood \
    --output_dir output/snips/pre/CED \
    --model roberta-base 


CUDA_VISIBLE_DEVICES=0 python md_eval.py \
    --dataset snips \
    --ood_datasets snips_ood \
    --base_dir output/snips/pre \
    --input_dir output/snips/pre/CED/features \
    --layer_pooling last \
    --token_pooling cls \
    --seed pre

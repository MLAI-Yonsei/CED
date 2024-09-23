for seed in 0
do 
    CUDA_VISIBLE_DEVICES=0 python ced_data.py \
        --dataset clinc \
        --ood_datasets clinc_ood \
        --input_dir output/clinc/$seed \
        --output_dir output/clinc/$seed/CED\
        --oracle_num 3\
        --auxiliary_num 5\
        --bound 100\
        --layer_pooling avg \
        --token_pooling avg 

    CUDA_VISIBLE_DEVICES=0 python ced_features.py \
        --dataset clinc \
        --ood_datasets clinc_ood \
        --output_dir output/clinc/$seed/CED \
        --model roberta-base \
        --pretrained_model output/clinc/$seed/model.pt

    CUDA_VISIBLE_DEVICES=0 python knn_eval.py \
        --dataset clinc \
        --ood_datasets clinc_ood \
        --base_dir output/clinc/$seed \
        --input_dir output/clinc/$seed/CED/features \
        --layer_pooling avg \
        --token_pooling avg \
        --seed pre
done
for seed in  0
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --model roberta-base \
        --output_dir ./output/clinc/$seed \
        --seed $seed \
        --dataset clinc \
        --log_file ./log/clinc/exp_$seed.txt \
        --lr 2e-5 \
        --epochs 5 \
        --batch_size 16

    CUDA_VISIBLE_DEVICES=0 python extract_full_features.py \
        --dataset clinc \
        --ood_datasets clinc_ood \
        --output_dir output/clinc/$seed \
        --model roberta-base \
        --pretrained_model output/clinc/$seed/model.pt
        
done



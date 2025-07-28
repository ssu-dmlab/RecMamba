# Use distributed data parallel
#CUDA_VISIBLE_DEVICES=1
python lightning_pre_mamba.py \
    --model_name_or_path bert-base-uncased \
    --train_file pretrain_data/train.json \
    --dev_file pretrain_data/dev.json \
    --item_attr_file pretrain_data/meta_data.json \
    --output_dir result/recmamba_pretraining \
    --num_train_epochs 32 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8  \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --temp 0.05 \
    --device 1 \
    --fp16 \
    --fix_word_embedding
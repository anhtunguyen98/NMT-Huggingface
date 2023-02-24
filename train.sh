python -m torch.distributed.launch --nproc_per_node=3 \
        train.py --model_checkpoint facebook/mbart-large-50\
                --batch_size 64 \
                --learning_rate 3e-5 \
                --num_train_epochs 30 \
                --train_fold data/train \
                --val_fold data/val \
                --test_fold data/test \
                --source_lang vi \
                --target_lang en \
                --src_lang vi_VN \
                --tgt_lang en_US \
                --max_input_length 100 \
                --max_target_length 100 \

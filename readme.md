# Neural Machine Translation Pipeline



## Format data
{ 'vi': "xin chào mọi người", 'en': "Hello everyone"}

## Train model


```shell
python -m torch.distributed.launch --nproc_per_node=4 \
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
```

## Training Distillation-aware quantization
Reimplementation of [DQ-BART:  Efficient Sequence-to-Sequence Model via Joint Distillation and Quantization](https://arxiv.org/pdf/2203.11239.pdf) for machine translation

```
python -m torch.distributed.launch --nproc_per_node=2 \
      train_distill.py --teacher_model facebook/mbart-large-50-many-to-many-mmt \
                --batch_size 16 \
                --learning_rate 3e-5 \
                --num_train_epochs 30 \
                --train_fold data/train \
                --val_fold data/dev \
                --test_fold data/test \
                --source_lang vi \
                --target_lang en \
                --src_lang vi_VN \
                --tgt_lang en_US \
                --max_input_length 100 \
                --max_target_length 100 \
                --input_bits 32 \
                --weight_bits 8 \
                --distill_encoder 3 \
                --distill_decoder 1 \
                --task_weight 1 \
                --logits_weight 1 \
                --hid_weight 1 \
```



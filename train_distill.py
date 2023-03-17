from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import concatenate_datasets
from utils.dataloader import preprocess_function
from utils.metrics import compute_metrics
from distill_trainer import QTrainer
import datasets
import glob
import argparse
from quant.modeling_mbart_quant import MBartForConditionalGeneration
from quant.configuration_mbart_quant import MBartConfig
import os


distill_mappings = {1: {0: 5},
                    2: {0: 0, 1: 5},
                    3: {0: 0, 1: 2, 2: 5},
                    4: {0: 0, 1: 2, 2: 3, 3: 5},
                    5: {0: 0, 1: 1, 2: 3, 3: 4, 4: 5},
                    6: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                    }
distill_mappings_new = {1: {0: 0}}
NUMS = [str(i) for i in range(6)]


def load_data(folder):
    dataset = []

    for part in glob.glob(folder+"/*.json"):
            data = datasets.load_dataset('json', data_files=part)['train']
            dataset.append(data)
    return concatenate_datasets(dataset)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_model')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_train_epochs', type=float)
    parser.add_argument('--input_bits', type=int)
    parser.add_argument('--weight_bits', type=int)
    parser.add_argument('--distill_encoder', type=int)
    parser.add_argument('--distill_decoder', type=int)
    parser.add_argument("--clip_val",
                        default=2.5,
                        type=float,
                        help="Initial clip value.")
    parser.add_argument('--task_weight', type=int)
    parser.add_argument('--logits_weight', type=int)
    parser.add_argument('--hid_weight', type=int)
    
    
    parser.add_argument('--train_fold', type=str)
    parser.add_argument('--val_fold', type=str)
    parser.add_argument('--test_fold', type=str)
    
    parser.add_argument('--source_lang', type=str )
    parser.add_argument('--target_lang', type=str)
    parser.add_argument('--src_lang', type=str)
    parser.add_argument('--tgt_lang', type=str)
     
    parser.add_argument('--max_input_length', type=int)
    parser.add_argument('--max_target_length', type=int)
    
    args,_ = parser.parse_known_args()
    
    
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang
    
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model)
    
    
    student_config = MBartConfig.from_pretrained(args.teacher_model,
                                                quantize_act=True,
                                                weight_bits=args.weight_bits,
                                                input_bits=args.input_bits,
                                                clip_val=args.clip_val,
                                                decoder_layers=args.distill_decoder,
                                                encoder_layers=args.distill_encoder)
    
    student_model = MBartForConditionalGeneration(student_config)
    model_name = args.teacher_model.split("/")[-1]

    distill_enc_mapping = distill_mappings[args.distill_encoder]
    distill_dec_mapping = distill_mappings[args.distill_decoder]
    maps = {'enc': distill_enc_mapping, 'dec': distill_dec_mapping}
    
    dst_dict = student_model.state_dict()  # Initilized student model state dict, needs loading weights
    src_dict = teacher_model.state_dict()  # Pretrained teacher model state dict, whose weights will be loaded

    for key in dst_dict.keys():
        if ("encoder" in key or "decoder" in key) and key[
            21] in NUMS:  # Determine if the key belongs to a encoder/decoder layer,
            # which starts with sth like model.decoder.layers.1

            m = maps[key[6:9]]  # Determin if it is an encoder or decoder, and get the layer mapping
            old_idx = int(key[21])  # The layer index of the student model that needs loading
            new_idx = str(m[old_idx])  # The layer index of the teacher model that should be loaded
            mapped_key = key[:21] + new_idx + key[22:]  # Get the full teacher layer key
            if mapped_key in src_dict.keys():  # Exclude the cases
                # which does not exist in the teacher model
                dst_dict[key] = src_dict[mapped_key]  # Load the weights of the layer
        else:
            if key in src_dict.keys():  # Load the weights of non-encoder/decoder layers
                dst_dict[key] = src_dict[key]

    student_model.load_state_dict(dst_dict, strict=False)  # Pass the dict to the student model


    train_args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{args.source_lang}-to-{args.target_lang}",
        evaluation_strategy="steps",
        metric_for_best_model='bleu',
        save_strategy = "steps",
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        eval_steps=5000,
        save_steps=5000,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        # fp16=True,
        push_to_hub=False,
        gradient_accumulation_steps=1,
    )
    

    
    
    num_proc = 10
    train_dataset = load_data(args.train_fold)
    valid_dataset = load_data(args.val_fold)
    test_dataset = load_data(args.test_fold)
    
    train_dataset = train_dataset.map(lambda x: preprocess_function(x,tokenizer, args),batched=True, num_proc=num_proc)
    valid_dataset = valid_dataset.map(lambda x: preprocess_function(x,tokenizer, args),batched=True, num_proc=num_proc)
    test_dataset = test_dataset.map(lambda x: preprocess_function(x,tokenizer, args),batched=True, num_proc=num_proc)
    
    print("Train set: ", len(train_dataset))
    print("Valid set: ", len(valid_dataset))
    print("Test set: ", len(test_dataset))
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=student_model)
    
    trainer = QTrainer(
        model=student_model,
        args=train_args,
        teacher_model=teacher_model,
        distill_enc_mapping=distill_enc_mapping,
        distill_dec_mapping=distill_dec_mapping,
        task_weight = args.task_weight,
        logits_weight = args.logits_weight,
        hid_weight = args.hid_weight,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics= lambda x: compute_metrics(x, tokenizer)
    )
    
    trainer.train()
    
    print(trainer.evaluate(test_dataset))
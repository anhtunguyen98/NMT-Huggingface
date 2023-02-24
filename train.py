from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import concatenate_datasets
from utils.dataloader import preprocess_function
from utils.metrics import compute_metrics
import datasets
import glob
import argparse





def load_data(folder):
    dataset = []

    for part in glob.glob(folder+"/*.json"):
            data = datasets.load_dataset('json', data_files=part)['train']
            dataset.append(data)
    return concatenate_datasets(dataset)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_train_epochs', type=float)
    
    
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
    
    
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    
    
    
    model_name = args.model_checkpoint.split("/")[-1]
    
    train_args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{args.source_lang}-to-{args.target_lang}",
        evaluation_strategy="steps",
        metric_for_best_model='bleu',
        save_strategy = "steps",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        eval_steps=5000,
        save_steps=5000,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        gradient_accumulation_steps=8,
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
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics= lambda x: compute_metrics(x, tokenizer)
    )
    
    trainer.train()
    
    print(trainer.evaluate(test_dataset))
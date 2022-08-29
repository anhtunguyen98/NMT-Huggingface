import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils import metrics, dataloader
from utils.dataloader import preprocess_function
from datasets import concatenate_datasets
import datasets
import glob


def load_data(folder):
    dataset = []

    for part in glob.glob(folder+"/*.json"):
            data = datasets.load_dataset('json', data_files=part)['train']
            dataset.append(data)
    return concatenate_datasets(dataset)



if __name__ == '__main__':
    model_checkpoint = "facebook/mbart-large-50"


    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    source_lang = "ko"
    target_lang = "vi"

    batch_size = 2
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
        evaluation_strategy="steps",
        metric_for_best_model='bleu',
        save_strategy = "steps",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        eval_steps=5000,
        save_steps=5000,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        gradient_accumulation_steps=8,
    )

    num_proc = 10
    train_dataset = load_data('data/train')
    valid_dataset = load_data('data/val')
    test_dataset = load_data('data/test')
    
    train_dataset = train_dataset.map(preprocess_function,batched=True, num_proc=num_proc)
    valid_dataset = valid_dataset.map(preprocess_function,batched=True, num_proc=num_proc)
    test_dataset = test_dataset.map(preprocess_function,batched=True, num_proc=num_proc)
    # print(test_dataset)

    print("Train set: ", len(train_dataset))
    print("Valid set: ", len(valid_dataset))
    print("Test set: ", len(test_dataset))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metrics.compute_metrics
    )

    trainer.train()

    trainer.evaluate(test_dataset)
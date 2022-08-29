from transformers import AutoTokenizer
import datasets


model_checkpoint = "facebook/mbart-large-50"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
source_lang = "ko"
target_lang = "vi"
tokenizer.src_lang = "ko_KR"
tokenizer.tgt_lang = "vi_VN"

max_input_length = 100
max_target_length = 100


def preprocess_function(examples):

    inputs = [example for example in examples[source_lang] ]
    targets = [example for example in examples[target_lang] ]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




def get_dataloader(train_path, valid_path, test_path, batch_size=2, num_proc=10):
    train_set = datasets.load_from_disk(train_path)
    valid_set = datasets.load_from_disk(valid_path)
    test_set = datasets.load_from_disk(test_path)

    return train_set, valid_set, test_set


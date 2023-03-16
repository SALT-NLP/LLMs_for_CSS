from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import argparse
import os
import torch
import pandas as pd
import nltk
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

parser = argparse.ArgumentParser("")
parser.add_argument("--model", type=str, default="t5-base")
parser.add_argument("--data", type=str, default="mrf-explain")
parser.add_argument("--gpu",
                    default="0",
                    type=str,
                    help="id(s) for CUDA_VISIBLE_DEVICES")
parser.add_argument("--bz", type=int, default=4)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"


class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings_1, encodings_2):
        self.encodings = encodings_1
        self.labels = encodings_2

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


nltk.download('punkt')
rouge = evaluate.load("rouge")
prefix = "Generate the implied message of the following news headline: "
all_file = "..\css_data\mrf\\mrf-explanation.csv"
test_file = "..\css_data\mrf\\test-explanation.json"
all_df = pd.read_csv(all_file, delimiter=',')
headlines = []
intend = []
max_len_1 = 0
max_len_2 = 0
max_input_length = 1024
max_target_length = 1024
for idx, row in all_df.iterrows():
    headlines.append(prefix + row.values[0])
    intend.append(row.values[1].strip('[').strip(']'))

tokenizer = AutoTokenizer.from_pretrained(args.model,
                                          model_max_length=max_target_length)
model_inputs = tokenizer(headlines,
                         max_length=max_input_length,
                         truncation=True)
with tokenizer.as_target_tokenizer():
    labels = tokenizer(intend, max_length=max_target_length, truncation=True)
all_dataset = Custom_Dataset(model_inputs, labels["input_ids"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions,
                                           skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip()))
        for label in decoded_labels
    ]

    result = rouge.compute(predictions=decoded_preds,
                           references=decoded_labels,
                           use_stemmer=True)

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id)
        for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
batch_size = 5
args = Seq2SeqTrainingArguments(args.data + args.model + "_output",
                                evaluation_strategy="epoch",
                                learning_rate=2e-5,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                weight_decay=0.01,
                                save_total_limit=3,
                                num_train_epochs=1,
                                predict_with_generate=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(model,
                         args,
                         train_dataset=all_dataset,
                         eval_dataset=all_dataset,
                         data_collator=data_collator,
                         tokenizer=tokenizer,
                         compute_metrics=compute_metrics)

trainer.train()
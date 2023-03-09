# baseline for classification tasks
# Zhehao Zhang
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import json
import torch
import random
import pandas as pd
import numpy as np
import argparse
import os
import wandb
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=60)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--bz", type=int, default=16)
parser.add_argument("--data", type=str, default="change")
parser.add_argument("--model", type=str, default="roberta-large")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--norm", type=float, default=1)
parser.add_argument("--gpu",
                    default="0",
                    type=str,
                    help="id(s) for CUDA_VISIBLE_DEVICES")
args = parser.parse_args()
wandb.init(project="social_chatgpt")
wandb.config = {
    "seed": args.seed,
    "lr": args.lr,
    "epoch": args.epoch,
    "data": args.data,
}
wandb.init(config=args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(args.seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "acc": accuracy_score(labels, predictions),
        "macro-F1:": f1_score(labels, predictions, average="macro"),
    }


class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def dataset_process(text_file, label_file, tokenizer):
    texts = []
    labels = []
    train_instances = load_instances(text_file)
    label_df = pd.read_csv(label_file, delimiter='\t', header=None)
    label_dic = {}
    for idx, row in label_df.iterrows():
        label_dic[row.values[0]] = row.values[1]
    for inst_idx, inst in enumerate(train_instances):
        idx = inst["id"]
        texts.append(inst['tweet1']['text'] + '</s>' + inst['tweet2']['text'])
        labels.append(label_dic[idx])
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = Custom_Dataset(encodings, labels)
    return dataset


def load_instances(fn):

    instances = []
    with open(fn, 'r', encoding='utf-8') as f:
        for jl_str in f:
            instances.append(json.loads(jl_str))

    return instances


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2).to(device)
    if args.data == "change":
        train_dataset = dataset_process(".\TempoWic\\train.data.jl",
                                        ".\TempoWic\\train.labels.tsv",
                                        tokenizer)
        dev_dataset = dataset_process(".\TempoWic\\validation.data.jl",
                                      ".\TempoWic\\validation.labels.tsv",
                                      tokenizer)
        Test_file = "..\css_data\\tempowic\\test.json"
        f = open(Test_file, "r")
        content = f.read()
        test_data = json.loads(content)
        test_text = []
        test_label = []
        #same format?
        for index in test_data["labels"].keys():
            test_text.append(test_data['context'][index])
            if test_data["labels"][index] == "Different":
                test_label.append(0)
            elif test_data["labels"][index] == "Same":
                test_label.append(1)
        encodings = tokenizer(test_text, truncation=True, padding=True)
        test_dataset = Custom_Dataset(encodings, test_label)

    training_args = TrainingArguments(
        output_dir=args.data + args.model + "_output",
        evaluation_strategy="steps",
        eval_steps=int(len(train_dataset) * args.epoch / args.bz / 10),
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bz,
        per_device_eval_batch_size=args.bz * 20,
        save_strategy="no",
    )

    # optimizer hyperparameter to be tuned
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # change to dev data
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_output = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print(test_output)


# optimizer hyperparameter to be tuned

if __name__ == "__main__":

    main()

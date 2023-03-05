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
'''
bert-base-uncased
lr: 2e-5
bz: 256/128

'''
parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=60)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--bz", type=int, default=32)
parser.add_argument("--data", type=str, default="humor")
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


def dataset_process(df, tokenizer):
    texts = []
    labels = []
    for idx, row in df.iterrows():
        texts.append(row.values[3])
        labels.append(int(row.values[1]))
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    dataset = Custom_Dataset(encodings, labels)
    return dataset


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2).to(device)

    if args.data == "humor":
        train_df = pd.read_csv(".\\reddit_humor\\train.tsv", delimiter=',')
        # To Do: change another test datasets
        test_df = pd.read_csv(".\\reddit_humor\\test.tsv", delimiter=',')
        dev_df = pd.read_csv(".\\reddit_humor\\dev.csv", delimiter=',')
        train_dataset = dataset_process(train_df, tokenizer)
        #test_dataset = dataset_process(test_df, tokenizer)
        dev_dataset = dataset_process(dev_df, tokenizer)
        Test_file = "..\css_data\\reddit_humor\\test.json"
        f = open(Test_file, "r")
        content = f.read()
        test_data = json.loads(content)
        length = len(test_data["labels"])
        test_text = [test_data["context"][str(i)] for i in range(length)]
        test_label = []
        for i in range(len(test_data["labels"])):
            if test_data["labels"][str(i)] == True:
                test_label.append(1)
            else:
                test_label.append(0)
        encodings = tokenizer(test_text,
                              truncation=True,
                              padding=True,
                              max_length=128)
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


if __name__ == "__main__":

    main()

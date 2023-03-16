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
import numpy as np
import argparse
import os
import wandb
import jsonlines
from sklearn.metrics import accuracy_score, f1_score

# To do: wandb version controling
parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=60)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--bz", type=int, default=8)
parser.add_argument("--data", type=str, default="raop")
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


def dataset_process(dic_data, tokenizer, label_map=None):
    text = []
    labels = []
    for index in dic_data["labels"].keys():
        text.append(dic_data["context"][index])
        if dic_data["labels"][index] == True:
            labels.append(0)
        else:
            labels.append(1)
    encodings = tokenizer(text, truncation=True, padding=True)
    dataset = Custom_Dataset(encodings, labels)
    return dataset, text


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=7).to(device)
    if args.data == "raop":
        Test_file = "..\css_data\\raop\\test.json"
        f = open(Test_file, "r")
        content = f.read()
        test_data = json.loads(content)
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        all_file = "..\css_data\\raop\\raop.json"
        f = open(all_file, "r")
        content = f.read()
        all_data = json.loads(content)
        train_val_text = []
        train_val_label = []
        count = 0
        test_text = []
        test_label = []
        for index in all_data['labels'].keys():
            if index in test_data['labels'].keys():
                test_text.append(all_data['context'][index])
                test_label.append(label_map[all_data['labels'][index]])
            else:
                train_val_text.append(all_data['context'][index])
                train_val_label.append(label_map[all_data['labels'][index]])
        text_train, text_val, label_train, label_val = train_test_split(
            train_val_text, train_val_label, test_size=0.2)
        encodings = tokenizer(text_train, truncation=True, padding=True)
        train_dataset = Custom_Dataset(encodings, label_train)
        encodings = tokenizer(text_val, truncation=True, padding=True)
        dev_dataset = Custom_Dataset(encodings, label_val)
        encodings = tokenizer(test_text, truncation=True, padding=True)
        test_dataset = Custom_Dataset(encodings, test_label)

    training_args = TrainingArguments(
        output_dir=args.data + args.model + "_output",
        evaluation_strategy="steps",
        eval_steps=int(len(train_dataset) * args.epoch / args.bz / 3),
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bz,
        per_device_eval_batch_size=args.bz * 25,
        weight_decay=args.wd,
        max_grad_norm=args.norm,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_output = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print(test_output)


if __name__ == "__main__":

    main()

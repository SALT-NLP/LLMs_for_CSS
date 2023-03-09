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
from convokit import Corpus, download
import wandb
import jsonlines
from sklearn.metrics import accuracy_score, f1_score

# To do: wandb version controling
parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=60)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--bz", type=int, default=4)
parser.add_argument("--data", type=str, default="power")
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
        args.model, num_labels=2).to(device)
    if args.data == "implicit_hate":
        label_map = {
            "white_grievance": 0,
            "incitement": 1,
            "inferiority": 2,
            "irony": 3,
            "stereotypical": 4,
            "threatening": 5,
            "other": 6,
        }
        Test_file = "..\css_data\implicit_hate\\test.json"
        f = open(Test_file, "r")
        content = f.read()
        test_data = json.loads(content)
        test_dataset, text_test = dataset_process(test_data, tokenizer,
                                                  label_map)

        train_raw = load_dataset("SALT-NLP/ImplicitHate")
        text_train = []
        label_train = []
        for i in range(len(train_raw["train"])):
            if train_raw["train"][i]["post"] not in text_test:
                text_train.append(train_raw["train"][i]["post"])
                label_train.append(
                    label_map[train_raw["train"][i]["implicit_class"]])
        text_train, text_val, label_train, label_val = train_test_split(
            text_train, label_train, test_size=0.2)
        train_encodings = tokenizer(text_train, truncation=True, padding=True)
        train_dataset = Custom_Dataset(train_encodings, label_train)
        val_encodings = tokenizer(text_val, truncation=True, padding=True)
        val_dataset = Custom_Dataset(val_encodings, label_val)
    if args.data == "power":
        Test_file = "..\css_data\wiki_corpus\\test.json"
        label_map = {'True': 0, 'False': 1}
        f = open(Test_file, "r")
        content = f.read()
        test_data = json.loads(content)
        test_dataset, text_test = dataset_process(test_data, tokenizer,
                                                  label_map)
        Train_file = r".\wiki-corpus\utterances.jsonl"
        text_train = []
        label_train = []
        with jsonlines.open(Train_file) as reader:
            for obj in reader:
                if obj['text'] not in text_test:
                    text_train.append(obj['text'])
                    if obj['meta']["is-admin"] == False:
                        label_train.append(1)
                    else:
                        label_train.append(0)
        text_train, text_val, label_train, label_val = train_test_split(
            text_train, label_train, test_size=0.2)
        text_train = text_train[0:int(len(text_train) / 20)]
        text_val = text_val[0:int(len(text_val) / 20)]
        label_train = label_train[0:int(len(label_train) / 20)]
        label_val = label_val[0:int(len(label_val) / 20)]

        train_encodings = tokenizer(text_train, truncation=True, padding=True)
        train_dataset = Custom_Dataset(train_encodings, label_train)
        val_encodings = tokenizer(text_val, truncation=True, padding=True)
        val_dataset = Custom_Dataset(val_encodings, label_val)

    training_args = TrainingArguments(
        output_dir=args.data + args.model + "_output",
        evaluation_strategy="steps",
        eval_steps=int(len(train_dataset) * args.epoch / args.bz / 10),
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bz,
        per_device_eval_batch_size=args.bz * 20,
        weight_decay=args.wd,
        max_grad_norm=args.norm,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_output = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print(test_output)


if __name__ == "__main__":

    main()

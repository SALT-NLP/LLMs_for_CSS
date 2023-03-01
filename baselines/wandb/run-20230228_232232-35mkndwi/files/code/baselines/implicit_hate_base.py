from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import json
import torch
import random
import numpy as np
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=60)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--bz", type=int, default=10)
parser.add_argument("--data", type=str, default="implicit_hate")
parser.add_argument("--model", type=str, default="roberta-large")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--norm", type=float, default=0.8)
parser.add_argument('--gpu',
                    default='0',
                    type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'acc': accuracy_score(labels, predictions),
        'macro-F1:': f1_score(labels, predictions, average='macro')
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
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def dataset_process(dic_data, tokenizer, label_map):
    length = len(dic_data['labels'])
    text = [dic_data['context'][str(i)] for i in range(length)]
    labels = [label_map[dic_data['labels'][str(i)]] for i in range(length)]
    encodings = tokenizer(text, truncation=True, padding=True)
    dataset = Custom_Dataset(encodings, labels)
    return dataset


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=7).to(device)
    if args.data == 'implicit_hate':
        label_map = {
            'white_grievance': 0,
            'incitement': 1,
            'inferiority': 2,
            'irony': 3,
            'stereotypical': 4,
            'threatening': 5,
            'other': 6
        }
        Test_file = "..\css_data\implicit_hate\\test.json"
        f = open(Test_file, 'r')
        content = f.read()
        test_data = json.loads(content)
        test_dataset = dataset_process(test_data, tokenizer, label_map)

        train_raw = load_dataset("SALT-NLP/ImplicitHate")
        train_data = {'context': {}, 'labels': {}}
        for i in range(len(train_raw['train'])):
            train_data['context'][str(i)] = train_raw['train'][i]['post']
            train_data['labels'][str(
                i)] = train_raw['train'][i]['implicit_class']
        train_dataset = dataset_process(train_data, tokenizer, label_map)

    training_args = TrainingArguments(output_dir=args.model,
                                      evaluation_strategy="epoch",
                                      num_train_epochs=1,
                                      learning_rate=args.lr,
                                      per_device_train_batch_size=args.bz,
                                      per_device_eval_batch_size=args.bz,
                                      weight_decay=args.wd,
                                      max_grad_norm=args.norm,
                                      save_strategy='no')

    # optimizer hyperparameter to be tuned
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # change to dev data
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_output = trainer.evaluate(test_dataset, metric_key_prefix='test')
    print(test_output)


if __name__ == "__main__":

    main()
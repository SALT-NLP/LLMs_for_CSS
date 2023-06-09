import argparse, csv, json, os, re, wget
from string import ascii_uppercase
import itertools
import numpy as np
import pandas as pd
from convokit import Corpus, download
from mappings import *
from latex_prompt_exporter import export_latex


def deep_get(recursive_key, recursive_dict):
    level = recursive_dict
    for key in recursive_key.split("."):
        level = level[key]
    return level


def boolify(df):
    column_names = df.select_dtypes(include=[np.number]).columns
    m = df[df.select_dtypes(include=[np.number]).columns].max().reset_index(name="max")
    n = m.loc[m["max"] == 1, "max"]
    p = column_names[n.index]
    df[p] = df[p].astype(bool)
    return df


def get_context_column(df, context_column, colname=True):
    def row(r, context_column):
        if colname or (len(context_column) > 1):
            return "\n\n".join([f"{col}: {r[col]}" for col in context_column])
        else:
            return "\n\n".join([r[col] for col in context_column])

    if type(context_column) == tuple:
        return [row(r, context_column) for _, r in df.iterrows()]
    return df[context_column]


def truncate(text, length=2048):
    # print(text, " ".join(text.split(" ")[:length]))
    return " ".join(text.split(" ")[:length])


def build_prompts(df, prompt_template):
    cols = re.findall(r"{\$([A-Za-z_ ]+)}", prompt_template)
    trunc_length = 2048 // max(len(cols), 1)

    prompts = []
    for _, row in df.iterrows():
        prompt = prompt_template
        for col in cols:
            prompt = prompt.replace(f"{{${col}}}", truncate(row[col], trunc_length))
        prompts.append(prompt)
    return prompts


def csv_process(dataset, save_dir, local=False, jsonl=False):
    context_column, label_columns = csv_column_map[dataset]
    additional_labels = []
    if type(label_columns) == tuple:
        additional_labels = label_columns[1:]
        label_columns = label_columns[0]

    df = pd.DataFrame()
    if local:
        df = pd.read_csv("{}/{}.csv".format(save_dir, dataset))
    elif jsonl:
        filename = "{}/{}.jsonl".format(save_dir, dataset)
        if not os.path.exists(filename):
            filename = wget.download(jsonl_download[dataset], out=filename)
        with open(filename, "r") as infile:
            data = data = {i: json.loads(L) for i, L in enumerate(infile.readlines())}
            df = pd.DataFrame.from_dict(data).T
    else:
        filename = "{}/{}.csv".format(save_dir, dataset)
        if not os.path.exists(filename):
            filename = wget.download(csv_download[dataset], out=filename)
        # df = pd.read_csv(filename)
        if type(context_column) in {str, tuple}:
            df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename, header=None)

    df["context"] = get_context_column(df, context_column)  # df[context_column]
    df["labels"] = df[label_columns]
    if additional_labels:
        df["additional_labels"] = get_context_column(
            df, additional_labels, colname=False
        )
    df = boolify(df)
    df["prompts"] = build_prompts(
        df, prompts_templates[dataset]
    )  # [prompts_templates[dataset]] * len(df["labels"])
    if dataset in drop_labels:
        df = df[~df["labels"].isin(drop_labels[dataset])]
    if additional_labels:
        df = df[["context", "labels", "prompts", "additional_labels"]]
    else:
        df = df[["context", "labels", "prompts"]]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    export_latex(
        dataset,
        df["context"][0],
        df["prompts"][0],
        df["labels"][0],
        "./latex/{}.json".format(dataset),
    )
    df.to_json("{}/{}.json".format(save_dir, dataset))


def sentence_alphaenumerate(text):
    def iter_all_strings():
        for size in itertools.count(1):
            for s in itertools.product(ascii_uppercase, repeat=size):
                yield "".join(s)

    string = ""
    for i, a in enumerate(iter_all_strings()):
        sents = text.split(". ")
        if i >= len(sents):
            break
        string += f"{a}: {sents[i]}"
        if i < len(sents) - 1:
            string += ".\n"
    return string


def alphaenumerate_process(dataset, save_dir):
    context_column, label_columns = csv_column_map[dataset]
    df = pd.read_csv("{}/{}.csv".format(save_dir, dataset))
    df["labels"] = df[label_columns]
    df = boolify(df)
    df["context"] = [
        sentence_alphaenumerate(text) for text in df[context_column].values
    ]
    df["prompts"] = build_prompts(df, prompts_templates[dataset])
    print(df.head()["prompts"])
    df = df[["context", "labels", "prompts"]]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    export_latex(
        dataset,
        df["context"][0],
        df["prompts"][0],
        df["labels"][0],
        "./latex/{}.json".format(dataset),
    )
    df.to_json("{}/{}.json".format(save_dir, dataset))


def convokit_process(dataset, save_dir):
    corpus = Corpus(
        filename=download(
            convokit_ds[dataset],
        ),
    )
    df = corpus.get_utterances_dataframe()
    contexts = []

    conversation_ids = []
    label_utterance_ids = []
    speaker_utterance_maps = []
    label_type, label_field, context_length = convokit_labels[dataset]
    for _, convo in df.groupby("conversation_id").__iter__():
        if context_length == "all":
            convo = convo.sort_values(by="timestamp")
            context = convo["text"].values.tolist()
            speakers = convo["speaker"].values.tolist()
            speakers = [
                speaker if type(speaker) == type("str") else speaker.id
                for speaker in speakers
            ]
            utterance_ids = convo.index.values.tolist()
            context = [
                "{}: {}".format(speaker, context)
                for context, speaker in zip(context, speakers)
            ]
            contexts.append("\n".join(context))
            conversation_ids.append(convo["conversation_id"].values.tolist()[0])
            label_utterance_ids.append(utterance_ids[-1])
            speaker_utterance_maps.append(
                {
                    speaker: convo.index[convo["speaker"] == speaker].values.tolist()[0]
                    for speaker in set(speakers)
                }
            )
        elif context_length == "first_two":
            convo = convo.sort_values(by="timestamp")
            first = convo.index.values.tolist()[0]
            initial = corpus.get_utterance(first)
            initial_message = "{}: {}".format(
                initial.speaker.id,
                initial.text,
            )
            replies = convo[convo["reply_to"] == first]
            context = replies["text"].values.tolist()
            speakers = replies["speaker"].values.tolist()
            reply_messages = [
                "{}: {}".format(speaker, context)
                for context, speaker in zip(context, speakers)
            ]
            context = ["\n".join([initial_message, reply]) for reply in reply_messages]
            contexts.extend(context)
            conversation_ids.extend(replies["conversation_id"].values.tolist())
            label_utterance_ids.extend(replies.index.values.tolist())
    labels = []
    prompts = []

    if label_type == "utterance":
        for utterance_id in label_utterance_ids:
            utterance = corpus.get_utterance(utterance_id).to_dict()
            label = deep_get(label_field, utterance)
            labels.append(label)
            prompts.append(prompts_templates[dataset])
    elif label_type == "speaker":
        speaker_contexts = []
        for i, utterances in enumerate(speaker_utterance_maps):
            for speaker, utterance_id in utterances.items():
                speaker_contexts.append(contexts[i])
                utterance = corpus.get_utterance(utterance_id).to_dict()
                label = deep_get(label_field, utterance)
                labels.append(label)
                prompts.append(
                    prompts_templates[dataset].replace("{$speaker}", speaker)
                )
        contexts = speaker_contexts
    elif label_type == "conversation":
        for conversation_id in conversation_ids:
            conversation = corpus.get_conversation(conversation_id).to_dict()
            label = deep_get(label_field, conversation)
            labels.append(label)
            prompts.append(prompts_templates[dataset])

    assert len(contexts) == len(labels) and len(contexts) == len(prompts)
    raw_data = {"context": contexts, "labels": labels, "prompts": prompts}
    data_f = pd.DataFrame.from_dict(raw_data)
    if dataset in drop_labels:
        data_f = data_f[~data_f["labels"].isin(drop_labels[dataset])]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    export_latex(
        dataset,
        contexts[0],
        prompts[0],
        labels[0],
        "./latex/{}.json".format(dataset),
    )
    data_f.to_json(save_dir + "/{}.json".format(dataset))


def main(dataset, save_dir):
    if dataset in convokit_ds:
        convokit_process(dataset, save_dir)
    elif dataset in {"hippocorpus"}:
        alphaenumerate_process(dataset, save_dir)
    elif dataset in csv_download:
        csv_process(dataset, save_dir)
    elif dataset in jsonl_download:
        csv_process(dataset, save_dir, jsonl=True)
    else:
        csv_process(dataset, save_dir, local=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="power",
        choices=list(prompts_templates.keys()),
    )
    parser.add_argument("--save_dir", "-s", type=str, default="processed")
    args = parser.parse_args()
    main(args.dataset, args.save_dir)

import argparse
import csv
import os
import pandas as pd
import wget
from convokit import Corpus, download
from mappings import *


def deep_get(recursive_key, recursive_dict):
    level = recursive_dict
    for key in recursive_key.split("."):
        level = level[key]
    return level


def csv_process(dataset, save_dir):
    filename = "{}.csv".format(dataset)
    if not os.path.exists(filename):
        filename = wget.download(csv_download[dataset], out="{}.csv".format(dataset))
    df = pd.read_csv(filename)
    context_column, label_columns = csv_column_map[dataset]
    df["context"] = df[context_column]
    df["labels"] = df[label_columns]
    df["prompts"] = [prompts[dataset]] * len(df["labels"])
    df = df[["context", "labels", "prompts"]]
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
            prompts.append(prompts[dataset])
    elif label_type == "speaker":
        speaker_contexts = []
        for i, utterances in enumerate(speaker_utterance_maps):
            print(contexts[i])
            for speaker, utterance_id in utterances.items():
                speaker_contexts.append(contexts[i])
                utterance = corpus.get_utterance(utterance_id).to_dict()
                label = deep_get(label_field, utterance)
                labels.append(label)
                prompts.append(prompts[dataset].replace("{$speaker}", speaker))
        contexts = speaker_contexts
    elif label_type == "conversation":
        for conversation_id in conversation_ids:
            conversation = corpus.get_conversation(conversation_id).to_dict()
            label = deep_get(label_field, conversation)
            labels.append(label)
            prompts.append(prompts[dataset])

    assert len(contexts) == len(labels) and len(contexts) == len(prompts)
    raw_data = {"context": contexts, "labels": labels, "prompts": prompts}
    data_f = pd.DataFrame.from_dict(raw_data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_f.to_json(save_dir + "/{}.json".format(dataset))


def main(dataset, save_dir):
    if dataset in convokit_ds:
        convokit_process(dataset, save_dir)
    elif dataset in csv_download:
        csv_process(dataset, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="power",
        choices=list(prompts.keys()),
    )
    parser.add_argument("--save_dir", "-s", type=str, default="processed")
    args = parser.parse_args()
    main(args.dataset, args.save_dir)

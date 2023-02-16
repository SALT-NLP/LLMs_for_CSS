import argparse
import os
import pandas as pd
from convokit import Corpus, download

convokit_ds = {
    "discourse": "reddit-coarse-discourse-corpus",
    "persuasion": "winning-args-corpus",
    "toxicity": "conversations-gone-awry-corpus",
    "politeness": "wikipedia-politeness-corpus",
    "stance": "supreme-corpus",
    "power": "wiki-corpus",
}

# (Where is the Label Stored, What is the Label Applied to, how much context)
convokit_labels = {
    "power": ("speaker", "meta.is-admin", "all"),
    "stance": ("speaker", "meta.side", "all"),
    "politeness": ("utterance", "meta.Binary", "all"),
    "toxicity": ("conversation", "meta.conversation_has_personal_attack", "first_two"),
    "persuasion": ("utterance", "meta.success", "first_two"),
    "discourse": ("utterance", "meta.majority_type", "all"),
}

convokit_prompts = {
    "power": "Is {$speaker} an administrator (Yes or No)?",
    "stance": "Does {$speaker} support {$title} (Yes or No)?",  # This Dataset is strange currently, too long of context
    "politeness": "Was this statement polite (Yes or No)? ",
    "toxicity": "Predict whether the given conversation has a personal attack (True or False).",
    "persuasion": "Does this reply convince the original poster (Yes Or No)?",
    "discourse": "Which of the following best characterizes the previous statement: question, answer, announcement, agreement, appreciation, disagreement, elaboration, or humor? ",
}


def deep_get(recursive_key, recursive_dict):
    level = recursive_dict
    for key in recursive_key.split("."):
        level = level[key]
    return level


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
    print(label_type)
    for _, convo in df.groupby("conversation_id").__iter__():
        if context_length == "all":
            convo = convo.sort_values(by="timestamp")
            context = convo["text"].values.tolist()
            speakers = convo["speaker"].values.tolist()
            speakers = [
                speaker if type(speaker) == type("str") else speaker.id
                for speaker in speakers
            ]
            print(speakers)
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
            prompts.append(convokit_prompts[dataset])
    elif label_type == "speaker":
        speaker_contexts = []
        for i, utterances in enumerate(speaker_utterance_maps):
            for speaker, utterance_id in utterances.items():
                speaker_contexts.append(contexts[i])
                utterance = corpus.get_utterance(utterance_id).to_dict()
                label = deep_get(label_field, utterance)
                labels.append(label)
                prompts.append(convokit_prompts[dataset].replace("{$speaker}", speaker))
                contexts = speaker_contexts
    elif label_type == "conversation":
        for conversation_id in conversation_ids:
            conversation = corpus.get_conversation(conversation_id).to_dict()
            label = deep_get(label_field, conversation)
            labels.append(label)
            prompts.append(convokit_prompts[dataset])

    assert len(contexts) == len(labels) and len(contexts) == len(prompts)
    raw_data = {"context": contexts, "labels": labels, "prompts": prompts}
    data_f = pd.DataFrame.from_dict(raw_data)
    data_f.to_json(save_dir + "/{}.json".format(dataset))


def main(dataset, save_dir):
    if dataset in convokit_ds:
        convokit_process(dataset, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-d", type=str, default="power", choices=list(convokit_ds.keys())
    )
    parser.add_argument("--save_dir", "-s", type=str, default="processed")
    args = parser.parse_args()
    main(args.dataset, args.save_dir)

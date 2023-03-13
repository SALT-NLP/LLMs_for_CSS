import json
import os
import pandas as pd
import numpy as np
from os.path import exists
from os import getenv
from sys import argv, exit
from ast import literal_eval
from transformers import GPT2TokenizerFast, AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import time
import torch
import re
import random
import openai
from sklearn.metrics import classification_report
from mappings import labelsets


def tokenized_labelset(args):
    ls = set()
    for x in args.tokenizer(args.labelset, add_special_tokens=False)["input_ids"]:
        for y in x:
            ls.add(y)
    return sorted(ls)


def data_split(raw_datapth, input_path, args):
    if os.path.exists(input_path):
        print("###### Testing Files Exist! ######")
        return

    contexts = []
    labels = []
    prompts = []

    print("###### Creating Testing Files! ######")
    with open(raw_datapth, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    indexes = raw_data["context"].keys()
    df = pd.DataFrame.from_dict(raw_data)

    print(df.groupby("labels").size())

    num_testing = min(args.testing_size, len(indexes))

    if args.no_stratify:
        sample = df.sample(n=num_testing, random_state=random.seed(0))
        sample.to_json(input_path)
    else:
        samples = min(
            int(num_testing / len(df.groupby("labels"))),
            min(df.groupby("labels").count()["context"]),
        )
        random.seed(0)
        sample = df.groupby("labels", group_keys=False).apply(
            lambda x: x.sample(n=samples, random_state=random.seed(0))
        )
        sample.to_json(input_path)


def get_gpt3_response(args, oneprompt):
    if args.labelset is not None:
        LS = tokenized_labelset(args)
        weight = 20
        bias = {str(i): weight for i in LS}
        stop = None
        max_tokens = 1
    else:
        bias = {}
        max_tokens = 256
        stop = "."

    api_query = openai.Completion.create(
        engine=args.model,
        prompt=oneprompt,
        logit_bias=bias,
        temperature=0,
        max_tokens=max_tokens,
        stop=stop,
        user="RESEARCH-DATASET-" + args.dataset,
    )

    # print(api_query)
    response = api_query["choices"][0]["text"]

    return response


def get_chatgpt_response(args, oneprompt):
    if args.labelset is not None:
        LS = tokenized_labelset(args)
        weight = 20
        bias = {str(i): weight for i in LS}
        stop = None
        max_tokens = 2
    else:
        bias = {}
        max_tokens = 256
        stop = "."

    api_query = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": oneprompt},
        ],
        logit_bias=bias,
        temperature=0,
        max_tokens=max_tokens,
        stop=stop,
        user="RESEARCH-DATASET-" + args.dataset,
    )
    response = api_query["choices"][0]["message"]["content"]
    return response


@torch.no_grad()
def get_flan_response(args, oneprompt):
    input_ids = args.tokenizer(oneprompt, return_tensors="pt").input_ids.cuda()
    args.labelset = [
        label.lower() if len(label) > 1 else label for label in args.labelset
    ]
    LS = tokenized_labelset(args)
    if args.labelset is not None:
        decoder_input_ids = args.tokenizer("", return_tensors="pt").input_ids.cuda()
        decoder_input_ids = args.flan._shift_right(decoder_input_ids)
        logits = args.flan(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids
        ).logits.flatten()
        probs = (
            torch.nn.functional.softmax(
                torch.tensor([logits[i] for i in LS]),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        LS_str_map = args.tokenizer.decode(LS).split(" ")
        print(LS_str_map)
        response = {i: LS_str_map[i] for i in range(len(LS))}[np.argmax(probs)]
    else:
        if "ul2" in args.model:
            gen_config = GenerationConfig.from_pretrained(
                args.model, max_new_tokens=256
            )
        else:
            gen_config = GenerationConfig.from_pretrained(
                "google/flan-t5-xxl", max_new_tokens=256
            )
        stop = args.tokenizer(".")[0]
        args.flan(input_ids, generation_config=gen_config, forced_eos_token_id=stop)

    return response


def get_response(allprompts, args):
    global errortime
    allresponse = []
    i = 0
    while i < len(allprompts):
        oneprompt = allprompts[i]

        if args.model == "chatgpt":
            max_tokens = 4094
        elif "flan" in args.model:
            max_tokens = 4094
        elif "text-" in args.model:
            max_tokens = 2040

        oneprompt = args.tokenizer.clean_up_tokenization(
            args.tokenizer.convert_tokens_to_string(
                args.tokenizer.convert_ids_to_tokens(
                    args.tokenizer(oneprompt, max_length=max_tokens, truncation=True)[
                        "input_ids"
                    ]
                )
            )
        )
        # print(oneprompt)
        try:
            if args.model == "chatgpt":
                response = get_chatgpt_response(args, oneprompt)
            elif "flan" in args.model:
                response = get_flan_response(args, oneprompt)
            elif "text-" in args.model:
                response = get_gpt3_response(args, oneprompt)
            print("######Response#####", response)

            allresponse.append(response)
            i += 1
            errortime = 0
        except Exception as exc:
            print(exc)
            print(f"Data point {i} went wrong!")

            allresponse.append("Error!")
            errortime += 1
            if errortime > 60:
                print("Error too many times! sleep 1200s and move on")
                errortime = 0
                time.sleep(1200)
                i += 1
            else:
                print("Error and Retry after 2 minutes.")
                time.sleep(120)

    return allprompts, allresponse


def get_answers(input_path, output_path, prompts_path, args):
    print("###### Getting Answers! ######")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    count = len(raw_data["labels"])
    print("###### Number of Data: ", count, " ######")
    allflag = [0 for i in range(count)]

    if not os.path.exists(output_path):
        print("no answer file! create now")
        f = open(output_path, "w+", encoding="utf-8")
        f.close()
    else:
        with open(output_path, "r", encoding="utf-8") as f:
            for oneline in f:
                onedata = oneline.strip().split("\t")
                if len(onedata) != 3:
                    continue
                thisindex = int(onedata[0])
                allflag[thisindex] = 1

    print(sum(allflag), len(allflag))
    if sum(allflag) == len(allflag):
        print("\n ###### Finished Answer! ###### \n")
        return

    while True:
        test_samples = []
        gold_label = []
        touseindex = []
        prompts = []

        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            conversations = raw_data["context"]
            i = 0
            for u, v in conversations.items():
                if allflag[i] == 1:
                    i += 1
                    continue
                test_samples.append(v)
                gold_label.append(raw_data["labels"][u])
                prompts.append(raw_data["prompts"][u])
                touseindex.append(i)
                i += 1

        input_prompts = []
        for i in range(len(test_samples)):
            oneres = test_samples[i]
            input_prompts.append(oneres + " " + prompts[i])

        print("DUMPING PROMPTS")
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in enumerate(input_prompts)},
                f,
                ensure_ascii=False,
                indent=4,
            )

        fw = open(output_path, "a+", encoding="utf-8")
        response = []
        for i in range(len(input_prompts)):
            while True:
                _, oneresponse = get_response([input_prompts[i]], args)
                touseresponse = oneresponse[0].replace("\n", "&&&&&&")
                response.append(touseresponse)
                if "Error" not in touseresponse and in_domain(
                    touseresponse, args
                ):  # implement: in_domain
                    print("no error for this sample")
                    allflag[touseindex[i]] = 1
                    print(touseindex[i], gold_label[i], touseresponse)
                    fw.write(
                        str(touseindex[i])
                        + "\t"
                        + str(gold_label[i])
                        + "\t"
                        + str(touseresponse)
                        + "\n"
                    )
                    fw.flush()
                    break
                else:
                    print("Error After Sleep and Repeat")
                    break
                if args.sleep:
                    time.sleep(args.sleep)
        fw.close()
        # end = time.time()
        # print("all used time: ", end - start)

        iffinish = True
        for oneflag in allflag:
            if oneflag == 0:
                iffinish = False
                break

        if iffinish:
            break


def in_domain(response, args):
    if args.labelset is not None:
        # labelset = literal_eval(args.labelset)
        for lbl in args.labelset:
            if lbl in response:
                return True
        return False
    return True


def calculateres(path, args):
    with open(args.input_path, "r") as f:
        a = json.load(f)
    label_set = set([str(v).lower() for (u, v) in a["labels"].items()])
    print("###### Label Space:", label_set)
    label_dict = {"None": 0}

    i = 1
    for u in label_set:
        label_dict[u] = i
        i += 1

    f = open(path, "r", encoding="utf-8")
    allnum = 0
    accnum = 0

    preds = []
    golds = []
    target_names = list(label_dict.keys())

    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        content = oneline.split("\t")
        if len(content) != 3:
            continue
        index = int(content[0])
        allnum += 1

        if args.dataset in [
            "humor",
            "supreme_corpus",
        ]:
            # print(content[1])
            gold = content[1].lower()
            pred = content[2].lower()
            print(gold, pred)
            if gold in pred:
                accnum += 1
        elif args.dataset in ["power", "conv_go_awry"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "true": ["true", "yes"],
                "false": ["false", "no"],
            }
            if pred in mapping[gold]:
                accnum += 1
        elif args.dataset in ["politeness"]:
            gold = content[1]
            pred = content[2].lower().replace("&", "")
            mapping = {
                "1": "A",
                "0": "B",
                "-1": "C",
            }
            if pred == mapping[gold].lower():
                accnum += 1
                
        elif args.dataset in ["persuasion"]:
            gold = content[1]
            pred = content[2].lower().replace("&", "")
            
            
            mapping = {
                "1.0": "True",
                "0.0": "False",
            }
            
            
            if pred == mapping[gold].lower():
                accnum += 1
                
        elif args.dataset in ["flute-classification"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "idiom": "A",
                "metaphor": "B",
                "sarcasm": "C",
                "simile": "D",
            }
            if pred == mapping[gold].lower():
                accnum += 1
        elif args.dataset in ["tempowic"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "same": "A",
                "different": "B",
            }
            if pred == mapping[gold].lower():
                accnum += 1
        elif args.dataset in ["semeval_stance"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {"against": "A", "favor": "B", "none": "C"}
            if pred == mapping[gold].lower():
                accnum += 1
        elif args.dataset in ["raop"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {"persuasive": "A", "not persuasive": "B"}
            if pred == mapping[gold].lower():
                accnum += 1
        elif args.dataset in ["ibc"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "liberal": "A",
                "conservative": "B",
                "neutral": "C",
            }
            if pred == mapping[gold].lower():
                accnum += 1
        elif args.dataset in ["emotion", "talklife"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            if pred == gold:
                accnum += 1
        elif args.dataset in ["media_ideology"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "left": "A",
                "right": "B",
                "center": "C",
            }
            if pred == mapping[gold].lower():
                accnum += 1
        elif args.dataset in ["hate"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "white_grievance": "A",
                "incitement": "B",
                "inferiority": "C",
                "irony": "D",
                "stereotypical": "E",
                "threatening": "F",
            }
            if pred == mapping[gold].lower():
                accnum += 1
        elif args.dataset in ["discourse"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "question": "A",
                "answer": "B",
                "agreement": "C",
                "disagreement": "D",
                "appreciation": "E",
                "elaboration": "F",
                "humor": "G",
            }
            if pred == mapping[gold].lower():
                accnum += 1
        elif args.dataset in ["indian_english_dialect"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "preposition omission": "R",
                "copula omission": "B",
                "resumptive subject pronoun": "S",
                "resumptive object pronoun": "T",
                "extraneous article": "D",
                "focus only": "F",
                "mass nouns as count nouns": "N",
                "stative progressive": "U",
                "lack of agreement": "K",
                "none of the above": "W",
                "lack of inversion in wh-questions": "L",
                "topicalized non-argument constituent": "V",
                "inversion in embedded clause": "J",
                "focus itself": "E",
                'general extender "and all"': "G",
                "object fronting": "P",
                'invariant tag "isn’t it, no, na"': "I",
                "habitual progressive": "H",
                "article omission": "A",
                "prepositional phrase fronting with reduction": "Q",
                'non-initial existential "is / are there"': "O",
                "left dislocation": "M",
                "direct object pronoun drop": "C",
            }
            if pred == mapping[gold].lower():
                accnum += 1
        else:
            pass

    print("\n ###### Results ###### \n")
    print("Acc: ", float(accnum) / float(allnum))
    print("Number of Correct Data: ", accnum)
    print("Number of Testing Data: ", allnum)

    if len(preds) > 0:
        print(classification_report(golds, preds, target_names=target_names))


def parse_arguments():
    parser = argparse.ArgumentParser(description="chatgpt-zero-shot-css")
    parser.add_argument(
        "--dataset",
        type=str,
        default="conv_go_awry",
        choices=[
            "discourse",
            "conv_go_awry",
            "power",
            "hate",
            "humor",
            "flute-classification",
            "flute-explanation",
            "supreme_corpus",
            "politeness",
            "media_ideology",
            "hippocorpus",
            "indian_english_dialect",
            "ibc",
            "semeval_stance",
            "tempowic",
            "sbic",
            "mrf-explanation",
            "mrf-classification",
            "talklife",
            "emotion",
            "raop",
            "tropes",
            "persuasion",
        ],
        help="dataset used for experiment",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="chatgpt",
        choices=[
            "chatgpt",
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
            "google/flan-ul2",
            "text-davinci-001",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "text-davinci-002",
            "text-davinci-003",
        ],
    )
    parser.add_argument("--labelset", default=None)
    parser.add_argument("--list_generation", action="store_true")
    parser.add_argument("--no_stratify", action="store_true")
    parser.add_argument("--sleep", type=int, default=0)
    parser.add_argument("--ngpu", "-g", type=int, default=2)
    args = parser.parse_args()

    if args.dataset == "conv_go_awry":
        args.raw_datapath = "css_data/conv_go_awry/toxicity.json"
        args.input_path = "css_data/conv_go_awry/test.json"
        args.answer_path = "css_data/conv_go_awry/answer"
    elif args.dataset == "power":
        args.raw_datapath = "css_data/wiki_corpus/power.json"
        args.input_path = "css_data/wiki_corpus/test.json"
        args.answer_path = "css_data/wiki_corpus/answer"
    elif args.dataset == "hate":
        args.raw_datapath = "css_data/implicit_hate/hate.json"
        args.input_path = "css_data/implicit_hate/test.json"
        args.answer_path = "css_data/implicit_hate/answer"
    elif args.dataset == "discourse":
        args.raw_datapath = "css_data/discourse/discourse.json"
        args.input_path = "css_data/discourse/test.json"
        args.answer_path = "css_data/discourse/answer"
    elif args.dataset == "humor":
        args.raw_datapath = "css_data/reddit_humor/humor.json"
        args.input_path = "css_data/reddit_humor/test.json"
        args.answer_path = "css_data/reddit_humor/answer"
    elif args.dataset == "persuasion":
        args.raw_datapath = "css_data/persuasion/persuasion.json"
        args.input_path = "css_data/persuasion/test.json"
        args.answer_path = "css_data/persuasion/answer"
    elif args.dataset == "flute-explanation":
        args.raw_datapath = "css_data/flute/flute-explanation.json"
        args.input_path = "css_data/flute/test-explanation.json"
        args.answer_path = "css_data/flute/answer-explanation"
    elif args.dataset == "flute-classification":
        args.raw_datapath = "css_data/flute/flute-classification.json"
        args.input_path = "css_data/flute/test-classification.json"
        args.answer_path = "css_data/flute/answer-classification"
    elif args.dataset == "supreme_corpus":
        args.raw_datapath = "css_data/supreme_corpus/stance.json"
        args.input_path = "css_data/supreme_corpus/test.json"
        args.answer_path = "css_data/supreme_corpus/answer"
        args.labelset = labelsets["stance"]
    elif args.dataset == "politeness":
        args.raw_datapath = "css_data/wiki_politeness/politeness.json"
        args.input_path = "css_data/wiki_politeness/test.json"
        args.answer_path = "css_data/wiki_politeness/answer"
    elif args.dataset == "media_ideology":
        args.raw_datapath = "css_data/media_ideology/media_ideology.json"
        args.input_path = "css_data/media_ideology/test.json"
        args.answer_path = "css_data/media_ideology/answer"
    elif args.dataset == "hippocorpus":
        args.raw_datapath = "css_data/hippocorpus/hippocorpus.json"
        args.input_path = "css_data/hippocorpus/test.json"
        args.answer_path = "css_data/hippocorpus/answer"
    elif args.dataset == "indian_english_dialect":
        args.raw_datapath = (
            "css_data/indian_english_dialect/indian_english_dialect.json"
        )
        args.input_path = "css_data/indian_english_dialect/test.json"
        args.answer_path = "css_data/indian_english_dialect/answer"
    elif args.dataset == "ibc":
        args.raw_datapath = "css_data/ibc/ibc.json"
        args.input_path = "css_data/ibc/test.json"
        args.answer_path = "css_data/ibc/answer"
    elif args.dataset == "semeval_stance":
        args.raw_datapath = "css_data/semeval_stance/semeval_stance.json"
        args.input_path = "css_data/semeval_stance/test.json"
        args.answer_path = "css_data/semeval_stance/answer"
    elif args.dataset == "tempowic":
        args.raw_datapath = "css_data/tempowic/tempowic.json"
        args.input_path = "css_data/tempowic/test.json"
        args.answer_path = "css_data/tempowic/answer"
    elif args.dataset == "sbic":
        args.raw_datapath = "css_data/sbic/sbic.json"
        args.input_path = "css_data/sbic/test.json"
        args.answer_path = "css_data/sbic/answer"
        args.no_stratify = True
    elif args.dataset == "talklife":
        args.raw_datapath = "css_data/talklife/talklife.json"
        args.input_path = "css_data/talklife/test.json"
        args.answer_path = "css_data/talklife/answer"
    elif args.dataset == "raop":
        args.raw_datapath = "css_data/raop/raop.json"
        args.input_path = "css_data/raop/test.json"
        args.answer_path = "css_data/raop/answer"
    elif args.dataset == "emotion":
        args.raw_datapath = "css_data/emotion/emotion.json"
        args.input_path = "css_data/emotion/test.json"
        args.answer_path = "css_data/emotion/answer"
    elif args.dataset == "mrf-explanation":
        args.raw_datapath = "css_data/mrf/mrf-explanation.json"
        args.input_path = "css_data/mrf/test-explanation.json"
        args.answer_path = "css_data/mrf/answer-explanation"
        args.no_stratify = True
    elif args.dataset == "mrf-classification":
        args.raw_datapath = "css_data/mrf/mrf-classification.json"
        args.input_path = "css_data/mrf/test-classification.json"
        args.answer_path = "css_data/mrf/answer-classification"
    elif args.dataset == "tropes":
        args.raw_datapath = "css_data/tropes/tropes.json"
        args.input_path = "css_data/tropes/test.json"
        args.answer_path = "css_data/tropes/answer"
    else:
        raise ValueError("dataset is not properly defined ...")
    if args.labelset is None:
        args.labelset = labelsets[args.dataset]
    if (args.list_generation) and (args.labelset is not None):
        args.labelset.extend([" ", ","])

    if args.model == "chatgpt" or "text-" in args.model:
        args.tokenizer = GPT2TokenizerFast.from_pretrained(
            "gpt2", truncation_side="left"
        )
        args.answer_path = args.answer_path + "-" + args.model
    elif "flan" in args.model:
        args.tokenizer = AutoTokenizer.from_pretrained(
            args.model, truncation_side="left"
        )
        args.flan = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        heads_per_gpu = len(args.flan.encoder.block) // args.ngpu
        device_map = {
            gpu: list(
                range(
                    0 + (gpu * heads_per_gpu),
                    (0 + (gpu * heads_per_gpu)) + heads_per_gpu,
                )
            )
            for gpu in range(args.ngpu)
        }
        args.flan.parallelize(device_map)
        args.flan.eval()
        print(args.model)
        args.answer_path = args.answer_path + "-" + args.model.split("/")[-1]
    # substitute this with your own access token!
    args.testing_size = 500

    return args


def main():
    print("We are using chatgpt to test different datasets now!\n")
    args = parse_arguments()

    try:
        input_path = args.input_path
        answer_path = args.answer_path
        prompts_path = args.answer_path.replace("/answer", "/prompts.json")

        raw_datapath = args.raw_datapath

        data_split(raw_datapath, input_path, args)

        get_answers(input_path, answer_path, prompts_path, args)

        calculateres(answer_path, args)

    except KeyboardInterrupt:
        print("\n !!!!!! Key Interruptions! Goodbye! !!!!!! \n")
        exit()
    except Exception as exc:
        print(exc)
        exit()


errortime = 0
if __name__ == "__main__":
    st = time.time()
    main()
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print("###### Execution Time:", elapsed_time, " seconds. ######")

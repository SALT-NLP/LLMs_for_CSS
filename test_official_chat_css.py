import json
import os
import pandas as pd
from os.path import exists
from os import getenv
from sys import argv, exit
from ast import literal_eval
from transformers import GPT2TokenizerFast
import time
import re
import random
import argparse
import openai
from sklearn.metrics import classification_report
from config import config_access_token

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", truncation_side="left")


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
    samples = int(num_testing / len(df.groupby("labels")))
    random.seed(0)
    sample = df.groupby("labels", group_keys=False).apply(
        lambda x: x.sample(n=samples, random_state=random.seed(0))
    )

    sample.to_json(input_path)


def get_response(allprompts, args):
    global errortime
    allresponse = []
    i = 0
    while i < len(allprompts):
        oneprompt = allprompts[i]
        oneprompt = tokenizer.clean_up_tokenization(
            tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(
                    tokenizer(oneprompt, max_length=4094, truncation=True)["input_ids"]
                )
            )
        )
        # print(oneprompt)
        try:
            if args.free_generation:
                bias = {}
                max_tokens = 256
                stop = '.'
            else:
                stop = None
                max_tokens = 2
                if True:
                    bias = {str(i): 10 for i in range(32, 39)}
                else:
                    bias = {
                        "5297": 20,
                        "2949": 20,
                        "17821": 20,
                        "25101": 20,
                    }
                
            api_query = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": oneprompt},
                ],
                logit_bias=bias,
                temperature=0,
                max_tokens=max_tokens,
                stop=stop,
            )
            response = api_query["choices"][0]["message"]["content"]
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
        labelset = literal_eval(args.labelset)
        for lbl in labelset:
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
            "reddit_humor",
            "supreme_corpus",
        ]:
            # print(content[1])
            gold = content[1].lower()
            pred = content[2].lower()
            print(gold, pred)
            if gold in pred:
                accnum += 1
        elif args.dataset in ["wiki_corpus", "conv_go_awry"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "true": ["true", "yes"],
                "false": ["false", "no"],
            }
            if pred in mapping[gold]:
                accnum += 1
        elif args.dataset in ["wiki_politeness"]:
            gold = content[1]
            pred = content[2].lower().replace("&", "")
            mapping = {
                "1": "A",
                "0": "B",
                "-1": "C",
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
        elif args.dataset in ["ibc"]:
            gold = content[1].lower()
            pred = content[2].lower().replace("&", "")
            mapping = {
                "left": "A",
                "right": "B",
                "center": "C",
            }
            if pred == mapping[gold].lower():
                accnum += 1
        elif args.dataset in ["implicit_hate"]:
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
                "announcement": "C",
                "agrement": "D",
                "appreciation": "E",
                "elaboration": "F",
                "humor": "G",
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
            "wiki_corpus",
            "implicit_hate",
            "reddit_humor",
            "flute-classification",
            "flute-explanation",
            "supreme_corpus",
            "wiki_politeness",
            "media_ideology",
            "hippocorpus",
            "indian_english_dialect",
            "ibc",
            "semeval_stance",
            "tempowic",
            "sbic",
        ],
        help="dataset used for experiment",
    )
    parser.add_argument("--labelset", default=None)
    parser.add_argument("--free_generation", action="store_true")
    parser.add_argument("--sleep", type=int, default=0)
    args = parser.parse_args()
    if args.dataset == "conv_go_awry":
        args.raw_datapath = "css_data/conv_go_awry/toxicity.json"
        args.input_path = "css_data/conv_go_awry/test.json"
        args.answer_path = "css_data/conv_go_awry/answer"
    elif args.dataset == "wiki_corpus":
        args.raw_datapath = "css_data/wiki_corpus/power.json"
        args.input_path = "css_data/wiki_corpus/test.json"
        args.answer_path = "css_data/wiki_corpus/answer"
    elif args.dataset == "implicit_hate":
        args.raw_datapath = "css_data/implicit_hate/hate.json"
        args.input_path = "css_data/implicit_hate/test.json"
        args.answer_path = "css_data/implicit_hate/answer"
    elif args.dataset == "discourse":
        args.raw_datapath = "css_data/discourse/discourse.json"
        args.input_path = "css_data/discourse/test.json"
        args.answer_path = "css_data/discourse/answer"
    elif args.dataset == "reddit_humor":
        args.raw_datapath = "css_data/reddit_humor/humor.json"
        args.input_path = "css_data/reddit_humor/test.json"
        args.answer_path = "css_data/reddit_humor/answer"
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
    elif args.dataset == "wiki_politeness":
        args.raw_datapath = "css_data/wiki_politeness/politeness.json"
        args.input_path = "css_data/wiki_politeness/test.json"
        args.answer_path = "css_data/wiki_politeness/answer"
    elif args.dataset == "media_ideology":
        args.raw_datapath = "css_data/media_ideology/media_ideology.json"
        args.input_path = "css_data/media_ideology/test.json"
        args.answer_path = "css_data/media_ideology/answer"
        args.labelset = "['left', 'right', 'center', 'centrist', 'neutral', 'liberal', 'conservative', 'A' , 'B', 'C']"
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
    else:
        raise ValueError("dataset is not properly defined ...")

    # substitute this with your own access token!
    args.testing_size = 500

    args.access_token = config_access_token
    # "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJqaWFhb2NAYWxsZW5haS5vcmciLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZ2VvaXBfY291bnRyeSI6IlVTIn0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJ1c2VyX2lkIjoidXNlci1JWVpYSEFrdVJKTXJrVXlLU2RSbFZEWWkifSwiaXNzIjoiaHR0cHM6Ly9hdXRoMC5vcGVuYWkuY29tLyIsInN1YiI6ImF1dGgwfDYwOWFjMWY0NGMxZjQ1MDA3MDQwYmExZiIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2NzY0OTA3NjAsImV4cCI6MTY3NzcwMDM2MCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvZmZsaW5lX2FjY2VzcyJ9.XFsYqMo1JpK58MYk0QzqkuIn2bTfknFzjBGkYFHznPj-dQjgHuyxB6HwgznSj7jYa2hmloBMK3FxV3peXQ5aLiqfh0QIBgHWUlr3CSCm2ypB82V8HjcgN-18WYlACIg_w7im7xYmMv3_1iRGWyq4d1-8vzxgtADrthqPNcjaib3nPwj9RzYOdcV6fZd4n54MqcuXn2l-Yge0weB539GvBRkinCmEbcNJZKJ3VYQu6EiO0t_MzRodCOLnD-auZBfs-sbyVMuRH65RSjIqVsdhp8S_f2gmTaMs4MRU2CC0b8QX-3mVFZmhRHhUYA5TEaEaHT8Y83AA0j3C6erwx-gMpg"

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

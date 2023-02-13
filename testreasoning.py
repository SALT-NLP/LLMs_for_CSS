import json
import os
from os.path import exists
from os import getenv
from sys import argv, exit
import time
import re
import random
import argparse
#from revChatGPT.ChatGPT import Chatbot
from revChatGPT.Official import Chatbot


def get_input(prompt):
    # prompt for input
    lines = []
    print(prompt, end="")
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    # Join the lines, separated by newlines, and print the result
    user_input = "\n".join(lines)
    # print(user_input)
    return user_input


def configure(args):
    try:
        inputpath = args.inputpath
        rationalepath = args.rantionalepath
        answerpath = args.answerpath

        getrationales(inputpath, rationalepath, args)
        getanswers(rationalepath, answerpath, args)
        calculateres(answerpath,args)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        exit()
    except Exception as exc:
        print(exc)
        exit()

def getResponseforPrompt(chatbot, allprompts):
    global errortime
    allresponse = []
    i = 0
    while i < len(allprompts):
        oneprompt = allprompts[i]
        try:
            message = chatbot.ask(oneprompt)
            allresponse.append(message["choices"][0]["text"])
            i += 1
            errortime = 0
            chatbot.reset()
        except Exception as exc:
            print(f"Data point {i} went wrong!")
            print(exc)
            allresponse.append("Error!")
            errortime += 1
            if errortime > 60:
                print("error too many times! sleep 1800s")
                errortime = 0
                #time.sleep(1800)
                time.sleep(180)
            i += 1
        #time.sleep(60)
        time.sleep(6)
    return allprompts, allresponse


def getrationales(inputpath, outputpath, args):
    print("get rationale")
    chatbot = Chatbot(api_key=args.api_key)

    start = time.time()

    with open(inputpath, "r", encoding="utf-8") as f:
        count = len(f.readlines())
    print("number of data: ", count)
    allflag = [0 for i in range(count)]  ####the number of all samples

    if not os.path.exists(outputpath):
        print("no rationale file! create now")
        f = open(outputpath, "w+", encoding="utf-8")
        f.close()
    else:
        with open(outputpath, "r", encoding="utf-8") as f:
            for oneline in f:
                onedata = oneline.strip().split("\t")
                if len(onedata) != 3:
                    continue
                thisindex = int(onedata[0])
                allflag[thisindex] = 1

    print(sum(allflag), len(allflag))
    if sum(allflag) == len(allflag):
        print("finished rationales")
        return

    while True:
        testsamples = []
        goldlabel = []
        touseindex = []
        with open(inputpath, "r", encoding="utf-8") as f:
            i = 0
            for oneline in f:
                if allflag[i] == 1:
                    i += 1
                    continue
                onedata = json.loads(oneline)
                wrap_que = onedata["wrap_que"]
                testsamples.append(wrap_que)
                goldlabel.append(onedata["gold_ans"])
                touseindex.append(i)
                i += 1

        inputprompts = []
        for i in range(len(testsamples)):
            oneres = testsamples[i]
            inputprompts.append(oneres)
            #print(oneres)
        #exit -1
        fw = open(outputpath, "a+", encoding="utf-8")
        response = []
        for i in range(len(inputprompts)):
            _, oneresponse = getResponseforPrompt(chatbot, [inputprompts[i]])
            touseresponse = oneresponse[0].replace('\n','&&&&&&')
            response.append(touseresponse)
            if "Error" not in touseresponse:
                print("no error for this sample")
                allflag[touseindex[i]] = 1
                touseprompt = inputprompts[i].replace('\n','&&&&&&')
                inputwithrationales = touseprompt + touseresponse + " " + args.direct_answer_trigger_for_zeroshot_cot
                print(touseindex[i], goldlabel[i], inputwithrationales)
                fw.write(str(touseindex[i]) + "\t" + goldlabel[i] + "\t" + inputwithrationales + "\n")
                fw.flush()
            else:
                print("this data point meets error! please repeat!")
        fw.close()
        end = time.time()
        print("all used time: ", end - start)

        iffinish = True
        for oneflag in allflag:
            if oneflag == 0:
                iffinish = False
                break

        if iffinish:
            break

def getanswers(inputpath, outputpath, args):
    print("get answers")
    chatbot = Chatbot(api_key=args.api_key)
    start = time.time()

    with open(inputpath, "r", encoding="utf-8") as f:
        count = len(f.readlines())
    print("number of data: ", count)
    allflag = [0 for i in range(count)]  ####the number of all samples

    if not os.path.exists(outputpath):
        print("no answer file! create now")
        f = open(outputpath, "w+", encoding="utf-8")
        f.close()
    else:
        with open(outputpath, "r", encoding="utf-8") as f:
            for oneline in f:
                onedata = oneline.strip().split("\t")
                if len(onedata) != 3:
                    continue
                thisindex = int(onedata[0])
                allflag[thisindex] = 1

    print(sum(allflag), len(allflag))
    if sum(allflag) == len(allflag):
        print("finished answer")
        return

    while True:
        testsamples = []
        goldlabel = []
        touseindex = []
        with open(inputpath, "r", encoding="utf-8") as f:
            i = 0
            for oneline in f:
                if allflag[i] == 1:
                    i += 1
                    continue
                onedata = oneline.strip().split('\t')
                wrap_que = onedata[2].replace('&&&&&&','\n')
                testsamples.append(wrap_que)
                goldlabel.append(onedata[1])
                touseindex.append(i)
                i += 1

        inputprompts = []
        for i in range(len(testsamples)):
            oneres = testsamples[i]
            inputprompts.append(oneres)
            #print(oneres)
        #exit -1
        fw = open(outputpath, "a+", encoding="utf-8")
        response = []
        for i in range(len(inputprompts)):
            _, oneresponse = getResponseforPrompt(chatbot, [inputprompts[i]])
            touseresponse = oneresponse[0].replace('\n','&&&&&&')
            response.append(touseresponse)
            if "Error" not in touseresponse:
                print("no error for this sample")
                allflag[touseindex[i]] = 1
                print(touseindex[i], goldlabel[i], touseresponse)
                fw.write(str(touseindex[i]) + "\t" + goldlabel[i] + "\t" + touseresponse + "\n")
                fw.flush()
            else:
                print("this data point meets error! please repeat!")
        fw.close()
        end = time.time()
        print("all used time: ", end - start)

        iffinish = True
        for oneflag in allflag:
            if oneflag == 0:
                iffinish = False
                break

        if iffinish:
            break

def calculateres(path, args):
    f = open(path, 'r', encoding="utf-8")
    allnum = 0
    accnum = 0
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        content = oneline.split('\t')
        if len(content) != 3:
            continue
        index = int(content[0])
        allnum += 1
        if args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
            gold = float(content[1])
        else:
            gold = content[1]
        pred = content[2]
        pred = pred.replace("&&&&&&", "")

        if args.dataset in ("aqua", "commonsensqa"):
            pred = re.findall(r'A|B|C|D|E', pred)
        elif args.dataset == "bigbench_date":
            pred = re.findall(r'A|B|C|D|E|F', pred)
        elif args.dataset in ("object_tracking"):
            pred = re.findall(r'A|B|C', pred)
        elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
        elif args.dataset in ("strategyqa", "coin_flip"):
            pred = pred.lower()
            pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
            pred = pred.split(" ")
            pred = [i for i in pred if i in ("yes", "no")]
        elif args.dataset == "last_letters":
            pred = re.sub("\"|\'|\n|\.|\s", "", pred)
            pred = [pred]
        else:
            raise ValueError("dataset is not properly defined ...")

        if len(pred) == 0:
            print("no answer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            pred = ""
        else:
            pred = pred[0]

        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]

        print(gold, pred, content[2])
        if args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
            if pred != "":
                if float(pred) == gold:
                    accnum += 1
        else:
            if pred == gold:
                accnum += 1

    print(allnum, accnum, float(accnum) / float(allnum))

def parse_arguments():
    parser = argparse.ArgumentParser(description="chatgpt-zero-shot-coT")

    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "coin_flip", "last_letters", "bigbench_date", "object_tracking"], help="dataset used for experiment"
    )
    args = parser.parse_args()
    if args.dataset == "aqua":
        args.inputpath = "data/reasoning/aqua_zero_shot_cot"
        args.rantionalepath = "data/reasoning/aqua_rationale_cot"
        args.answerpath = "data/reasoning/aqua_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.inputpath = "data/reasoning/gsm8k_zero_shot_cot"
        args.rantionalepath = "data/reasoning/gsm8k_rationale_cot"
        args.answerpath = "data/reasoning/gsm8k_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.inputpath = "data/reasoning/commonsensqa_zero_shot_cot"
        args.rantionalepath = "data/reasoning/commonsensqa_rationale_cot"
        args.answerpath = "data/reasoning/commonsensqa_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, among A through E, the answer is"
    elif args.dataset == "addsub":
        args.inputpath = "data/reasoning/addsub_zero_shot_cot"
        args.rantionalepath = "data/reasoning/addsub_rationale_cot"
        args.answerpath = "data/reasoning/addsub_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.inputpath = "data/reasoning/multiarith_zero_shot_cot"
        args.rantionalepath = "data/reasoning/multiarith_rationale_cot"
        args.answerpath = "data/reasoning/multiarith_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.inputpath = "data/reasoning/strategyqa_zero_shot_cot"
        args.rantionalepath = "data/reasoning/strategyqa_rationale_cot"
        args.answerpath = "data/reasoning/strategyqa_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.inputpath = "data/reasoning/svamp_zero_shot_cot"
        args.rantionalepath = "data/reasoning/svamp_rationale_cot"
        args.answerpath = "data/reasoning/svamp_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.inputpath = "data/reasoning/singleeq_zero_shot_cot"
        args.rantionalepath = "data/reasoning/singleeq_rationale_cot"
        args.answerpath = "data/reasoning/singleeq_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.inputpath = "data/reasoning/date_zero_shot_cot"
        args.rantionalepath = "data/reasoning/date_rationale_cot"
        args.answerpath = "data/reasoning/date_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.inputpath = "data/reasoning/object_tracking_zero_shot_cot"
        args.rantionalepath = "data/reasoning/object_tracking_rationale_cot"
        args.answerpath = "data/reasoning/object_tracking_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.inputpath = "data/reasoning/coin_flip_zero_shot_cot"
        args.rantionalepath = "data/reasoning/coin_flip_rationale_cot"
        args.answerpath = "data/reasoning/coin_flip_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.inputpath = "data/reasoning/last_letters_zero_shot_cot"
        args.rantionalepath = "data/reasoning/last_letters_rationale_cot"
        args.answerpath = "data/reasoning/last_letters_answer_cot"
        args.direct_answer_trigger_for_zeroshot_cot = "&&&&&&Therefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
    args.api_key = "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"
    return args

def main():
    print("We are using chatgpt to test different datasets now!\n")
    args = parse_arguments()
    configure(args)

errortime = 0
if __name__ == "__main__":
    main()
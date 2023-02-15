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
from revChatGPT.V2 import Chatbot


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

        #getrationales(inputpath, rationalepath, args)
        #getanswers(rationalepath, answerpath, args)
        getanswers(inputpath, answerpath, args)
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
    chatbot = Chatbot(session_token=args.session_token)

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
    chatbot = Chatbot(session_token=args.session_token)
    start = time.time()

    with open(inputpath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    count = len(raw_data['Label'])
    #   count = len(f.readlines())
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
            raw_data = json.load(f)
            conversations = raw_data['Conversation']
            i = 0
            for (u,v) in raw_data['Conversation'].items():
                if allflag[i] == 1:
                    i += 1
                    continue
                #onedata = oneline.strip().split('\t')
                #wrap_que = onedata[2].replace('&&&&&&','\n')
                testsamples.append(v)
                goldlabel.append(raw_data['Label'][u])
                touseindex.append(i)
                i += 1

        inputprompts = []
        for i in range(len(testsamples)):
            oneres = testsamples[i]
            inputprompts.append(oneres + args.direct_answer_trigger_for_zeroshot_cot)
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
        #if args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        #    gold = float(content[1])
        #else:
            #gold = content[1]
        gold = content[1] == 'true'
        
        #pred = 'True' in content[2]
        pred = 'true' in pred.replace("&&&&&&", "").lower()
        
        print(gold, pred, content[2])
        if pred == gold:
            accnum += 1

    print(allnum, accnum, float(accnum) / float(allnum))

def parse_arguments():
    parser = argparse.ArgumentParser(description="chatgpt-zero-shot-css")

    parser.add_argument(
        "--dataset", type=str, default="conv_go_awry",
        choices=["conv_go_awry"], help="dataset used for experiment"
    )
    args = parser.parse_args()
    if args.dataset == "conv_go_awry":
        args.inputpath = "css_data/aqua_zero_shot_cot"
        args.rantionalepath = "css_data/aqua_rationale_cot"
        args.answerpath = "css_data/aqua_answer_cot"
        
        args.direct_answer_trigger_for_zeroshot_cot = "Predict whether the given conversation has a personal attack (True or False)."
    else:
        raise ValueError("dataset is not properly defined ...")
    args.session_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJqaWFhb2NAYWxsZW5haS5vcmciLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZ2VvaXBfY291bnRyeSI6IlVTIn0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJ1c2VyX2lkIjoidXNlci1JWVpYSEFrdVJKTXJrVXlLU2RSbFZEWWkifSwiaXNzIjoiaHR0cHM6Ly9hdXRoMC5vcGVuYWkuY29tLyIsInN1YiI6ImF1dGgwfDYwOWFjMWY0NGMxZjQ1MDA3MDQwYmExZiIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2NzY0OTA3NjAsImV4cCI6MTY3NzcwMDM2MCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvZmZsaW5lX2FjY2VzcyJ9.XFsYqMo1JpK58MYk0QzqkuIn2bTfknFzjBGkYFHznPj-dQjgHuyxB6HwgznSj7jYa2hmloBMK3FxV3peXQ5aLiqfh0QIBgHWUlr3CSCm2ypB82V8HjcgN-18WYlACIg_w7im7xYmMv3_1iRGWyq4d1-8vzxgtADrthqPNcjaib3nPwj9RzYOdcV6fZd4n54MqcuXn2l-Yge0weB539GvBRkinCmEbcNJZKJ3VYQu6EiO0t_MzRodCOLnD-auZBfs-sbyVMuRH65RSjIqVsdhp8S_f2gmTaMs4MRU2CC0b8QX-3mVFZmhRHhUYA5TEaEaHT8Y83AA0j3C6erwx-gMpg"
    return args

def main():
    print("We are using chatgpt to test different datasets now!\n")
    args = parse_arguments()
    configure(args)

errortime = 0
if __name__ == "__main__":
    main()
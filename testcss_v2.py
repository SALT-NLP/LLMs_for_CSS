import json
import os
from os.path import exists
from os import getenv
from sys import argv, exit
import time
import re
import random
import argparse
from revChatGPT.V1 import Chatbot


def data_split():
    pass



def get_answers(input_path, output_path, args):
    print("###### Getting Answers! ######")
    chatbot = Chatbot(config={"access_token":args.access_token})
    
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    count = len(raw_data['Label'])
    print("###### Number of Data: ", count, " ######")
    allflag = [0 for i in range(count)]  
    
    if not os.path.exists(outputpath):
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
        
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            conversations = raw_data['context']
            i = 0
            for (u,v) in conversations.items():
                if allflag[i] == 1:
                    i += 1
                    continue
                test_samples.append(v)
                gold_label.append(raw_data['Label'][u])
                touseindex.append(i)
                i += 1

        inputprompts = []
        for i in range(len(test_samples)):
            oneres = test_samples[i]
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
                print(touseindex[i], gold_label[i], touseresponse)
                fw.write(str(touseindex[i]) + "\t" + str(gold_label[i]) + "\t" + str(touseresponse) + "\n")
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
    with open(args.input_path, 'r') as f:
        a = json.load(f)
    label_set = set([v.lower() for (u,v) in a['labels'].items()])
    print('###### Label Space:', label_set)
    
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
        
        if args.dataset in ['conv_go_awry', 'wiki_corpus']:
            gold = content[1].lower()
            pred = content[2].lower()
            print(gold, pred)
            if gold in pred:
                accnum += 1
        else:
            pass
    
    print("\n ###### Results ###### \n")
    print("Acc: ",  float(accnum) / float(allnum))
    print('umber of Correct Data: ', accnum)
    print("Number of Testing Data: ",  allnum)





def parse_arguments():
    parser = argparse.ArgumentParser(description="chatgpt-zero-shot-css")

    parser.add_argument(
        "--dataset", type=str, default="conv_go_awry",
        choices=["conv_go_awry", "wiki_corpus"], help="dataset used for experiment"
    )
    args = parser.parse_args()
    if args.dataset == "conv_go_awry":
        # To update this dataset
        args.input_path = "css_data/conversations-gone-awry-corpus/raw_data.json"
        args.answer_path = "css_data/conversations-gone-awry-corpus/answer"
    elif args.dataset == "wiki_corpus":
        args.input_path = "css_data/wiki_corpus/power.json"
        args.answer_path = "css_data/wiki_corpus/answer"
        
    else:
        raise ValueError("dataset is not properly defined ...")
    
    
    # substitute this with your own access token!
    args.access_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJqaWFhb2NAYWxsZW5haS5vcmciLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZ2VvaXBfY291bnRyeSI6IlVTIn0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJ1c2VyX2lkIjoidXNlci1JWVpYSEFrdVJKTXJrVXlLU2RSbFZEWWkifSwiaXNzIjoiaHR0cHM6Ly9hdXRoMC5vcGVuYWkuY29tLyIsInN1YiI6ImF1dGgwfDYwOWFjMWY0NGMxZjQ1MDA3MDQwYmExZiIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2NzY0OTA3NjAsImV4cCI6MTY3NzcwMDM2MCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvZmZsaW5lX2FjY2VzcyJ9.XFsYqMo1JpK58MYk0QzqkuIn2bTfknFzjBGkYFHznPj-dQjgHuyxB6HwgznSj7jYa2hmloBMK3FxV3peXQ5aLiqfh0QIBgHWUlr3CSCm2ypB82V8HjcgN-18WYlACIg_w7im7xYmMv3_1iRGWyq4d1-8vzxgtADrthqPNcjaib3nPwj9RzYOdcV6fZd4n54MqcuXn2l-Yge0weB539GvBRkinCmEbcNJZKJ3VYQu6EiO0t_MzRodCOLnD-auZBfs-sbyVMuRH65RSjIqVsdhp8S_f2gmTaMs4MRU2CC0b8QX-3mVFZmhRHhUYA5TEaEaHT8Y83AA0j3C6erwx-gMpg"
    return args


def main():
    print("We are using chatgpt to test different datasets now!\n")
    args = parse_arguments()
    
    try:
        inputpath = args.inputpath
        answerpath = args.answerpath
        
        get_answers(inputpath, answerpath, args)
        
        calculateres(answerpath, args)  
        
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
    print('###### Execution Time:', elapsed_time, ' seconds. ######')
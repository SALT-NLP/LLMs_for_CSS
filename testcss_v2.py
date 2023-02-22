import json
import os
import pandas as pd
from os.path import exists
from os import getenv
from sys import argv, exit
import time
import re
import random
import argparse
from revChatGPT.V1 import Chatbot
from sklearn.metrics import classification_report



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
    indexes = raw_data['context'].keys()
    
    num_testing = args.testing_size
    random.seed(0)
    selected_indexs = random.sample(indexes, num_testing)
    
    for u in selected_indexs:
        contexts.append(raw_data['context'][u])
        labels.append(raw_data['labels'][u])
        prompts.append(raw_data['prompts'][u])
    
    testing_data = {"context": contexts, "labels": labels, "prompts": prompts}
    data_f = pd.DataFrame.from_dict(testing_data)
    data_f.to_json(input_path)
    

def get_response(chatbot, allprompts):
    global errortime
    allresponse = []
    i = 0
    while i < len(allprompts):
        oneprompt = allprompts[i]
        #print(oneprompt)
        try:
            
            response = ""
            for data in chatbot.ask(oneprompt):
                response = data["message"]
            print("######Response#####", response)
            
            if len(response) <2:
                i += 1
                allresponse.append("Error!")
                chatbot.reset_chat()
                continue

            allresponse.append(response)
            i += 1
            errortime = 0
            chatbot.reset_chat()
        except Exception as exc:
            print(f"Data point {i} went wrong!")
            print(exc)
            
            allresponse.append("Error!")
            errortime += 1
            if errortime > 60:
                print("Error too many times! sleep 600s")
                errortime = 0
                time.sleep(600)
            i += 1
        time.sleep(6)
    return allprompts, allresponse

def get_answers(input_path, output_path, args):
    print("###### Getting Answers! ######")
    chatbot = Chatbot(config={"access_token":args.access_token})
    
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    count = len(raw_data['labels'])
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
            conversations = raw_data['context']
            i = 0
            for (u,v) in conversations.items():
                if allflag[i] == 1:
                    i += 1
                    continue
                test_samples.append(v)
                gold_label.append(raw_data['labels'][u])
                prompts.append(raw_data['prompts'][u])
                touseindex.append(i)
                i += 1

        input_prompts = []
        for i in range(len(test_samples)):
            oneres = test_samples[i]
            input_prompts.append(oneres + ' ' + prompts[i])
            
        fw = open(output_path, "a+", encoding="utf-8")
        response = []
        for i in range(len(input_prompts)):
            _, oneresponse = get_response(chatbot, [input_prompts[i]])
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
        #end = time.time()
        #print("all used time: ", end - start)

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
    label_set = set([str(v).lower() for (u,v) in a['labels'].items()])
    print('###### Label Space:', label_set)
    label_dict = {'None':0}
    
    i = 1
    for u in label_set:
        label_dict[u] = i
        i += 1
    
    f = open(path, 'r', encoding="utf-8")
    allnum = 0
    accnum = 0
    
    
    preds = []
    golds = []
    target_names = list(label_dict.keys())
    
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        content = oneline.split('\t')
        if len(content) != 3:
            continue
        index = int(content[0])
        allnum += 1
        
        if args.dataset in ['conv_go_awry', 'wiki_corpus', 'reddit_humor']:
            #print(content[1])
            gold = content[1].lower()
            pred = content[2].lower()
            print(gold, pred)
            if gold in pred:
                accnum += 1
        elif args.dataset == 'implicit_hate':
            gold = label_dict[content[1].lower()]
            pred = content[2].lower()
            for u in label_set:
                if u in pred:
                    pred = label_dict[u]
                    break
                pred = 0
            if gold == pred:
                accnum += 1
            golds.append(gold)
            preds.append(pred)
                
        else:
            pass
        
    
    print("\n ###### Results ###### \n")
    print("Acc: ",  float(accnum) / float(allnum))
    print('Number of Correct Data: ', accnum)
    print("Number of Testing Data: ",  allnum)
    
    if len(preds) > 0:
        print(classification_report(golds, preds, target_names=target_names))





def parse_arguments():
    parser = argparse.ArgumentParser(description="chatgpt-zero-shot-css")

    parser.add_argument(
        "--dataset", type=str, default="conv_go_awry",
        choices=["conv_go_awry", "wiki_corpus", "implicit_hate", "reddit_humor"], help="dataset used for experiment"
    )
    
    args = parser.parse_args()
    if args.dataset == "conv_go_awry":
        args.raw_datapath = "css_data/conv_go_awry/toxicity.json"
        args.input_path = "css_data/conv_go_awry/test.json"
        args.answer_path = "css_data/conv_go_awry/answer"
    elif args.dataset == "wiki_corpus":
        args.raw_datapath = "css_data/wiki-corpus/power.json"
        args.input_path = "css_data/wiki-corpus/test.json"
        args.answer_path = "css_data/wiki-corpus/answer"
    elif args.dataset == "implicit_hate":
        args.raw_datapath = "css_data/implicit_hate/hate.json"
        args.input_path = "css_data/implicit_hate/test.json"
        args.answer_path = "css_data/implicit_hate/answer"
    elif args.dataset == "reddit_humor":
        args.raw_datapath = "css_data/reddit_humor/humor.json"
        args.input_path = "css_data/reddit_humor/test.json"
        args.answer_path = "css_data/reddit_humor/answer"
    
        
    else:
        raise ValueError("dataset is not properly defined ...")
    
    
    # substitute this with your own access token!
    args.testing_size = 500
    args.access_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJqaWFhb2NAYWxsZW5haS5vcmciLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZ2VvaXBfY291bnRyeSI6IlVTIn0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJ1c2VyX2lkIjoidXNlci1JWVpYSEFrdVJKTXJrVXlLU2RSbFZEWWkifSwiaXNzIjoiaHR0cHM6Ly9hdXRoMC5vcGVuYWkuY29tLyIsInN1YiI6ImF1dGgwfDYwOWFjMWY0NGMxZjQ1MDA3MDQwYmExZiIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2NzY0OTA3NjAsImV4cCI6MTY3NzcwMDM2MCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvZmZsaW5lX2FjY2VzcyJ9.XFsYqMo1JpK58MYk0QzqkuIn2bTfknFzjBGkYFHznPj-dQjgHuyxB6HwgznSj7jYa2hmloBMK3FxV3peXQ5aLiqfh0QIBgHWUlr3CSCm2ypB82V8HjcgN-18WYlACIg_w7im7xYmMv3_1iRGWyq4d1-8vzxgtADrthqPNcjaib3nPwj9RzYOdcV6fZd4n54MqcuXn2l-Yge0weB539GvBRkinCmEbcNJZKJ3VYQu6EiO0t_MzRodCOLnD-auZBfs-sbyVMuRH65RSjIqVsdhp8S_f2gmTaMs4MRU2CC0b8QX-3mVFZmhRHhUYA5TEaEaHT8Y83AA0j3C6erwx-gMpg"
    return args


def main():
    print("We are using chatgpt to test different datasets now!\n")
    args = parse_arguments()
    
    try:
        input_path = args.input_path
        answer_path = args.answer_path
        raw_datapath = args.raw_datapath
        
        data_split(raw_datapath, input_path, args)
        
        get_answers(input_path, answer_path, args)
        
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
    print('###### Execution Time:', elapsed_time, ' seconds. ######')
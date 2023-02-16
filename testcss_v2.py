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












def parse_arguments():
    parser = argparse.ArgumentParser(description="chatgpt-zero-shot-css")

    parser.add_argument(
        "--dataset", type=str, default="conv_go_awry",
        choices=["conv_go_awry"], help="dataset used for experiment"
    )
    args = parser.parse_args()
    if args.dataset == "conv_go_awry":
        args.inputpath = "css_data/conversations-gone-awry-corpus/raw_data.json"
        args.answerpath = "css_data/conversations-gone-awry-corpus/answer"
    else:
        raise ValueError("dataset is not properly defined ...")
    
    
    # substitute this with your own access token!
    args.access_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJqaWFhb2NAYWxsZW5haS5vcmciLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZ2VvaXBfY291bnRyeSI6IlVTIn0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJ1c2VyX2lkIjoidXNlci1JWVpYSEFrdVJKTXJrVXlLU2RSbFZEWWkifSwiaXNzIjoiaHR0cHM6Ly9hdXRoMC5vcGVuYWkuY29tLyIsInN1YiI6ImF1dGgwfDYwOWFjMWY0NGMxZjQ1MDA3MDQwYmExZiIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2NzY0OTA3NjAsImV4cCI6MTY3NzcwMDM2MCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvZmZsaW5lX2FjY2VzcyJ9.XFsYqMo1JpK58MYk0QzqkuIn2bTfknFzjBGkYFHznPj-dQjgHuyxB6HwgznSj7jYa2hmloBMK3FxV3peXQ5aLiqfh0QIBgHWUlr3CSCm2ypB82V8HjcgN-18WYlACIg_w7im7xYmMv3_1iRGWyq4d1-8vzxgtADrthqPNcjaib3nPwj9RzYOdcV6fZd4n54MqcuXn2l-Yge0weB539GvBRkinCmEbcNJZKJ3VYQu6EiO0t_MzRodCOLnD-auZBfs-sbyVMuRH65RSjIqVsdhp8S_f2gmTaMs4MRU2CC0b8QX-3mVFZmhRHhUYA5TEaEaHT8Y83AA0j3C6erwx-gMpg"
    return args

def configure(args):
    try:
        inputpath = args.inputpath
        answerpath = args.answerpath
        getanswers(inputpath, answerpath, args)
        calculateres(answerpath,args)
    except KeyboardInterrupt:
        print("\n Key Interruptions! Goodbye!")
        exit()
    except Exception as exc:
        print(exc)
        exit()

def main():
    print("We are using chatgpt to test different datasets now!\n")
    args = parse_arguments()
    configure(args)

errortime = 0
if __name__ == "__main__":
    st = time.time()
    main()
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
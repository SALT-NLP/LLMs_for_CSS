# ChatNLP: a ChatGPT Baseline on NLP Benchmarks

Current Progress:

-> AQuA exp in progress

zzs:

| Dataset  | Rationale | Answer  | Accuracy  | 
| -------- | --------- | ------- | ------- |
| Date     | 369/369   | 369/369 | 66.1  |
| Coin     | 500/500   | 500/500 | 69.6   |
| SingleEq | 508/508    | 508/508 |  91.3  |
| StrategyQA | 2290/2290    | 2290/2290 |  62.5  |

## Steps to use testchatgpt.py

1. install revChatGPT and dependencies
```   
pip install revChatGPT==0.0.43
python3 -m playwright install
```

2. create ~/.config/revChatGPT/config.json and populate your token as shown below

```
{
    "session_token": "<YOUR TOKEN>",
    "accept_language": "en-US,en"
}
```
Refer to https://github.com/acheong08/ChatGPT/wiki/Setup for how to obtain the session token.

3. usage:
```
python testchatgpt.py
```

A Chrome/Chromium/Firefox window will show up and close automatically.

FAQ:
Q: Error "Looks like you launched a headed browser without having a XServer running"
A: Use a local machine with Internet Browsers

Q: Error "Exception: cf challenge fail"
A: Turn off VPN if necessary.



if you want to use multiple threads,

```
python testchatgpt.py --threading
```

the default thread number is 16 (line 185)

if you want to obtain text output rather than stream output from chatgpt,

```
python testchatgpt.py --text
```
## Steps to run run_chat.py (reasoning tasks)

Prerequisite:

```
pip install torch
pip install openai
```


```python run_chat.py --dataset multiarith --demo_path demos/multiarith_auto --output_dir experiment/multiarith_auto```

if exceptions encountered, it is possible to resume the experiments by using the argument ```--resume_id```

```python run_chat.py --dataset multiarith --demo_path demos/multiarith_auto --output_dir experiment/multiarith_auto --resume_id 89``` if there are 88 lines of results stored in ```experiment/multiarith_auto```

## About Mutual

In ```experiments```, we have the text-davinci-003 results (few-shot and zero-shot) on the Mutual dev set (100 random samples).

Each line is composed of the following contents:

```
{
    "question": <prompted question>
    "gold_ans": <gold answer (A|B|C|D)>
    "rationale": <output from GPT-3>
    "pred_ans": <parsed answer (A|B|C|D)>
    "wrap_que": <complete input (question, choices, prompts)>
}
```

When testing other models (e.g., ChatGPT), you can directly feed the model with "wrap_que", parse the output ```pred = re.findall(r'A|B|C|D', output.strip())```, and calculate the accuracy after comparing the answer with the gold one (```cal_acc.py```).

Accuracy of text-davinci-003: 

Zero-Shot: 72.0%, Few-Shot (w/ 4 random demos from the training set): 81.0%

Accuracy of chatgpt: 

Zero-Shot: 76.0%, Few-Shot (w/ 4 random demos from the training set): 68.0%


## 0127 update

To run chatgpt zero-cot, please follow the instructions below.

1. install revChatGPT and dependencies
```   
pip3 install revChatGPT==0.1.1
```
I use the 0.1.1 version. You can also try others if this version does not work.

2. create ./config.json and populate your token as shown below

```
{
    "session_token": "<YOUR TOKEN>"
}
```
Refer to https://github.com/acheong08/ChatGPT/wiki/Setup for how to obtain the session token.

3. usage:
```
python testreasoning.py --dataset task_name
```

It will first generate all rationales, then generate all answers.




## 0201 update

To use official api through your api-key, please follow the instructions below.

1. install revChatGPT and dependencies
```   
pip3 install --upgrade revChatGPT
```

2. modify args.api_key in testreasoning.py to your own api-key

```
https://platform.openai.com/account/api-keys
```

3. usage:
```
python testreasoning.py --dataset task_name
```

you can modify time.sleep(x) based on your own experience

It will first generate all rationales, then generate all answers.


## CSS update


To run chatgpt zero-cot, please follow the instructions below.

1. install revChatGPT and dependencies
```   
pip3 install --upgrade revChatGPT
```

2. substitute the arg.access_token with your own.

Refer to: https://chat.openai.com/api/auth/session

3. usage:
```
python testcss.py --dataset conv_go_awry
```

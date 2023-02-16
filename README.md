# SocialChatGPT

## How to run testing with ChatGPT?

1. Install ConvoKit
```
git clone https://github.com/CornellNLP/ConvoKit.git
cd ConvoKit
pip3 install -e .
```

2. Download the datasets:
```
python data_loader.py -d power --save_dir ./css_data/wiki-corpus
```

3. install revChatGPT and dependencies
```   
pip3 install --upgrade revChatGPT
```

4. substitute the arg.access_token with your own.

Refer to: https://chat.openai.com/api/auth/session


5. usage:
```
python testcss_v2.py --dataset wiki_corpus
```

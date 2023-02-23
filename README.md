# SocialChatGPT

## How to run the testing with ChatGPT?

1. Install ConvoKit
```
git clone https://github.com/CornellNLP/ConvoKit.git
cd ConvoKit
pip3 install -e .
```

2. Download the datasets and pre-process the datasets:
```
python data_loader.py -d power --save_dir ./css_data/wiki-corpus
```

3. Install revChatGPT and dependencies
```   
pip3 install --upgrade revChatGPT
```

4. Create a config.py and substitute the config_access_token with your own in the config.py.

```
config_access_token = "*"
# keep your own copy
```

Refer to: https://chat.openai.com/api/auth/session


5. Usage:
```
python testcss_v2.py --dataset wiki_corpus
```

# LLMs for CSS

## How to run testing?

1. Install ConvoKit
```
git clone https://github.com/CornellNLP/ConvoKit.git
cd ConvoKit
pip3 install -e .
```

2. Download the datasets and pre-process the datasets:
```
python data_loader.py -d power --save_dir ./css_data/wiki_corpus
```

3. Install dependencies
```   
pip3 install -r requirements.txt
```

4. Add your OpenAI Key to your environment.

5. Usage:
```
python test_official_chat_css --model [MODEL_NAME_HERE] --dataset wiki_corpus
```

We evaluated the following models - but any model which can be loaded with HuggingFace AutoModelForSeq2SeqLM should work out of the box.
```
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
```

## File Roadmap
`mappings.py` - Configuration used for each dataset in the paper. Describes the type of dataset, how it should be processed from the raw format, and how the task should be formatted into a prompt from our prompting guidelines.

`data_loader.py` - Downloads and Converts Raw Datasets into the Seq2Seq format used by LLMs.

`test_official_chat_css.py` - Runs zero-shot LLM of choice - contains code for HuggingFace, ChatGPT API, and Traditional GPT API.

`eval_significance.py` - Computes Pairwise Bootstrap significance between the answer files of two models.

`eval_agreement.py` - Computes the Kappa between the LLM and the gold labels.

## Citation
If you find this work useful, please cite it as follows!
```
@article{salt-2023-llms-for-css,
  title = {Can Large Language Models Transform Computational Social Science?},
  author = {Ziems, Caleb and Held, William and Shaikh, Omar and Chen, Jiaao and Zhang, Zhehao and Yang, Diyi},
  journal = {arXiv submission 4840038},
  year = {2023},
  month = apr,
}
```



import pandas as pd
import numpy as np
import json, os, re
import argparse
from tqdm import tqdm
from glob import glob
from statsmodels.stats.inter_rater import fleiss_kappa

DATASETS = ["discourse",
            "conv_go_awry",
            "power",
            "hate",
            "humor",
            "flute-classification",
            "persuasion",
            "politeness",
            "media_ideology",
            "indian_english_dialect",
            "ibc",
            "semeval_stance",
            "tempowic",
            "mrf-classification",
            "talklife",
            "emotion",
            "raop"]
MODELS = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "google/flan-ul2",
        "text-ada-001",    
        "text-babbage-001",
        "text-curie-001",
        "text-davinci-001",    
        "text-davinci-002",
        "text-davinci-003",
        "chatgpt",
]

MAPPINGS = {
       "power": {
                "true": "yes",
                "false": "no",
            },
       "persuasion": {
                "1.0": "True",
                "0.0": "False",
            },
       "conv_go_awry": {
                "true": "yes",
                "false": "no",
            },
       "mrf-classification": {
                "misinformation": "A",
                "trustworthy": "B",
            },
       "politeness": {
                "1": "A",
                "0": "B",
                "-1": "C",
            },
       "persuasion": {
                "1.0": "True",
                "0.0": "False",
            },
       "flute-classification": {
                "idiom": "A",
                "metaphor": "B",
                "sarcasm": "C",
                "simile": "D",
            },
       "media_ideology": {
                "left": "A",
                "right": "B",
                "center": "C",
            },
       "tempowic": {
                "same": "A",
                "different": "B",
            },
       "semeval_stance": {"against": "A", "favor": "B", "none": "C"},
       "ibc": {
                "liberal": "A",
                "conservative": "B",
                "neutral": "C",
            },
       "hate": {
                "white_grievance": "A",
                "incitement": "B",
                "inferiority": "C",
                "irony": "D",
                "stereotypical": "E",
                "threatening": "F",
            },
       "discourse": {
                "question": "A",
                "answer": "B",
                "agreement": "C",
                "disagreement": "D",
                "appreciation": "E",
                "elaboration": "F",
                "humor": "G",
            },
        "indian_english_dialect": {
                "preposition omission": "R",
                "copula omission": "B",
                "resumptive subject pronoun": "S",
                "resumptive object pronoun": "T",
                "extraneous article": "D",
                "focus only": "F",
                "mass nouns as count nouns": "N",
                "stative progressive": "U",
                "lack of agreement": "K",
                "none of the above": "W",
                "lack of inversion in wh-questions": "L",
                "topicalized non-argument constituent": "V",
                "inversion in embedded clause": "J",
                "focus itself": "E",
                'general extender "and all"': "G",
                '"general extender ""and all"""': "G",
                "object fronting": "P",
                'invariant tag "isn’t it, no, na"': "I",
                '"invariant tag ""isn’t it, no, na"""': "I",
                "habitual progressive": "H",
                "article omission": "A",
                "prepositional phrase fronting with reduction": "Q",
                'non-initial existential "is / are there"': "O",
                '"non-initial existential ""is / are there"""': "O",
                "left dislocation": "M",
                "direct object pronoun drop": "C",
            },
}

def clean(txt, mapping={}):
    c = str(txt).replace('&', '').lower().strip()
    if c in mapping:
        return mapping[c].lower()
#     else:
#         print(txt, c, mapping)
    return c.lower()

def build_annotation_matrix(df, idx='idx'):
    
    labels = set()
    columns = []
    for col in df.columns:
        if col == idx:
            continue
        else:
            labels.update(set( [x for x in df[col].values]))
            columns.append(col)
    labels = list(sorted(labels))
    #print(labels)
    M = np.zeros((len(df), len(labels)))
    i = 0
    for _, row in df.iterrows():
        for col in columns: 
            M[i, labels.index( row[col]) ] += 1
        i+=1
    return M

def main(args):
    if args.dataset == "conv_go_awry":
        args.raw_datapath = "css_data/conv_go_awry/toxicity.json"
        args.input_path = "css_data/conv_go_awry/test.json"
        args.answer_path = "css_data/conv_go_awry/answer"
    elif args.dataset == "wikievents":
        args.raw_datapath = "css_data/wikievents/wikievents.json"
        args.input_path = "css_data/wikievents/test.json"
        args.answer_path = "css_data/wikievents/answer"
    elif args.dataset == "power":
        args.raw_datapath = "css_data/wiki_corpus/power.json"
        args.input_path = "css_data/wiki_corpus/test.json"
        args.answer_path = "css_data/wiki_corpus/answer"
    elif args.dataset == "hate":
        args.raw_datapath = "css_data/implicit_hate/hate.json"
        args.input_path = "css_data/implicit_hate/test.json"
        args.answer_path = "css_data/implicit_hate/answer"
    elif args.dataset == "discourse":
        args.raw_datapath = "css_data/discourse/discourse.json"
        args.input_path = "css_data/discourse/test.json"
        args.answer_path = "css_data/discourse/answer"
    elif args.dataset == "humor":
        args.raw_datapath = "css_data/reddit_humor/humor.json"
        args.input_path = "css_data/reddit_humor/test.json"
        args.answer_path = "css_data/reddit_humor/answer"
    elif args.dataset == "persuasion":
        args.raw_datapath = "css_data/persuasion/persuasion.json"
        args.input_path = "css_data/persuasion/test.json"
        args.answer_path = "css_data/persuasion/answer"
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
    elif args.dataset == "politeness":
        args.raw_datapath = "css_data/wiki_politeness/politeness.json"
        args.input_path = "css_data/wiki_politeness/test.json"
        args.answer_path = "css_data/wiki_politeness/answer"
    elif args.dataset == "media_ideology":
        args.raw_datapath = "css_data/media_ideology/media_ideology.json"
        args.input_path = "css_data/media_ideology/test.json"
        args.answer_path = "css_data/media_ideology/answer"
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
    elif args.dataset == "talklife":
        args.raw_datapath = "css_data/talklife/talklife.json"
        args.input_path = "css_data/talklife/test.json"
        args.answer_path = "css_data/talklife/answer"
    elif args.dataset == "raop":
        args.raw_datapath = "css_data/raop/raop.json"
        args.input_path = "css_data/raop/test.json"
        args.answer_path = "css_data/raop/answer"
    elif args.dataset == "emotion":
        args.raw_datapath = "css_data/emotion/emotion.json"
        args.input_path = "css_data/emotion/test.json"
        args.answer_path = "css_data/emotion/answer"
    elif args.dataset == "mrf-explanation":
        args.raw_datapath = "css_data/mrf/mrf-explanation.json"
        args.input_path = "css_data/mrf/test-explanation.json"
        args.answer_path = "css_data/mrf/answer-explanation"
    elif args.dataset == "mrf-classification":
        args.raw_datapath = "css_data/mrf/mrf-classification.json"
        args.input_path = "css_data/mrf/test-classification.json"
        args.answer_path = "css_data/mrf/answer-classification"
    elif args.dataset == "tropes":
        args.raw_datapath = "css_data/tropes/tropes.json"
        args.input_path = "css_data/tropes/test.json"
        args.answer_path = "css_data/tropes/answer"
    elif args.dataset == "positive_reframing":
        args.raw_datapath = "css_data/positive_reframing/positive_reframing.json"
        args.input_path = "css_data/positive_reframing/test.json"
        args.answer_path = "css_data/positive_reframing/answer"
    else:
        raise ValueError("dataset is not properly defined ...")
    if args.model == "chatgpt" or "text-" in args.model:
        args.answer_path = args.answer_path + "-" + args.model
    elif "flan" in args.model:
        args.answer_path = args.answer_path + "-" + args.model.split("/")[-1]
        
    mapping = {}
    if args.dataset in MAPPINGS:
        mapping = MAPPINGS[args.dataset]
    else:
        print(args.dataset, 'not in MAPPINGS')
    try:
        df = pd.read_csv(args.answer_path, sep='\t', names=['idx', 'gold', 'pred'], on_bad_lines='skip')
    except:
        print(args.answer_path)
        return {'accuracy': None, "kappa": None}
    #if args.dataset=='tropes': print(df.head())
    df['gold'] = [clean(x, mapping) for x in df.gold]
    df['pred'] = [clean(x, mapping) for x in df.pred]
    #if args.dataset=='tropes': print(df.head())
    M = build_annotation_matrix(df)
    if len(M):
        kappa = fleiss_kappa(M, method='fleiss')
        acc = sum(df['gold']==df['pred'])/len(df)
        if args.verbose:
            print("Dataset:", args.dataset)
            print("Model:", args.model)
            print("Accuracy:", acc)
            print("Fleiss Kappa:", kappa)
            print('--------------')
        return {'accuracy': acc, "kappa": kappa}
    else:
        return {'accuracy': None, "kappa": None}
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval_agreement")
    parser.add_argument(
            "--dataset",
            type=str,
            default="all",
            choices=["all"]+DATASETS,
            help="dataset used for experiment",
        )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all",
        choices=["all"]+MODELS,
    )
    args = parser.parse_args()
    args.verbose = False
    results = {}
    if args.model=="all" or args.dataset=="all":
        for j, dataset in enumerate(DATASETS):
            for model in MODELS:
                args.model = model
                args.dataset = dataset
                r = main(args)
                results[str((args.dataset, args.model))] = r
    else:
        r = main(args)
        results[str((args.dataset, args.model))] = r
    #print(results)
    with open('results/clf.json', 'w') as outfile:
        json.dump(results, outfile)
import bert_score
import sacrebleu
from bleurt import score
import pandas as pd
import numpy as np
import json, os, re
import evaluate
import argparse
from ast import literal_eval
from tqdm import tqdm
from glob import glob
from nltk.stem.porter import PorterStemmer

def clean_generation(gen):
    return gen.replace("&", "")

def get_cands_refs(args, split_refs=False, list_cands=False, clean=clean_generation):
    cand_list = []
    refs_list = []
    
    literalref = True
    label = "labels"
    if args.dataset in {"flute-explanation"}:
        literalref = False
        label = "additional_labels"
        
    R = lambda txt: literal_eval(txt)
    if not literalref:
        R = lambda txt: txt
        
    C = lambda cand: clean(cand)
    if list_cands:
        C = lambda cand: [clean(cand)]
        
    with open(args.raw_datapath, "r") as f:
        a = json.load(f)

    f = open(args.answer_path, "r", encoding="utf-8")
    
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        content = oneline.split("\t")
        if len(content) != 3:
            continue
            
        index = content[0]
        
        refs = R(content[1]) # R(a[label][index]) #R(content[1]) #
        cand = content[2].lower()
    
        if split_refs:
            for ref in refs:
                cand_list.append(C(cand))
                refs_list.append(ref)
        else:
            cand_list.append(C(cand))
            refs_list.append(refs)
    return cand_list, refs_list

def score_max(cands, 
              refs, 
              scoring_function=lambda cand, ref: sacrebleu.sentence_bleu(cand, ref).score,
             ):
    max_scores = []
    for cand, refs in tqdm(zip(cands, refs), total=len(cands)):
        scores = []
        for ref in refs:
            score = scoring_function(cand, ref)
            scores.append(score)
        max_scores.append(max(scores))
    return max_scores

def calculateres_gen(path, args, cands=None, refs=None):
    
    bleurt_scorer = score.BleurtScorer()
    stemmer = PorterStemmer()
    def stem(txt):
        return ' '.join([stemmer.stem(w) for w in txt.split(' ')])
    
    metrics = ["sacrebleu", "bleurt", "bertscore"]
    scoring_functions = [
        lambda cand, ref: sacrebleu.sentence_bleu(stem(cand), [stem(ref)]).score,
        lambda cand, ref: bleurt_scorer.score(references=[ref], candidates=[cand]),
        lambda cand, ref: bert_score.score(cands, refs, lang="en", batch_size=1, device=1)[-1].numpy()
    ]

    scores = {}
    if (not cands) or (not refs):
        cands, refs = get_cands_refs(args)
    print("Candidate:", cands[0])
    print("Reference:", refs[0][0])
    for metric, scoring_function in zip(metrics, scoring_functions):
        if metric in {'bertscore'}: # batch max already implemented
            score_list = scoring_function(cands, refs)
        else:
            score_list = score_max(cands, refs, scoring_function)
            if metric == "bleurt":
                del bleurt_scorer
        
        scores[metric] = str(np.mean(score_list))
        print(metric, scores[metric])#, score_list)

    return scores
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval_generation")
    parser.add_argument("--dataset", type=str, default="sbic", choices=["sbic", "mrf", "flute"])
    args = parser.parse_args()
    args.clean_path = f"hit/input/{args.dataset}"
    args.pred = "Generated"
    args.literal_eval = False
    if args.dataset == "sbic":
        args.gold = "targetStereotype"
        args.literal_eval = True
    elif args.dataset == "mrf":
        args.gold = "writer_intent"
        args.literal_eval = True
    elif args.dataset == "flute":
        args.gold = "additional_labels"
        
    results = {}
    for fn in glob(os.path.join(args.clean_path, "*")):
        
        model = fn.split("_")[-1][:-4]
        
        df = pd.read_csv(fn)
        
        cands = list(df[args.pred].values)
        refs = list(df[args.gold].values)
        
        if args.literal_eval:
            refs = [literal_eval(line) for line in df[args.gold].values] 
        
        print(fn, model)

        results[model] = calculateres_gen("", args, cands=cands, refs=refs)
        
    with open(f"results/{args.dataset}.json", "w") as outfile:
        json.dump(results, outfile)
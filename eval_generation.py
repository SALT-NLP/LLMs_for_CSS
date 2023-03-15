import bert_score
import sacrebleu
from bleurt import score
import pandas as pd
import numpy as np
import json
from ast import literal_eval

def clean_generation(gen):
    return gen.replace("&", "")

def get_cands_refs(args, split_refs=False, list_cands=False, clean=clean_generation):
    cand_list = []
    refs_list = []
    
    C = lambda cand: clean(cand)
    if list_cands:
        C = lambda cand: [clean(cand)]
        
    path = args.answer_path
    with open(args.input_path, "r") as f:
        a = json.load(f)

    f = open(path, "r", encoding="utf-8")
    
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        content = oneline.split("\t")
        if len(content) != 3:
            continue
            
        refs = literal_eval(content[1])
        cand = content[2]
    
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
    print('cand:', cands[0])
    print('ref:', refs[0][0])
    print('score:', scoring_function(cands[0], refs[0][0]))
    max_scores = np.array([ max([scoring_function(cand, ref)
                                     for ref in refs
                                ])
                               for cand, refs in zip(cands, refs)
                          ])
    return max_scores

def calculateres_gen(path, args):
    bleurt_scorer = score.BleurtScorer()

    metrics = ["sacrebleu", "bleurt", "bertscore"]
    scoring_functions = [
        lambda cand, ref: sacrebleu.sentence_bleu(cand, [ref]).score,
        lambda cand, ref: bleurt_scorer.score(references=[ref], candidates=[cand]),
        lambda cand, ref: bert_score.score([cand], [ref], lang="en", batch_size=1),
    ]

    scores = {}
    cands, refs = get_cands_refs(args)
    for metric, scoring_function in zip(metrics, scoring_functions):
        score_list = score_max(cands, refs, scoring_function)
        scores[metric] = score_list        
        print(metric, np.mean(scores[metric]))#, score_list)

    print(scores)
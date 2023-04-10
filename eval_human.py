import argparse
import pandas as pd
import numpy as np
from glob import glob
from collections import Counter


BLACKLIST = set(["A1FHH0LTJMMRL"])

def get_ranking(row):
    ranks = ["rank_1", "rank_2", "rank_3", "rank_4"]
    models = ["Model_1", "Model_2", "Model_3", "Model_4"]
    for i in range(0, 4):
        lbl = row[f"Answer.gen_{i+1}_statement_correct"]
        model = row[f"Input.Model_{i+1}"]
        models[ranks.index(lbl)] = model
    return models

def get_rankings(df):
    rankings = []
    for _, row in df.iterrows():
        rankings.append(get_ranking(row))
    return rankings

def main():
    evals = pd.concat([pd.read_csv(fn) for fn in glob("hit/output/*balanced*.csv")])
    
    names = ['baseline',
     'text-ada-001',         
     'text-babbage-001',
     'text-curie-001',
     'text-davinci-001',         
     'text-davinci-002',         
     'text-davinci-003',
     'chatgpt',
     'human']

    tasks = ['flute', 'mrf', 'sbic', 'positive_reframing']

    results = {task: { 
        m: {n: Counter() for n in names if m!=n} for m in names
    } for task in tasks}

    c=0
    for hid in evals["HITId"].values:
        consider = evals[evals["HITId"]==hid].copy()
        task = consider.iloc[0]['Input.Task']
        rankings = np.array(get_rankings(consider))
        models = set(rankings[0])
        if len(models)!=4:
            print(models)
        def get_indices(m):
            return np.argwhere(rankings==m)[:, 1]

        #print(rankings)
        for m1 in models:
            for m2 in models:
                if m1!=m2 and (m1!='nan') and (m2!='nan'): 

                    #print(m1, m2)
                    m1_idx = get_indices(m1)
                    m2_idx = get_indices(m2)
                    better = np.sum(m1_idx < m2_idx)
                    total = len(rankings)

                    #print(m1, m2, better, total)  
                    if (better/total) >= 1: # majority vote
                        results[task][m1][m2]['unanimous_vote'] += 1
                    if (better/total) >= 0.5: # majority vote
                        results[task][m1][m2]['majority_vote'] += 1
                    results[task][m1][m2]['total_votes'] += 1
                    
    s = " & MRF & FLUTE & SBIC & Positive Reframing \\\ \\midrule "
    print(s)
    comp = args.compare_with
    vote = args.vote
    for model in names:
        s = f"{model} & "
        for task in ['mrf', 'flute', 'sbic', 'positive_reframing']:
            if task in results and model in results[task]:
                x = results[task][model]
                if comp in x and x[comp]['total_votes']:
                    s += f"{100*x[comp][vote]/x[comp]['total_votes']:.1f}\% & "
                else:
                    s += "- & "
        s = s[:-2] + "\\\ "  #\\hline
        print(s)
        #print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval_human")
    parser.add_argument("--dir", type=str, default="hit/output/*balanced*.csv", help="glob string to indicate where the HIT output files are located")
    parser.add_argument("--compare_with", type=str, default="human", choices=["human", "baseline"])
    parser.add_argument(
            "--vote",
            type=str,
            default="unanimous_vote",
            choices=["unanimous_vote", "majority_vote"],
            help="use 1 (unanimous) or 0.5 (majority) as threshold for gold decision",
        )
    args = parser.parse_args()
    main()
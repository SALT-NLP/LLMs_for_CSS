import argparse
import pandas as pd
import numpy as np
from glob import glob
from collections import Counter


BLACKLIST = set(["A1FHH0LTJMMRL"])
MATCHING = ['Input.Generated_1', 'Input.Generated_2', 'Input.Generated_3', 'Input.Generated_4', 'Input.Model_1', 'Input.Model_2', 'Input.Model_3', 'Input.Model_4']

def internal_match(df):
    """Match the HITIds to MTurk where lab evaluation HITId is NaN"""
    for idx, row in df.iterrows():
        hid = row['HITId']
        if (not type(hid)==str):
            consider = df[~df['HITId'].isna()].copy()
            for col in MATCHING:
                consider = consider[consider[col]==row[col]]
            if len(consider):
                new_hid = consider['HITId'].iloc[0]
                #print(new_hid)
                df.at[idx, 'HITId'] = new_hid
            elif not any(row[MATCHING].isna()):
                new_hid = str(hash(' '.join([str(x) for x in row[MATCHING]])))
                #print(idx, new_hid)
                df.at[idx, 'HITId'] = new_hid

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
    evals = pd.concat([pd.read_csv(fn) for fn in glob(args.input)]).reset_index()
    internal_match(evals)
    evals = evals[~evals['HITId'].isna()].copy()
    
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
                    if (better/total) >= 1: # unanimous vote
                        results[task][m1][m2]['unanimous_vote'] += 1
                    if (better/total) > 0.5: # majority vote
                        results[task][m1][m2]['majority_vote'] += 1
                    results[task][m1][m2]['total_votes'] += 1
                    
    s = " & MRF & FLUTE & SBIC & Positive Reframing \\\ \\midrule "
    print(s)
    comp = args.compare_with
    vote = args.vote
    for model in names:
        s = f"\\textttt{{{model}}} & "
        if model == 'chatgpt':
            s = "ChatGPT & "
        elif model == 'baseline':
            s = "Baseline & "
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
    parser.add_argument("--input", type=str, default="hit/output/*balanced*.csv", help="glob string to indicate where the HIT output files are located to be used as input to the table generation")
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
import pandas as pd
df = pd.read_csv('figures/datasheet.csv')
df["Category_idx"] = df.replace({"Category": {"utterance": 0, "conversation": 1, "document": 2}})["Category"]
df["Task"] = df.replace({"Task": {"Classification": "Clf", 
                                  "Prediction": "Pred", 
                                  "Detection": "Det", 
                                  "Generation": "Gen"}})["Task"]
df["Dataset"] = [f"\\texttt{{{row['Dataset']}}} \\citep{{{row['citep']}}}" for _, row in df.iterrows()]
df.sort_values(by=["Category_idx", "Domain", "#Classes"], inplace=True)
df["ID"] = list(range(len(df)))

for cat in sorted(set(df['Category_idx'])):
    consider = df[df['Category_idx']==cat].copy()
    consider.index = consider["ID"]
    consider = consider[['Domain', 'Task', 'Dataset', 'Subject', '#Classes', 'Input', 'Output']]
    
    consider = df[['Domain', 'Task', 'Dataset', 'Subject', '#Classes', 'Input', 'Output']]
    with pd.option_context("max_colwidth", 1000):
        latex = consider.to_latex(formatters={
            'Category': (lambda f: "\\texttt{%s}" % f)
        })
        latex = latex.replace("\\\\", "\\\\ \hline").replace("\\textbackslash ", "\\")
        latex = latex.replace("\{", "{").replace("\}", "}").replace("NaN", "")
        if latex:
            print( latex )
            print()
            print()
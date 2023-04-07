

import json
from ast import literal_eval as make_tuple
with open('sig.json') as f:
    data = json.load(f)

all_models = {make_tuple(x)[2] for x in list(data.keys())}

# NOTE: sig is missing wikievents, hippocorpus, and tropes

best_pairs = [
    ("indian_english_dialect", "google/flan-ul2"),
    ("emotion", "google/flan-ul2"),
    ("flute-classification", "google/flan-ul2"),
    ("humor", "google/flan-t5-xl"),
    ("ibc", "text-davinci-002"),
    ("hate", "google/flan-t5-xl"),
    ("mrf-classification", "google/flan-ul2"),
    ("persuasion", "google/flan-t5-large"),
    ("tempowic", "google/flan-t5-large"),
    ("semeval_stance", "chatgpt"),
    ("discourse", "google/flan-t5-xxl"),
    ("talklife", "google/flan-ul2"),
    ("raop", "google/flan-t5-large"),
    ("politeness", "google/flan-t5-xl"),
    ("power", "chatgpt"),
    ("conv_go_awry", "google/flan-ul2"),

    ("media_ideology", "google/flan-t5-xxl")
]

def get_relevant_sigs(task, best_model):
    left = all_models.copy()
    left.remove(best_model)
    for curr in left:
        curr_sig = data[str((task, curr, best_model))]
        try:
            if curr_sig["Significance"] < 0.05:
                print(curr)
        except:
            print("ERROR with: " + str(curr))
            # print(curr_sig)
            # new_p = data[str((task, best_model, curr))]#["Significance"]
            # print(new_p)
            continue

for dataset, best_model in best_pairs:
    print(dataset)
    get_relevant_sigs(dataset, best_model)
    print()

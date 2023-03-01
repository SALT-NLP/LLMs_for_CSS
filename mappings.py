convokit_ds = {
    "discourse": "reddit-coarse-discourse-corpus",
    "persuasion": "winning-args-corpus",
    "toxicity": "conversations-gone-awry-corpus",
    "politeness": "wikipedia-politeness-corpus",
    "stance": "supreme-corpus",
    "power": "wiki-corpus",
}

# (Where is the Label Stored, What is the Label Applied to, how much context)
convokit_labels = {
    "power": ("speaker", "meta.is-admin", "all"),
    "stance": ("speaker", "meta.side", "all"),
    "politeness": ("utterance", "meta.Binary", "all"),
    "toxicity": ("conversation", "meta.conversation_has_personal_attack", "first_two"),
    "persuasion": ("utterance", "meta.success", "first_two"),
    "discourse": ("utterance", "meta.majority_type", "all"),
}

# (context, label)
csv_column_map = {
    "hate": ("post", "implicit_class"),
    "reframe": ("original_text", "reframed_text"),
    "humor": (3, 1),
    "flute-explanation": (("premise", "hypothesis"), "label"),
    "flute-classification": (("premise", "hypothesis"), "type"),
    "media_ideology": ("content", "bias_text"),
    "hippocorpus": ("story", "events"),
}

csv_download = {
    "hate": "https://huggingface.co/datasets/SALT-NLP/ImplicitHate/raw/main/implicit_hate.csv",
    "reframe": "https://raw.githubusercontent.com/SALT-NLP/positive-frames/main/data/wholetest.csv",
    "humor": "https://raw.githubusercontent.com/orionw/RedditHumorDetection/master/data/reddit_full/test.tsv",
}

jsonl_download = {
    "flute-explanation": "https://huggingface.co/datasets/ColumbiaNLP/FLUTE/raw/main/train.jsonl",
    "flute-classification": "https://huggingface.co/datasets/ColumbiaNLP/FLUTE/raw/main/train.jsonl",
}

prompts_templates = {
    "power": """

Can you predict if the above message comes from someone who is an administrator on Wikipedia? 
A: True
B: False

Constraint: Even if you are uncertain, you must pick reply either True or False with without using any other words.""",
    "stance": """

Does {$speaker} support {$title}? 
A: True
B: False

Constraint: Even if you are uncertain, you must pick reply either True or False with without using any other words.
""",  # This Dataset is strange currently, too long of context
    "politeness": """

Which of the following best describes the statement above? 
A: Polite
B: Neutral
C: Impolite

Constraint: Answer with only the option above that is most accurate and nothing else.
""",
    "persuasion": """

If you were the original poster, would this reply convince you?
A: True
B: False

Constraint: Even if you are uncertain, you must pick reply either True or False with without using any other words.
""",
    "discourse": """

Which of the following best characterizes the discourse type of the previous statement?
A: Question
B: Answer
C: Announcement
D: Agreement
E: Appreciation
F: Elaboration
G: Humor

Constraint: Answer with only the option above that is most accurate and nothing else.
""",
    "hate": """

Which of the following categories of hate speech best describes the sentence above?
A: White Grievance (frustration over a minority group’s perceived privilege and casting majority groups as the real victims of racism)
B: Incitement to Violence (flaunting in-group unity and power or elevating known hate
groups and ideologies)
C: Inferiority Language (implies one group or individual is inferior to another, including dehumanization and toxification)
D: Irony (the use of sarcasm, humor, and satire to attack or demean a protected class or individual)
E: Stereotypes and Misinformation (associating a protected class with negative attributes)
F: Threatening and Intimidation (conveys a speaker commitment to a target’s pain, injury, damage, loss, or violation of rights)

Constraint: Answer with one or more of the options above that is most accurate and nothing else. Always choose at least one of the options.

""",
    "reframe": """

Reframe the previous sentence positively, without changing the core meaning of the sentence.
""",
    "humor": """


Is the above joke humourous by standard societal norms?
A: True
B: False

Constraint: Even if you are uncertain, you must pick reply either True or False with without using any other words.
""",
    "toxicity": """

Will the following conversation eventually going to derail into a personal attack? 
A: True
B: False

Constraint: Even if you are uncertain, you must pick reply either True or False with without using any other words.
""",
    "flute-explanation": """

Which of the following describes the relationship between the premise and the hypothesis?
A: Entailment
B: Contradiction

Constraint: Explain your answer step by step and then list which of the two options is most accurate.
""",
    "flute-classification": """

Which of the following best describes the type of figurative language used in the hypothesis?
A: Idiom
B: Metaphor
C: Creative Paraphrase
D: Sarcasm
E: Simile

Constraint: Answer with only the option above that is most accurate and nothing else.
""",
    "media_ideology": """" 

Which of the following leanings would a political scientist say that the above article has?
A: Left
B: Right
C: Center

Constraint: Answer with only the option above that is most accurate and nothing else.
""",
    "hippocorpus": """

Which sentences above indicate new events? Which of the events are surprising?""",
}

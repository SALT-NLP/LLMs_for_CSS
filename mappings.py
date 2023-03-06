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
    "discourse": ("utterance", "meta.majority_type", "first_two"),
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
    "indian_english_dialect": ("Example", "Feature"),
    "ibc": ("sentence", "leaning"),
    "semeval_stance": ("Tweet", "trump_stance"),
    "tempowic": (("text1", "text2", "word"), "label"),
    "talklife": (("Seeker", "Response"), "remap_level"),
    "sbic": ("post", "targetStereotype"),
    "raop": ("posts", "labels"),
    "emotion": ("text", "labels"),
}

drop_labels = {
    "flute-classification": ["CreativeParaphrase"],
    "flute-explanation": ["CreativeParaphrase"],
    "hate": ["other"],
    "discourse": ["announcement", "other", "negativereaction"],
}

csv_download = {
    "hate": "https://huggingface.co/datasets/SALT-NLP/ImplicitHate/raw/main/implicit_hate.csv",
    "reframe": "https://raw.githubusercontent.com/SALT-NLP/positive-frames/main/data/wholetest.csv",
    "humor": "https://raw.githubusercontent.com/orionw/RedditHumorDetection/master/data/reddit_full/test.tsv",
}

jsonl_download = {
    "flute-explanation": "https://huggingface.co/datasets/ColumbiaNLP/FLUTE/raw/main/train.jsonl",
    "flute-classification": "https://huggingface.co/datasets/ColumbiaNLP/FLUTE/raw/main/train.jsonl",
    # "tempowic": "https://github.com/cardiffnlp/TempoWiC/raw/main/data/test-codalab-10k.data.jl",
}

prompts_templates = {
    "power": """

Based on the conversation is {$speaker} in a position of power? 
True
False

Constraint: Even if you are uncertain, you must pick either True or False with without using any other words.""",
    "stance": """

Does {$speaker} support {$title}? 
True
False

Constraint: Even if you are uncertain, you must pick either True or False with without using any other words.
""",  # This Dataset is strange currently, too long of context
    "politeness": """

Based on formal workplace social norms, which of the following best describes the above conversation? 
A: Polite
B: Neutral
C: Impolite

Constraint: Answer with only the option above that is most accurate and nothing else.
""",
    "persuasion": """

If you were the original poster, would this reply convince you?
True
False

Constraint: Even if you are uncertain, you must pick either True or False with without using any other words.
""",
    "discourse": """

Which of the following best characterizes the discourse type of the previous statement?
A: Question
B: Answer
C: Agreement
D: Disagreement
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

Is the above joke humorous to most of the people? You must pick between \"True\" or \"False\" You cannot use any other words except for \"True\" or \"False\" 
""",
    "toxicity": """

Will the previouse conversation eventually derail into a personal attack? 
True
False

Constraint: Even if you are uncertain, you must pick either True or False with without using any other words.
""",
    "flute-explanation": """

Which of the following describes the relationship between the premise and the hypothesis?
A: Entailment
B: Contradiction

Constraint: Explain the figurative language in the hypothesis in one sentence and then answer with which option is the most accurate.
""",
    "flute-classification": """

Which of the following best describes the type of figurative language used in the hypothesis?
A: Idiom
B: Metaphor
C: Sarcasm
D: Simile

Constraint: Answer with only the option above that is most accurate and nothing else.
""",
    "media_ideology": """" 

Which of the following leanings would a political scientist say that the above article has?
A: Left
B: Right
C: Center

Constraint: Answer with only the option above that is most accurate and nothing else.
""",
    "hippocorpus": """This is an Event Extraction task. Which sentences above indicate new events?""",
    "indian_english_dialect": """
    
Which of the following features would a linguist say that the above sentence has?
A: Article Omission (e.g., 'Person I like most is here.')
B: Copula Omission (e.g., 'Everything busy in our life.')
C: Direct Object Pronoun Drop (e.g., 'He didn’t give me.')
D: Extraneous Article (e.g, 'Educated people get a good money.')
E: Focus Itself (e.g, 'I did it in the month of June itself.')
F: Focus Only (e.g, 'I was there yesterday only'.)
G: General Extender "and all" (e.g, 'My parents and siblings and all really enjoy it'.)
H: Habitual Progressive (e.g., 'They are getting H1B visas.')
I: Invariant Tag "isn’t it, no, na" (e.g., 'It’s come from me, no?')
J: Inversion In Embedded Clause (e.g., 'The school called to ask when are you going back.')
K: Lack Of Agreement (e.g., 'He talk to them.')
L: Lack Of Inversion In Wh-questions (e.g., 'What are you doing?')
M: Left Dislocation (e.g., 'My parents, they really enjoy playing board games.')
N: Mass Nouns As Count Nouns (e.g., 'They use proper grammars there.')
O: Non-initial Existential "is / are there" (e.g., 'Every year inflation is there.')
P: Object Fronting (e.g., 'In fifteen years, lot of changes we have seen.')
Q: Prepositional Phrase Fronting With Reduction (e.g., 'First of all, right side we can see a plate.')
R: Preposition Omission (e.g., 'I stayed alone two years.')
S: Resumptive Object Pronoun (e.g., 'Some teachers when I was in school I liked them very much.')
T: Resumptive Subject Pronoun (e.g., 'A person living in Calcutta, which he didn’t know Hindi earlier, when he comes to Delhi he has to learn English.')
U: Stative Progressive (e.g., 'We will be knowing how much the structure is getting deflected.')
V: Topicalized Non-argument Constituent (e.g., 'in the daytime I work for the courier service')

Constraint: Answer with only the option above that is most accurate and nothing else.
""",
    "ibc": """" 

Which of the following leanings would a political scientist say that the above article has?
A: Liberal
B: Conservative
C: Neutral

Constraint: Answer with only the option above that is most accurate and nothing else.""",
    "semeval_stance": """

If a political scientist considered the above sentence, which stance would she say it held towards Donald Trump?
A: Against
B: Favor
C: None

Constraint: Answer with only the option above that is most accurate and nothing else.""",
    "tempowic": """
    
If a linguist considered the word above in text1 and text2, would she consider the meaning of this word to be the
A: Same
B: Different

Constraint: Answer with only the option above that is most accurate and nothing else.""",
    "sbic": """
    
Explain the stereotype that this sentence communicates? Please use a single sentence like "[GROUP] are ______"
""",

"talklife": """

If a mental health professional considered the above "respone" in the context of a mental health seeker's "message," which of the following would they consider the "response" to be?
A: Strong (A strong exploration is specific and labels the seeker’s experiences and feelings which the peer supporter wants to explore (e.g., replying with "What happened?" to "Are you feeling alone right now?"))
B: Weak (A weak exploration is generic (e.g., replying with "What happened?" to "I am about to have an anxiety attack."))
A: None (No explanation is a response that offers no support (e.g. "okay"))

Constraint: Answer with only the option above that is most accurate and nothing else.""",

"raop": """

If a user on a online community for giving away free pizza saw the above post, they would consider the post to be
A: Persuasive
B: Not persuasive

Constraint: Answer with only the option above that is most accurate and nothing else.""",

"emotion": """

If a mental health professional saw the above text, what emotion would they categorize it to be (using the following six basic emotions according to Paul Ekman)?
A: Fear
B: Anger
C: Joy
D: Sadness
E: Love
F: Surprise

Constraint: Answer with only the option above that is most accurate and nothing else."""
}

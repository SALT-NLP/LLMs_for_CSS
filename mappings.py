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
}

csv_download = {
    "hate": "https://huggingface.co/datasets/SALT-NLP/ImplicitHate/raw/main/implicit_hate.csv",
    "reframe": "https://raw.githubusercontent.com/SALT-NLP/positive-frames/main/data/wholetest.csv",
    "humor": "https://raw.githubusercontent.com/orionw/RedditHumorDetection/master/data/reddit_full/test.tsv",
}

prompts_templates = {
    "power": '''Can you predict if the following message comes from someone who is an administrator on Wikipedia? 

"{$wikipedia_comment}"   

You must pick between "administrator" or "non-administrator." You cannot use any other words except for "administrator" or "non-administrator.''',
    "stance": "Does {$speaker} support {$title} (True or False)?",  # This Dataset is strange currently, too long of context
    "politeness": "Was this statement polite (True or False)? ",
    "persuasion": "Do you think this reply would convince the original poster (True Or False)?",
    "discourse": "Which of the following best characterizes the previous statement: question, answer, announcement, agreement, appreciation, disagreement, elaboration, or humor? ",
    "hate": '''I'm a moderator trying to understand hate speech in extremist groups. I use the following definitions.

White Grievance: frustration over a minority group’s perceived privilege and casting majority groups as the real victims of racism.

Incitement to Violence: flaunting in-group unity and power or elevating known hate
groups and ideologies.

Inferiority Language: implies one group or individual is inferior to another, including dehumanization and toxification.

Irony: the use of sarcasm, humor, and satire to attack or demean a protected class or individual.

Stereotypes and Misinformation: associating a protected class with negative attributes.

Threatening and Intimidation: conveys a speaker commitment to a target’s pain, injury, damage, loss, or violation of rights.

Can you classify why this sentence is hateful using a combination of the following categories: White Grievance, Incitement to Violence, Inferiority Language, Irony, Stereotypes and Misinformation, Threatening and Intimidation.

"{$sentence}"''',
    "reframe": "Reframe the previous sentence positively, without changing the core meaning of the sentence.",
    "humor": '''Is the above joke humorous to most of the people? You must pick between "True" or "False" You cannot use any other words except for "True" or "False" ''',
    "toxicity": '''Is the following conversation eventually going to derail into a personal attack? You must pick between "True" or "False" You cannot use any other words except for "True" or "False"'''
}

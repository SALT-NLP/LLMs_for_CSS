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

convokit_prompts = {
    "power": "Is {$speaker} an administrator (True or False)?",
    "stance": "Does {$speaker} support {$title} (True or False)?",  # This Dataset is strange currently, too long of context
    "politeness": "Was this statement polite (True or False)? ",
    "toxicity": "Predict whether the given conversation has a personal attack (True or False).",
    "persuasion": "Does this reply convince the original poster (Yes Or No)?",
    "discourse": "Which of the following best characterizes the previous statement: question, answer, announcement, agreement, appreciation, disagreement, elaboration, or humor? ",
}

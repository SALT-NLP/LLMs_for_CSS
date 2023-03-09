
export TRANSFORMERS_CACHE=/nlp/scr/oshaikh/flan-cache
declare -a arr=("conv_go_awry" "wiki_corpus" "implicit_hate" "discourse" "reddit_humor" "flute-explanation" "flute-classification" "supreme_corpus" "wiki_politeness" "media_ideology" "hippocorpus" "indian_english_dialect" "ibc" "semeval_stance" "tempowic" "sbic" "talklife" "raop" "emotion" "mrf-explanation" "mrf-classification" "tropes")

for i in "${arr[@]}"
do
    python test_official_chat_css.py --dataset "$i" -m google/flan-t5-xxl -g 2
done


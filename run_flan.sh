export TRANSFORMERS_CACHE=/nlp/scr/oshaikh/flan-cache
declare -a arr=("hate" "conv_go_awry" "wiki_corpus" "discourse" "humor" "flute-explanation" "flute-classification" "politeness" "media_ideology" "hippocorpus" "indian_english_dialect" "ibc" "semeval_stance" "tempowic" "sbic" "mrf-explanation" "mrf-classification" "tropes")

# "talklife" "raop" "emotion"

for i in "${arr[@]}"
do
    echo "USING: $i"
    python test_official_chat_css.py --dataset "$i" -m google/flan-t5-xxl -g 2
done


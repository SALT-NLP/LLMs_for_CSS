export TRANSFORMERS_CACHE=/nlp/scr/oshaikh/flan-cache
declare -a arr=("hate" "reframe" "humor" "flute-explanation" "flute-classification" "media_ideology" "hippocorpus" "indian_english_dialect" "ibc" "semeval_stance" "tempowic" "talklife" "sbic" "raop" "emotion" "mrf-explanation" "mrf-classification" "tropes")

for i in "${arr[@]}"
do
    python test_official_chat_css.py --dataset "$i" -m google/flan-t5-xxl -g 2
done


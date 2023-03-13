export TRANSFORMERS_CACHE=/nlp/scr/oshaikh/flan-cache
declare -a arr=("conv_go_awry" "power" "hate" "discourse" "humor" "flute-explanation" "flute-classification" "supreme_corpus" "politeness" "media_ideology" "indian_english_dialect" "ibc" "semeval_stance" "tempowic" "talklife" "raop" "emotion") 

# declare -a arr=("hippocorpus" "mrf-explanation" "mrf-classification" "tropes" "sbic") 

for i in "${arr[@]}"
do
    echo "USING: $i"
    python test_official_chat_css.py --dataset "$i" -m google/flan-t5-xxl -g 2
done


# for i in "${arr[@]}"
# do
#     echo "USING: $i"
#     python test_official_chat_css.py --dataset "$i"
# done



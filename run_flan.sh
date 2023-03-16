export OPENAI_API_KEY="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"
export TRANSFORMERS_CACHE=/nlp/scr/oshaikh/flan-cache

for i in "raop" "tropes" "mrf-explanation" "mrf-classification" "sbic"
do
    echo "USING: $i"
    python test_official_chat_css.py --dataset "$i" -m google/flan-t5-xxl -g 2
done


# for i in "${arr[@]}"
# do
#     echo "USING: $i"
#     python test_official_chat_css.py --dataset "$i"
# done



export OPENAI_API_KEY="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"

for i in "raop"
do
    echo "USING: $i"
    python test_official_chat_css.py --dataset "$i"
done


# for i in "${arr[@]}"
# do
#     echo "USING: $i"
#     python test_official_chat_css.py --dataset "$i"
# done



for i in "talklife" "tropes" "mrf-explanation" "mrf-classification" "sbic"
do
    echo "USING: $i"
    python test_official_chat_css.py --dataset "$i" -m google/flan-t5-small -g 4
    python test_official_chat_css.py --dataset "$i" -m google/flan-t5-base -g 4
    python test_official_chat_css.py --dataset "$i" -m google/flan-t5-large -g 4
    python test_official_chat_css.py --dataset "$i" -m google/flan-t5-xl -g 4
done


# for i in "${arr[@]}"
# do
#     echo "USING: $i"
#     python test_official_chat_css.py --dataset "$i"
# done



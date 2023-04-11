:> results/table_2.txt
:> results/table_3.txt

for best_pair in "indian_english_dialect","google/flan-ul2" "emotion","google/flan-ul2" "flute-classification","google/flan-ul2" "humor","google/flan-t5-xl" "ibc","text-davinci-002" "hate","google/flan-t5-xl" "mrf-classification","google/flan-ul2" "raop","google/flan-t5-xxl" "tempowic","google/flan-t5-large" "semeval_stance","chatgpt" "discourse","google/flan-t5-xxl" "talklife","google/flan-ul2"  "persuasion","google/flan-t5-large" "politeness","google/flan-t5-xl" "power","chatgpt" "conv_go_awry","google/flan-ul2" "media_ideology","chatgpt"
do IFS=","
   set -- $best_pair
   for comparison in "google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl" "google/flan-t5-xxl" "google/flan-ul2" "chatgpt" "text-ada-001" "text-babbage-001" "text-curie-001" "text-davinci-001" "text-davinci-002" "text-davinci-003" 
   do 
       python eval_significance.py -m $2 -mm $comparison --dataset $1 >> results/table_2.txt
   done
   python eval_agreement.py --dataset $1 -m $2 >> results/table_3.txt
done

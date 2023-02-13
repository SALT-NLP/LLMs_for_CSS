python run_mutual.py --dataset mutual --model text-davinci-003 --method zero_shot --limit_dataset_size 100 --output_dir experiment/mutual_zero_shot_003
python run_mutual.py --dataset mutual --model text-davinci-003 --method few_shot --demo_path demos/mutual_train --limit_dataset_size 100 --output_dir experiment/mutual_few_shot_003

############# Newsela ####################

dataset="newsela"

# CP-Single 
for target in NbWords NbChars LevSim WordRank DepTreeDepth; do
	python train_catboost.py --train_file data/${dataset}/train.csv --eval_file data/${dataset}/valid.csv --config_file experiments/exp-${dataset}-catboost-sg-single-${target}/config.pkl --mode eval --use-ari
done;

# Single
python stich_outputs.py --input_files "experiments/exp-${dataset}-catboost-sg-single-*/output_single_ari.txt" --output_file experiments/newsela-single-sg-ari.txt --model_name T5
python ../scripts/generate.py --complex_file experiments/newsela-single-sg-ari.txt --simple_file ../resources/datasets/newsela-grade/newsela-grade.test.simple --output_file ../experiments/exp_1679357298907587/main_results/newsela-grade.test.single-sg-ari

# Multi
python train_catboost.py --train_file data/${dataset}/train.csv --eval_file data/${dataset}/valid.csv --config_file experiments/exp-${dataset}-catboost-sg-multi/config.pkl  --mode eval --use-ari
python ../scripts/generate.py --complex_file experiments/exp-${dataset}-catboost-sg-multi/output_ari.txt --simple_file ../resources/datasets/newsela-grade/newsela-grade.test.simple --output_file ../experiments/exp_1679357298907587/main_results/newsela-grade.test.multi-sg-ari

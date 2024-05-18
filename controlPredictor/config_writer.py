import os
from utils import write_to_pickle

# CP-single
dataset= "newsela"

# CP-single
for target in  ['NbWords', 'NbChars', 'LevSim', 'WordRank', 'DepTreeDepth', 'ReplaceLevSim']:
    CONFIG = {
        "target" : [target],
        "features": [ 'char_len', 'word_len', 'wordrank', 'dependency_feat', 'TG', 'AoA_mean', 'SG'],
        "model": "single-out-regressor", # multi-out-regressor or regchain
    } 

    exp_dir=f"experiments/exp-{dataset}-catboost-sg-single-{target}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        write_to_pickle(CONFIG, f"{exp_dir}/config.pkl")


CONFIG = {
        "target" : ['NbWords', 'NbChars', 'LevSim', 'WordRank', 'DepTreeDepth'],
        "features": [ 'char_len', 'word_len', 'wordrank', 'dependency_feat', 'TG', 'AoA_mean', 'SG'],
        "model": "multi-out-regressor", # multi-out-regressor or regchain
    } 

exp_dir=f"experiments/exp-{dataset}-catboost-sg-multi"
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

write_to_pickle(CONFIG, f"{exp_dir}/config.pkl")

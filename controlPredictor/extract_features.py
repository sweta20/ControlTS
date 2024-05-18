from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from utils import read_file
from source import *
from source.helper import tokenize
from source.preprocessor import WordRankRatioFeature, DependencyTreeDepthRatioFeature
from source.preprocessor import get_AoA_feats
import argparse


def extract_features(complex_file, simple_file, output_file):
    complex_text = read_file(complex_file)
    simple_text = read_file(simple_file)

    SG, TG = [], []
    for i in range(len(complex_text)):
        sg, tg = [int(a.split("_")[1]) for a in complex_text[i].split(" ")[:2]]
        SG.append(sg)
        TG.append(tg)

    df = pd.DataFrame({"SG": sg, "TG": tg,  "Complex": complex_text, "Simple": simple_text})

    df["char_len"] = df["Complex"].apply(lambda x: len(x))

    df["word_len"] = df["Complex"].apply(lambda x: len(tokenize(x)))

    feature = WordRankRatioFeature()
    df["wordrank"] = df["Complex"].apply(lambda x: feature.get_lexical_complexity_score(x))

    feature = DependencyTreeDepthRatioFeature()
    df["dependency_feat"] = df["Complex"].apply(lambda x: feature.get_dependency_tree_depth(x))

    df['AoA_mean'],  df['AoA_std'] = zip(*df["Complex"].apply(get_AoA_feats))

    df.to_csv(output_file)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complex-file", type=str, default="../data/example.complex")
    parser.add_argument("--simple-file", type=str, default="../data/example.simple")
    parser.add_argument("--output-file", type=str, default="../data/feats.csv")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    extract_features(args.complex_file, args.simple_file, args.output_file)
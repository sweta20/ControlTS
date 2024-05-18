# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import glob
import os
import argparse
from source.resources import DATASETS_DIR, REPO_DIR, PROCESSED_DATA_DIR
from source.evaluate import simplify_file
from easse.report import get_all_scores


# features_kwargs = {
#     'WordRatioFeature': {'target_ratio': 0.95},
#     'CharRatioFeature': {'target_ratio': 0.80},
#     'LevenshteinRatioFeature': {'target_ratio': 0.70},
#     'WordRankRatioFeature': {'target_ratio': 0.90},
#     'DependencyTreeDepthRatioFeature': {'target_ratio': 0.20}
# }

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--complex_file", required=True, type=str)
	parser.add_argument("--simple_file", required=False, default=None, type=str)
	parser.add_argument("--testset", required=False, default="custom", type=str)

	parser.add_argument("--output_file", required=True, type=str)
	parser.add_argument("--model_dirname", default="exp_1679357298907587")
	args = parser.parse_args()

	simplify_file(Path(args.complex_file), Path(args.output_file), features_kwargs=None, model_dirname=args.model_dirname)  

	if args.simple_file != None:
		if args.testset == "custom":
			scores = get_all_scores(args.complex_file, args.output_file, [args.simple_file], lowercase=True)
		else:
			scores = evaluate_system_output(test_set=args.testset, sys_sents_path=str(pred_filepath), lowercase=True)
		print(scores)

if __name__ == '__main__':
	main()
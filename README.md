
This repository contains the code for our paper: [Controlling Pre-trained Language Models
for Grade-Specific Text Simplification](https://aclanthology.org/2023.emnlp-main.790.pdf).

## Abstract

Text simplification (TS) systems rewrite text to make it more readable while preserving its content. However, what makes a text easy to read depends on the intended readers. Recent work has shown that pre-trained language models can simplify text using a wealth of techniques to control output simplicity, ranging from specifying only the desired reading grade level, to directly specifying low-level edit operations. Yet it remains unclear how to set these controlparameters in practice. Existing approaches set them at the corpus level, disregarding the complexity of individual inputs and considering only one level of output complexity. In this work, we conduct an empirical study to understand how different control mechanisms impact the adequacy and simplicity of text simplification systems. Based on these insights, we introduce a simple method that predicts the edit operations required for simplifying a text for a specific grade level on an instance-per-instance basis. This approach improves the quality of the simplified outputs over corpus-level searchbased heuristics.

## How to run the code?

The training of ControlT5 models is based on the [linked](https://github.com/KimChengSHEANG/TS_T5) codebase (recommended to use  Python 3.8.xx). 

1. Install all necessary libraries using `pip install -r requirements.txt`

2. For training the control predictor models, we need to first extract features from the complex and simple texts. This can be done using:

```
  python controlPredictor/extract_features.py --complex-file resources/data/examples.complex --simple-file resources/data/examples.simple --output-file resources/data/features.csv
```

See examples on training single and multi regressor models in `controlPredictor/run_exps.sh`.s


3. For training T5 models, run:

```
python scripts/train.py
```

For more details, refer the original repository linked above. 

## Citation

```
@inproceedings{agrawal-carpuat-2023-controlling,
    title = "Controlling Pre-trained Language Models for Grade-Specific Text Simplification",
    author = "Agrawal, Sweta  and
      Carpuat, Marine",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.790",
    doi = "10.18653/v1/2023.emnlp-main.790",
    pages = "12807--12819",
    abstract = "Text simplification systems rewrite text to make it more readable while preserving its content. However, what makes a text easy to read depends on the intended readers. Recent work has shown that pre-trained language models can simplify text using a wealth of techniques to control output simplicity, ranging from specifying only the desired reading grade level, to directly specifying low-level edit operations. Yet it remains unclear how to set these control parameters in practice. Existing approaches set them at the corpus level, disregarding the complexity of individual inputs and considering only one level of output complexity. In this work, we conduct an empirical study to understand how different control mechanisms impact the adequacy and simplicity of text simplification systems. Based on these insights, we introduce a simple method that predicts the edit operations required for simplifying a text for a specific grade level on an instance-per-instance basis. This approach improves the quality of the simplified outputs over corpus-level search-based heuristics.",
}
```

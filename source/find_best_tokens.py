from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent)) # fix path
from optuna.integration.wandb import WeightsAndBiasesCallback

from source.helper import log_stdout
from source.evaluate import evaluate_on_TurkCorpus, evaluate_on_NewselaCorpus
from source.resources import EXP_DIR
import optuna

def evaluate(params):
    features_kwargs = {
        'WordRatioFeature': {'target_ratio': params['WordRatio']},
        'CharRatioFeature': {'target_ratio': params['CharRatio']},
        'LevenshteinRatioFeature': {'target_ratio': params['LevenshteinRatio']},
        'WordRankRatioFeature': {'target_ratio': params['WordRankRatio']},
        'DependencyTreeDepthRatioFeature': {'target_ratio': params['DepthTreeRatio']}
    }

    return evaluate_on_NewselaCorpus(features_kwargs, 'valid', 'exp_1679357298907587')


def objective(trial: optuna.trial.Trial) -> float:
    params = {
        'WordRatio' : trial.suggest_float('WordRatio', 0.20, 1.5, step=0.05),
        'CharRatio' : trial.suggest_float('CharRatio', 0.20, 1.5, step=0.05),
        'LevenshteinRatio' : trial.suggest_float('LevenshteinRatio', 0.20, 1.5, step=0.05),
        'WordRankRatio' : trial.suggest_float('WordRankRatio', 0.20, 1.5, step=0.05),
        'DepthTreeRatio' : trial.suggest_float('DepthTreeRatio', 0.20, 1.5, step=0.05),
    }
    return evaluate(params)

if __name__=='__main__':

    # pruner: optuna.pruners.BasePruner = (
    #     optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    # )
    tuning_log_dir = EXP_DIR / 'tuning_logs'
    tuning_log_dir.mkdir(parents=True, exist_ok=True)
    i = 1
    tuning_logs = tuning_log_dir / f'tokens_logs_{i}.txt'
    while(tuning_logs.exists()):
        i += 1
        tuning_logs = tuning_log_dir / f'tokens_logs_{i}.txt'
    
    wandb_kwargs = {"project": "optimize-corpus"}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

        
    with log_stdout(tuning_logs):
        study = optuna.create_study(study_name='optimize-corpus', direction="maximize", storage=f'sqlite:///{tuning_log_dir}/Tokens_study.db', load_if_exists=True)
        
        study.optimize(objective, n_trials=100, callbacks=[wandbc])

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
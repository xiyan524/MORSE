import os
from scorer.eval.run_scorer import cal_metrics_score
#from scorer.eval.scorer_morse import cal_metrics_score


folder_path = "result_path"
evaluation_root_dir = "scorer/"

prediction_path = os.path.join(folder_path, "predictions")
print("evaluation begin")
prediction_results_path = os.path.join(folder_path, "predictions_results")
cal_metrics_score(prediction_path, prediction_results_path, evaluation_root_dir,task="task_1")
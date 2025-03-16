""" Evaluation script for EntailmentBank models. """

import codecs
import csv
import os
import argparse
import glob
import logging
import json
import re
import sys
from bleurt import score

from tqdm import tqdm

from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip

from scorer.utils.angle_utils import decompose_slots, load_jsonl, save_json, shortform_angle, formatting
from scorer.utils.eval_utils import collate_scores, score_prediction_whole_proof

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, required=True, type=str,
                        help="Task name: task_1, task_2, task_3")
    parser.add_argument("--output_dir", default=None, required=True, type=str,
                        help="Directory to store scores.")
    parser.add_argument("--split", default=None, required=True, type=str, help="Which split (train/dev/test) to evaluate.")
    parser.add_argument("--prediction_file", default=None, required=True, type=str,
                        help="Prediction file(s) to score.")
    parser.add_argument("--bleurt_checkpoint", default="true", type=str,
                        help="Path to the BLEURT model checkpoint (Download from https://github.com/google-research/bleurt#checkpoints) "
                             "We use bleurt-large-512 model for EntailmentBank evaluation")

    args = parser.parse_args()
    return args

def split_info_sentences(context):
    words_list = context.split(" ")
    sentence_ids = re.findall(r'[\w\.]+[0-9]+:', context)
    sentence_dict = dict()
    prev_sid = ""
    prev_sentence_parts = []
    for word in words_list:
        if word in sentence_ids:
            if prev_sid:
                sentence_dict[prev_sid] = ' '.join(prev_sentence_parts)
                prev_sentence_parts = []
            prev_sid = word
        else:
            prev_sentence_parts.append(word)

    if prev_sid:
        sentence_dict[prev_sid] = ' '.join(prev_sentence_parts)
    return sentence_dict

def find_all_substr(s, sub):
	index_list = []
	index = s.find(sub)
	while index != -1:
		if s[index+4].isdigit():
			index_list.append(index)
		index = s.find(sub, index + 1)

	return index_list

def make_context_dict(context):
    sent_index = find_all_substr(context, "sent")
    candidate_sents_dict = {}
    for index_s in range(len(sent_index)):
        if index_s != len(sent_index) - 1:
            candidate_sents_dict["sent"+str(index_s+1)] = context[sent_index[index_s]:sent_index[index_s + 1] - 1]
        else:
            candidate_sents_dict["sent"+str(index_s+1)] = context[sent_index[index_s]:]
    return candidate_sents_dict

def score_predictions(predictions_file, score_file, gold_file, angle_file=None, dataset=None, bleurt_checkpoint="scorer/bleurt/bleurt-large-512"):
    #bleurt_checkpoint = None

    if bleurt_checkpoint:
        bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)
    else:
        bleurt_scorer = None

    gold_data = load_jsonl(gold_file)
    gold_by_id = {g['id']: g for g in gold_data}

    gold_train_file = gold_file.replace("dev", "train")
    gold_train_data = load_jsonl(gold_train_file)
    # gold_train_by_id = {g['id']: g for g in gold_train_data}
    train_context_dict = dict()
    #train_answers_dict = dict()
    for g in gold_train_data:
        # print(f"{g['meta']['triples']}")
        # context_dict = dict(g['meta']['triples'])
        context_dict = make_context_dict(g['new_context'])
        for context in context_dict.values():
            train_context_dict[context] = 1
        #train_answers_dict[g['answer']] = 1

    is_jsonl = predictions_file.endswith(".jsonl")
    if not is_jsonl:
        angle_data = load_jsonl(angle_file)
    scores = []
    sort_angle = False

    num_dev_answers = 0
    num_dev_answers_seen_in_train_context = 0
    num_dev_answers_seen_in_train_answers = 0
    diagnostics_tsv = open(score_file+".diagnostics.tsv", "w")
    diagnostics_tsv.write(f"Q-ID"
                          f"\tI:Context"
                          f"\tI:Question"
                          f"\tI:Answer"
                          f"\tI:Hypothesis"
                          f"\tO:Gold Proof"
                          f"\tO:Predicted Proof"
                          f"\tPredicted to gold int alignment"
                          f"\tRewritten predicted proof after alignment"
                          f"\t% Leaves-P"
                          f"\t% Leaves-R"
                          f"\t% Leaves-F1"
                          f"\t% Leaves-F1"
                          f"\t% Leaves-Correct"
                          f"\t% Steps-F1"
                          f"\t% Steps-Correct"
                          f"\t% Interm-BLEURT-P"
                          f"\t% Interm-BLEURT-R"
                          f"\t% Interm-BLEURT-F1"
                          f"\tInterm-BLEURT-score"
                          f"\t% Interm-BLEURT-Acc"
                          f"\t% perfect alignment"
                          f"\t% Overall Correct"
                          f"\tNum Distractors"
                          f"\tContext Length"
                          f"\tFraction of distractors"
                          f"\tDistractor Ids"
                          "\n")

    with open(f"{score_file}.json", "w") as score_file, open(predictions_file, "r") as preds_file:
        for line_idx, line in tqdm(enumerate(preds_file)):
            if is_jsonl:
                pred = json.loads(line.strip())
            else:
                pred = {'id': angle_data[line_idx]['id'],
                        'angle': angle_data[line_idx]['angle'],
                        'prediction': line.strip()}

            angle = pred['angle']
            angle_canonical = shortform_angle(angle, sort_angle=sort_angle)
            pred['angle_str'] = angle_canonical
            item_id = pred['id']


            if item_id not in gold_by_id:
                continue
                raise ValueError(f"Missing id in gold data: {item_id}")
            slots = decompose_slots(pred['prediction'])

            pred['slots'] = slots

            num_dev_answers += 1
            # print(f"======= pred: {pred}")
            # print(f">>>>>>>>>>>> id:{item_id}")
            metrics = score_prediction_whole_proof(pred, gold_by_id[item_id], dataset,
                                                      scoring_spec={
                                                          "hypothesis_eval": "nlg",
                                                          "proof_eval": "entail_whole_proof_align_eval",
                                                          # "proof_eval": "entail_whole_polish_proof_align_eval",
                                                      },
                                                    bleurt_scorer=bleurt_scorer)

            pred['metrics'] = metrics
            score_file.write(json.dumps(pred) + "\n")
            id = angle_data[line_idx]['id']
            diagnostics_tsv.write(f"{id}"
                                  f"\t{metrics['proof-leaves']['P']*100}"
                                  f"\t{metrics['proof-leaves']['R']*100}"
                                  f"\t{metrics['proof-leaves']['F1']*100}"
                                  f"\t{metrics['proof-leaves']['acc']*100}"
                                  f"\t{metrics['proof-steps']['F1']*100}"
                                  f"\t{metrics['proof-steps']['acc']*100}"
                                  f"\t{metrics['proof-intermediates']['BLEURT_P']*100}"
                                  f"\t{metrics['proof-intermediates']['BLEURT_R']*100}"
                                  f"\t{metrics['proof-intermediates']['BLEURT_F1']*100}"
                                  f"\t{metrics['proof-intermediates']['BLEURT']}"
                                  f"\t{metrics['proof-intermediates']['BLEURT_acc']*100}"
                                  f"\t{metrics['proof-intermediates']['fraction_perfect_align']*100}"
                                  f"\t{metrics['proof-overall']['acc']*100}"                                  
                                  "\n")

            scores.append(pred)

    return scores


# Sample command
def main(args):
    prediction_files = args.prediction_file
    if "," in prediction_files:
        prediction_files = prediction_files.split(",")
    elif os.path.isdir(prediction_files):
        dir_path = prediction_files
        prediction_files = [f"{dir_path}/{f}" for f in os.listdir(prediction_files) if re.match(r'.*_predictions', f)]
    else:
        prediction_files = glob.glob(prediction_files)

    prediction_files.sort()
    root_dir = "/Users/xiyan/OneDrive - mail.nankai.edu.cn/code/entailment_bank-main/data/dbpedia/fine-tuning"
    #root_dir = "/Users/xiyan/OneDrive - mail.nankai.edu.cn/code/entailment_bank-main/data/processed_data"
    angle_data_dir = f"{root_dir}/angles/{args.task}/"
    slot_data_dir = f"{root_dir}/slots/{args.task}-slots/"
    angle_base_name = os.path.basename(angle_data_dir)
    slot_file = os.path.join(slot_data_dir, args.split + '.jsonl')
    if not os.path.exists(slot_file):
        if args.split == 'val' and os.path.exists(os.path.join(slot_data_dir, "dev.jsonl")):
            slot_file = os.path.join(slot_data_dir, "dev.jsonl")
        else:
            raise ValueError(f"Slot data file {slot_file} does not exist!")
    predictions_jsonl_format = True
    for prediction_file in prediction_files:
        if not prediction_file.endswith(".jsonl"):
            predictions_jsonl_format = False
    angle_file = None
    # If predictions not in jsonl format, need angle data to get correct ids and angles
    if not predictions_jsonl_format:
        angle_file = os.path.join(angle_data_dir, args.split + '.jsonl')
        if not os.path.exists(angle_file):
            if args.split == 'val' and os.path.exists(os.path.join(angle_data_dir, "dev.jsonl")):
                slot_file = os.path.join(angle_data_dir, "dev.jsonl")
            else:
                raise ValueError(f"Angle data file {angle_file} does not exist!")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Scoring the following files: %s", prediction_files)
    all_metrics_aggregated = {}

    split = args.split
    output_dir = args.output_dir
    bleurt_checkpoint = args.bleurt_checkpoint

    sys.argv = sys.argv[:1]

    for prediction_file in prediction_files:
        if not os.path.exists(prediction_file):
            logger.warning(f"  File not found: {prediction_file}")
            continue
        score_file_base = f"scores-{split}"
        score_file = os.path.join(output_dir, score_file_base)
        logger.info(f"***** Scoring predictions in {prediction_file} *****")
        logger.info(f"   Gold data from: {slot_file}")
        logger.info(f"   Full output in: {score_file}")

        scores = score_predictions(
            predictions_file=prediction_file,
            score_file=score_file,
            gold_file=slot_file,
            angle_file=angle_file,
            dataset=angle_base_name,
            bleurt_checkpoint=bleurt_checkpoint)
        collated = collate_scores(scores)
        all_metrics_aggregated[score_file] = collated['metrics_aggregated']
        logger.info(f"    Aggregated metrics:")
        for key, val in collated['metrics_aggregated'].items():
            logger.info(f"       {key}: {val}")
        print(f"\n======================")
        colmns_str = '\t'.join([
                                # 'leave-P', 'leave-R',
                                'leave-F1',	'leaves-Acc',
                                'steps-F1',	'steps-Acc',
                                'int-BLEURT-F1', 'int-BLEURT-Acc',
                                #'int-BLEURT_align',	'int-BLEURT-Acc_align',
                                'overall-Acc'
                                #,'overall-Acc_align','int-fraction-align'
                                ])
        print(f"collated:{collated['metrics_aggregated']}")
        aggr_metrics = collated['metrics_aggregated']['QAHC->P']
        metrics_str = '\t'.join([
            # prediction_file,
            # str(round(aggr_metrics['proof-leaves']['P']*100.0, 2)),
            # str(round(aggr_metrics['proof-leaves']['R']*100.0, 2)),
            str(round(aggr_metrics['proof-leaves']['F1']*100.0, 2)),
            str(round(aggr_metrics['proof-leaves']['acc']*100.0, 2)),
            str(round(aggr_metrics['proof-steps']['F1']*100.0, 2)),
            str(round(aggr_metrics['proof-steps']['acc']*100.0, 2)),
            str(round(aggr_metrics['proof-intermediates']['BLEURT_F1'] * 100.0, 2)),
            str(round(aggr_metrics['proof-intermediates']['BLEURT_acc'] * 100.0, 2)),
            #str(round(aggr_metrics['proof-intermediates']['BLEURT_perfect_align'], 2)),
            #str(round(aggr_metrics['proof-intermediates']['BLEURT_acc_perfect_align']*100.0,2)),
            str(round(aggr_metrics['proof-overall']['acc']*100.0, 2)),
            #str(round(aggr_metrics['proof-overall']['acc_perfect_align']*100.0, 2)),
            #str(round(aggr_metrics['proof-intermediates']['fraction_perfect_align'] * 100.0, 2)),
        ])
        print(f"{colmns_str}")
        print(f"{metrics_str}")
    save_json(f"{score_file}.metrics.json", all_metrics_aggregated)

def cal_metrics_score(prediction_file, output_dir, root_dir, task="task_1", split="test", bleurt_checkpoint="scorer/bleurt/bleurt-large-512"):
    """
    prediction_files: the predict results
    output_dir: where to save the result
    """
    #bleurt_checkpoint = None

    # 参数1, 预测的文件
    prediction_files = prediction_file
    if "," in prediction_files: # 多个文件, 以逗号隔开
        prediction_files = prediction_files.split(",")
    elif os.path.isdir(prediction_files): # 传入的是目录
        dir_path = prediction_files
        # prediction_files = [f"{dir_path}/{f}" for f in os.listdir(prediction_files) if re.match(r'.*_predictions', f)]
        prediction_files = [f"{dir_path}/{f}" for f in os.listdir(prediction_files) if re.match(r'Test_.*', f)]
    else:
        prediction_files = [prediction_files] # 传入的是单个文件

    # 预处理txt结果, 筛去特殊字符, 加上$proof$
    processed_prediction_files = []
    for pred_file in prediction_files:
        origin_dir = os.path.dirname(pred_file)
        file_name = pred_file.split("/")[-1].split(".")[0]
        new_pred_file = os.path.join(origin_dir, "{}.tsv".format(file_name))
        processed_prediction_files.append(new_pred_file)
        predictions = codecs.open(pred_file).read().split("\n")
        with open(new_pred_file, 'w') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            for prediction in predictions[:-1]:
                prediction = prediction.replace('<pad>', '')
                prediction = prediction.replace('int<', 'int: wrong <')
                prediction = prediction.replace('</s>', '')
                prediction = '$proof$ =' + prediction
                tsv_w.writerow([prediction])

    processed_prediction_files.sort()
    # 这些目录名需要改一下
    #root_dir = "scorer/data/depth_processed_data"
    angle_data_dir = f"{root_dir}/angles/{task}/"
    slot_data_dir = f"{root_dir}/slots/{task}-slots/"
    angle_base_name = os.path.basename(angle_data_dir)
    slot_file = os.path.join(slot_data_dir, split + '.jsonl')
    if not os.path.exists(slot_file):
        if split == 'val' and os.path.exists(os.path.join(slot_data_dir, "dev.jsonl")):
            slot_file = os.path.join(slot_data_dir, "dev.jsonl")
        else:
            raise ValueError(f"Slot data file {slot_file} does not exist!")
    predictions_jsonl_format = True
    for prediction_file in processed_prediction_files:
        if not prediction_file.endswith(".jsonl"):
            predictions_jsonl_format = False
    angle_file = None
    # If predictions not in jsonl format, need angle data to get correct ids and angles
    if not predictions_jsonl_format:
        angle_file = os.path.join(angle_data_dir, split + '.jsonl')
        if not os.path.exists(angle_file):
            if split == 'val' and os.path.exists(os.path.join(angle_data_dir, "dev.jsonl")):
                slot_file = os.path.join(angle_data_dir, "dev.jsonl")
            else:
                raise ValueError(f"Angle data file {angle_file} does not exist!")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info("Scoring the following files: %s", processed_prediction_files)
    all_metrics_aggregated = {}

    for prediction_file in processed_prediction_files:
        if not os.path.exists(prediction_file):
            logger.warning(f"  File not found: {prediction_file}")
            continue
        step_num = prediction_file.split("/")[-1].split(".")[0].split("_")[-1]
        score_file_base = f"scores-{step_num}-{split}"
        score_file = os.path.join(output_dir, score_file_base)
        logger.info(f"***** Scoring predictions in {prediction_file} *****")
        logger.info(f"   Gold data from: {slot_file}")
        logger.info(f"   Full output in: {score_file}")

        scores = score_predictions(
            predictions_file=prediction_file,
            score_file=score_file,
            gold_file=slot_file,
            angle_file=angle_file,
            dataset=angle_base_name,
            bleurt_checkpoint=bleurt_checkpoint)
        collated = collate_scores(scores)
        all_metrics_aggregated[score_file] = collated['metrics_aggregated']
        logger.info(f"    Aggregated metrics:")
        for key, val in collated['metrics_aggregated'].items():
            logger.info(f"       {key}: {val}")
        # print(f"\n======================")
        colmns_str = '\t'.join([
            # 'leave-P', 'leave-R',
            'leave-F1', 'leaves-Acc',
            'steps-F1', 'steps-Acc',
            'int-BLEURT-F1', 'int-BLEURT-Acc',
            # 'int-BLEURT_align',	'int-BLEURT-Acc_align',
            'overall-Acc'
            # ,'overall-Acc_align','int-fraction-align'
        ])
        # print(f"collated:{collated['metrics_aggregated']}")
        aggr_metrics = collated['metrics_aggregated']['QAHC->P']
        metrics_str = '\t'.join([
            # prediction_file,
            # str(round(aggr_metrics['proof-leaves']['P']*100.0, 2)),
            # str(round(aggr_metrics['proof-leaves']['R']*100.0, 2)),
            str(round(aggr_metrics['proof-leaves']['F1'] * 100.0, 2)),
            str(round(aggr_metrics['proof-leaves']['acc'] * 100.0, 2)),
            str(round(aggr_metrics['proof-steps']['F1'] * 100.0, 2)),
            str(round(aggr_metrics['proof-steps']['acc'] * 100.0, 2)),
            str(round(aggr_metrics['proof-intermediates']['BLEURT_F1'] * 100.0, 2)),
            str(round(aggr_metrics['proof-intermediates']['BLEURT_acc'] * 100.0, 2)),
            # str(round(aggr_metrics['proof-intermediates']['BLEURT_perfect_align'], 2)),
            # str(round(aggr_metrics['proof-intermediates']['BLEURT_acc_perfect_align']*100.0,2)),
            str(round(aggr_metrics['proof-overall']['acc'] * 100.0, 2)),
            # str(round(aggr_metrics['proof-overall']['acc_perfect_align']*100.0, 2)),
            # str(round(aggr_metrics['proof-intermediates']['fraction_perfect_align'] * 100.0, 2)),
        ])
        # print(f"{colmns_str}")
        # print(f"{metrics_str}")
    save_json(f"{score_file}.metrics.json", all_metrics_aggregated)


def get_argparse():
    parser = argparse.ArgumentParser()

    ## data
    parser.add_argument("--task", default="task_1", type=str, help="")
    parser.add_argument("--split", default="test", type=str, help="")
    parser.add_argument("--prediction_file", default="/Users/xiyan/OneDrive - mail.nankai.edu.cn/code/entailment_bank-main/Test_27300.tsv", type=str, help="")
    parser.add_argument("--output_dir", default="result", type=str, help="")
    parser.add_argument("--bleurt_checkpoint", default="/Users/xiyan/OneDrive - mail.nankai.edu.cn/code/entailment_bank-main/bleurt/bleurt-large-512/", type=str, help="")

    return parser

if __name__ == "__main__":
    #args = get_args()
    args = get_argparse().parse_args()
    main(args)
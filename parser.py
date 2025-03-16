import argparse

def get_argparse():
    parser = argparse.ArgumentParser()

    ## data
    parser.add_argument("--data_dir", default="dataset_path", type=str, help="path of input dataset")
    parser.add_argument("--output_dir", default="result_depth", type=str, help="path of output results")
    parser.add_argument("--logging_dir", default='result_depth/log', type=str, help="the dir for log")
    parser.add_argument("--base_model", default="baseline", type=str, help="base_model")
    parser.add_argument("--data_ratio", default="1", type=float, help="ratio of data to be used for training")
    
    ## model
    parser.add_argument("--config_name", default="module/bert/config.json", type=str, help="the config of define model")
    parser.add_argument("--vocab_file", default="module/bert/vocab.txt", type=str, help="the vocab file for bert")
    parser.add_argument("--model_name_or_path", default="module/bert/pytorch_model.bin", type=str, help="the pretrained bert path")
    parser.add_argument("--max_input_num", default=512, type=int, help="the max length of input sequence")
    parser.add_argument("--max_output_num", default=30, type=int, help="the max length of input sequence")
    parser.add_argument("--threshold", default=0.1, type=float, help="the threshold for function selection")
    parser.add_argument("--checkpoint_name", default="check450", type=str, help="the checkpoint name")
    parser.add_argument("--checkpoint_path", default="", type=str, help="the checkpoint path")
    parser.add_argument("--evaluation_root_dir", default="", type=str, help="the evaluation path")


    ## train
    parser.add_argument("--seed", default=106524, type=int, help="the seed used to initiate parameters")
    parser.add_argument("--do_shuffle", default=True, type=bool, help="do shuffle for each piece dataset or not")
    parser.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_finetuning", default="p1", type=str, help="Whether to run training.")
    parser.add_argument("--max_steps", default=-1, type=int, help="the total number of training steps")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch, only work when max_step==-1")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="the training batch size")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="the eval batch size")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="the weight of L2 normalization")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--warmup_steps", default=0, type=int, help="the number of warmup steps")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max clip gradient?")
    parser.add_argument("--logging_steps", default=2, type=int, help="Log every X updates steps")
    parser.add_argument("--save_steps", default=1, type=int, help="How often to save the model chekcpoint")
    parser.add_argument("--evaluate_during_training", default=True, action="store_true", help="Whether do evuation during training.")
    parser.add_argument("--evaluate_epoch_num", default=1, type=int, help="epoch num for evaluation")
    parser.add_argument("--freeze_param_list", default="module_embedding,", type=str, help="epoch num for evaluation")


    #test
    parser.add_argument("--beam_size", default=4, type=int, help="beam number for beam size searching")

    return parser
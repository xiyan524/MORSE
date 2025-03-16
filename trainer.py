import codecs
import time
import os
import random
import numpy as np
import torch
import logging

from tqdm.auto import tqdm, trange
from transformers import T5Tokenizer
from module.t5.configuration_t5 import T5Config
#from module.t5.modeling_t5 import T5ForConditionalGeneration
from module.t5.module_v1 import T5ForConditionalGeneration
#from module.data.dataset_tree import TreeDataset
from module.data.dataset_dbpedia import TreeDataset
from module.sampler import SequentialDistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from parser import get_argparse
#from scorer.eval.run_scorer import cal_metrics_score
from scorer.eval.scorer_morse import cal_metrics_score

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # output to handler
chlr.setFormatter(formatter)
logfile = './result_length_new/log/test_{}.txt'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))
fh = logging.FileHandler(logfile)
fh.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fh)

PREFIX_CHECKPOINT_DIR = "checkpoint"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode='train'):
    """
    generator datasetloader for training.
    Note that: for training, we need random sampler, same to shuffle
               for eval or predict, we need sequence sampler, same to no shuffle
    Args:
        dataset:
        args:
        mode: train or non-train
    """
    print("Dataset length: ", len(dataset))
    if mode == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    if mode == 'train':
        batch_size = args.per_gpu_train_batch_size
    else:
        batch_size = args.per_gpu_eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader


def get_optimizer(model, args):
    """
    Setup the optimizer and the learning rate scheduler

    we provide a reasonable default that works well
    If you want to use something else, you can pass a tuple in the Trainer's init,
    or override this method in a subclass.
    """
    # freeze some parameters
    freeze_param_list = args.freeze_param_list.split(",")[:-1]
    for name, param in model.named_parameters():
        for freeze_param in freeze_param_list:
            if freeze_param in name:
                param.requires_grad = False

    no_bigger = ["generate_text"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_bigger)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_bigger)],
            "lr": 0.0001
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = get_linear_schedule_with_warmup(
    #    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    #)

    #return optimizer, scheduler
    return optimizer


def print_log(logs, epoch, global_step, eval_type, writer, iterator=None):
    if epoch is not None:
        logs['epoch'] = epoch
    if global_step is None:
        global_step = 0
    if eval_type in ["Dev", "Test"]:
        print("#############  %s's result  #############"%(eval_type))
    """if writer:
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(k, v, global_step)
            else:
                logger.warning(
                    "Trainer is attempting to log a value of "
                    '"%s" of type %s for key "%s" as a scalar. '
                    "This invocation of Tensorboard's writer.add_scalar() "
                    "is incorrect so we dropped this attribute.",
                    v,
                    type(v),
                    k,
                )
        writer.flush()"""

    output = {**logs, **{"step": global_step}}
    if iterator is not None:
        iterator.write(output)
    else:
        logger.info(output)


def train(model, args, train_dataset, dev_dataset, test_dataset, tokenizer, writer, model_path=None):
    """ train the model"""
    print_cnt = 0
    ## 1.prepare data
    train_dataloader = get_dataloader(train_dataset, args, mode='train')
    num_train_epochs = args.num_train_epochs

    ## 2.optimizer and model
    optimizer = get_optimizer(model, args)
    logger.info("***** freeze parameters *****")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            logger.info(name)

    ## 3.begin train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per device = %d", args.per_gpu_train_batch_size)

    global_step = 0
    epoch_num = 0
    epochs_trained = 0

    tr_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.train()

            # new batch data: [inputs, attention_mask, decoder_outputs, decoder_attention_mask]
            new_batch = (batch[0], batch[1], batch[2], batch[3], )
            batch = tuple(t.to(args.device) for t in new_batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "labels": batch[2], "decoder_attention_mask": batch[3],
                      "return_dict": False}
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            tr_loss += loss.item()

            ## update gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            global_step += 1

            ## logger and evaluate
            if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                logs = {}
                logs["loss"] = (tr_loss - logging_loss) / args.logging_steps
                logging_loss = tr_loss
                print_log(logs, epoch, global_step, "", writer)

        # evaluate after each epoch
        if args.evaluate_during_training and epoch_num > args.evaluate_epoch_num and epoch_num % 5 == 0:
        #if args.evaluate_during_training and epoch_num > args.evaluate_epoch_num:
            # for dev & test
            #evaluate(model, args, dev_dataset, global_step, tokenizer, description="Dev", write_file=True)
            evaluate(model, args, test_dataset, global_step, tokenizer, description="Test", write_file=True)

            if args.do_finetuning == "p1":
                output_dir = os.path.join(args.folder_path, "checkpoints")
                output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        epoch_num += 1
    print("global_step: ", global_step)
    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    return global_step, tr_loss / global_step


def evaluate(model, args, dataset, global_step, tokenizer, description="dev", write_file=False):
    """evaluate the model's performance"""

    dataloader = get_dataloader(dataset, args, mode=description)
    model = model.to(args.device)

    batch_size = dataloader.batch_size
    logger.info("***** Running %s *****", description)
    logger.info("  Num examples = %d", len(dataloader.dataset))
    logger.info("  Batch size = %d", batch_size)
    model.eval()

    predictions = []
    labels = []
    for batch in tqdm(dataloader, desc=description):
        # new batch data: [inputs, attention_mask, decoder_outputs, decoder_attention_mask]
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        decoder_ids = batch[2].to(args.device)

        with torch.no_grad():
            beam_outputs = model.generate(
                input_ids,
                max_length=args.max_output_num,
                #num_beams=5,
                #no_repeat_ngram_size=2,
                #num_return_sequences=1,
                #bos_token_id=104,
                #early_stopping=True
            )

        beam_outputs = beam_outputs.cpu().numpy().tolist()
        decoder_ids = decoder_ids.cpu().numpy().tolist()
        prediction = [tokenizer.decode(beam_outputs[i]) for i in range(len(beam_outputs))]
        for idx in range(len(decoder_ids)):
            tmp = np.array(decoder_ids[idx])
            tmp[tmp == -100] = 0
            decoder_ids[idx] = tmp.tolist()
        label = [tokenizer.decode(decoder_ids[i]) for i in range(len(beam_outputs))]
        for i in range(len(prediction)):
            predictions.append(prediction[i])
            labels.append(label[i])

    # store predictions
    file_name = description + "_" + str(global_step) + ".txt"
    file_path = os.path.join(args.folder_path, "predictions")
    file_path = os.path.join(file_path, file_name)
    file = codecs.open(file_path, 'a')
    for prediction in predictions:
        file.write(prediction+"\n")
    file.close()


def main():
    args = get_argparse().parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("device: %s", args.device)
    logger.info("Training/evaluation parameters %s", args)
    writer = SummaryWriter(log_dir=args.logging_dir)
    set_seed(args.seed)

    ## prepare data
    train_data_file = os.path.join(args.data_dir, "train.jsonl")
    #dev_data_file = os.path.join(args.data_dir, "dev.jsonl")
    test_data_file = os.path.join(args.data_dir, "test.jsonl")

    # store predictions and checkpoint
    config = T5Config.from_pretrained("t5-base")
    folder_name = str(args.base_model) + str(args.data_ratio) + "_epoch" + str(args.num_train_epochs) + "_input" + str(args.max_input_num) + "_output" + str(args.max_output_num) + "_batch" + str(args.per_gpu_train_batch_size) + "_lr" + str(args.learning_rate) + "_t" + str(config.threshold) + "_" + str(args.do_finetuning)
    if args.do_finetuning == "p2":
        folder_name = folder_name + "_" + str(args.checkpoint_name)
    folder_path = os.path.join(args.output_dir, folder_name)
    args.folder_path = folder_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, "predictions"))
        os.makedirs(os.path.join(folder_path, "checkpoints"))
        os.makedirs(os.path.join(folder_path, "predictions_results"))

    ## define model
    if args.do_finetuning == "p1":
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
    else:
        config = T5Config.from_pretrained("t5-base")
        model = T5ForConditionalGeneration(config=config)
        model.load_state_dict(torch.load(args.checkpoint_path))
    print("checkpoint:", args.checkpoint_path)

    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        model = model.to(args.device)

    #tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    dataset_params = {
        'tokenizer': tokenizer,
        'max_input_num': args.max_input_num,
        'max_output_num': args.max_output_num,
        'data_ratio': args.data_ratio,
    }

    if args.do_train:
        print("train")
        train_dataset = TreeDataset(train_data_file, "train", params=dataset_params, do_shuffle=args.do_shuffle)
        #dev_dataset = TreeDataset(dev_data_file, params=dataset_params, do_shuffle=False)
        test_dataset = TreeDataset(test_data_file, "test", params=dataset_params, do_shuffle=False)
        dev_dataset = test_dataset
        train(model, args, train_dataset, dev_dataset, test_dataset, tokenizer, writer)
        evaluate(model, args, test_dataset, 0, tokenizer, description="Test", write_file=True)

    prediction_path = os.path.join(folder_path, "predictions")
    prediction_results_path = os.path.join(folder_path, "predictions_results")
    cal_metrics_score(prediction_path, prediction_results_path, args.evaluation_root_dir)


if __name__ == "__main__":
    main()
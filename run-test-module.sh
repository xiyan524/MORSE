
# params below not changed very often
# CHANGEME - path to your environment python (when environment is activated write "which python")
python_exec=python_path

# CHANGEME - your script for running one single file.
#JOB_SCRIPT="PYTHONPATH=. $python_exec test.py"
JOB_SCRIPT="PYTHONPATH=. $python_exec trainer.py  \
        --data_dir="dataset_path"  \
        --output_dir=""  \
        --base_model="module"  \
        --max_input_num=200  \
        --max_output_num=200  \
        --per_gpu_train_batch_size=4  \
        --per_gpu_eval_batch_size=4  \
        --learning_rate=1e-4  \
        --num_train_epochs=70   \
        --evaluate_epoch_num=30 \
        --data_ratio=${RATIO}  \
        --checkpoint_name=""  \
        --checkpoint_path=""  \
        --do_finetuning="p1"  \
        --do_train \
        --evaluation_root_dir="scorer/"  \
        --freeze_param_list=""   \
"


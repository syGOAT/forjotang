export CUDA_VISIBLE_DEVICES=0
train_file=./dataset/train.json
predict_file=./dataset/test.json
# model_name_or_path=/yourpath/uncased_L-12_H-768_A-12
# training
python main.py   --model_type 'bert' \
  --do_train  \
 --train_file  $train_file \
 --predict_file  $predict_file \
  --config_name 'bert-base-uncased' \
 --tokenizer_name 'bert-base-uncased' \
 --model_name 'bert-base-uncased' \
 --output_dir ./checkpoints_prog \
 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8  \
 --gradient_accumulation_steps 1 \
 --max_grad_norm inf \
 --adam_epsilon 1e-6 --adam_beta_2 0.98 --weight_decay 0.01  \
 --warmup_proportion 0.06 --num_train_epochs 5 --overwrite_output_dir  --save_freq 1   \
 --model_descri 'multi_prog'  --task prog_classification    \
 --train_last_checkpoint ./checkpoints_prog/checkpoints/checkpoint_multi_prog/model.pt


# evaluation
# python main.py   --model_type 'bert' \
#   --do_eval  \
#  --train_file  $train_file \
#  --predict_file  $predict_file \
#  --model_name_or_path  $model_name_or_path \
#  --output_dir ./checkpoints_prog \
#  --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 12  \
#  --gradient_accumulation_steps 1 \
#  --max_grad_norm inf \
#  --adam_epsilon 1e-6 --adam_beta_2 0.98 --weight_decay 0.01  \
#  --warmup_proportion 0.06 --num_train_epochs 300 --overwrite_output_dir  --save_freq 1   \
#  --model_descri 'multi_prog'  --task prog_classification  --multi_prog \
#  --eval_checkpoint  /home/jujz/FinQA/code/program_selection/checkpoints_prog/checkpoints/checkpoint_multi_prog/model.pt
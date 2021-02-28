#! /usr/bin/env bash
python sentiment_beta.py \
  --data_dir '../input' \
  --bert_model_dir '../input/pre_training_models/chinese_L-12_H-768_A-12' \
  --output_dir '../output/models/' \
  --checkpoint '../output/models/checkpoint-33426' \
  --max_seq_length 300 \
  --do_predict \
  --do_lower_case \
  --train_batch_size 60 \
  --gradient_accumulation_steps 3 \
  --predict_batch_size 15 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 

#!/bin/bash

# Train models on wiki+coco dataset.
# For SimCSE baseline, you just need to (1) set new output_dir (2) --framework simcse (3) remove --feature_file

IMG=data/train_coco_ViT_L14.json
CAPTION=data/coco_random_captions.txt
TEXT=data/wiki1m_for_simcse.txt

SEED=1234
MODEL=bert-base-uncased
LR=3e-5
BATCH=64
EPS=3
LBD=0.01
MARGIN1=0.15
MARGIN2=0.15
SCORE_BASE=0.8

OUT_DIR=result/mix_coco/${SEED}/mcse

python src/train_mix.py \
    --framework mcse \
    --model_name_or_path $MODEL \
    --text_file $TEXT \
    --caption_file $CAPTION  \
    --feature_file $IMG \
    --output_dir $OUT_DIR \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH \
    --num_train_epochs $EPS \
    --seed $SEED  \
    --margin1 $MARGIN1 \
    --margin2 $MARGIN2 \
    --score_base $SCORE_BASE \
    --lbd $LBD ${@:5}


python simcse_to_huggingface.py --path $OUT_DIR

python src/evaluation.py \
      --model_name_or_path $OUT_DIR \
      --pooler cls_before_pooler \
      --task_set sts \
      --mode test

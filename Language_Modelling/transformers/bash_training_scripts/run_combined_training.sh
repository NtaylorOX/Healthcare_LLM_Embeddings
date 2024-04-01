#!/usr/bin/env bash
# one GPU with only mlm loss
CUDA_VISIBLE_DEVICES=1 python run_combined_pretraining.py --train_batch_size 8 --eval_batch_size 8 --max_epochs 2 --contrastive_loss_weight 1.0 --compute_mlm_loss_only
# with note loss only
# CUDA_VISIBLE_DEVICES=0 python run_combined_pretraining.py --train_batch_size 8 --eval_batch_size 8 --max_epochs 2 --contrastive_loss_weight 1.0 --compute_note_loss_only
#!/usr/bin/env bash
python main.py --config dropout_fc_cl_with_aug.yaml
python main.py --config augmentation_train1000.yaml
python main.py --config dropout_fc_cl_with_aug_train1000.yaml

#!/bin/sh

experiment=$(date +'%m/%d/%Y-%H:%M')
mkdir -p logs
python ./scripts/training/retrieve.py > logs/pipeline.log
python ./scripts/training/preprocess.py > logs/pipeline.log
python ./scripts/training/split.py > logs/pipeline.log
python ./scripts/training/train_val_test.py --model sklearn.tree.DecisionTreeClassifier --experiment ${experiment} > logs/pipeline.log
python ./scripts/training/train_val_test.py --model sklearn.ensemble.RandomForestClassifier --experiment ${experiment} > logs/pipeline.log
python ./scripts/training/train_val_test.py --model sklearn.svm.SVC --experiment ${experiment} > logs/pipeline.log
python ./scripts/training/train_val_test.py --model src.model.pytorch.MultiClassClassifier --experiment ${experiment} > logs/pipeline.log
python ./scripts/training/register.py --experiment ${experiment} > logs/pipeline.log

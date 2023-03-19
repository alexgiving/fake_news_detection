python3 script_train.py \
    --device cpu \
    --fake-path "./dataset/Fake.csv" \
    --true-path "./dataset/True.csv" \
    --cache-folder "./cache/" \
    --batch-size 10 \
    --epoches 25 \
    --last-states 2 \
    --arch deep_normalized_class_bert \
    --optim adam

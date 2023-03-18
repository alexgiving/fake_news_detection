python3 script_train.py \
    --device cpu \
    --fake-path "./dataset/Fake.csv" \
    --true-path "./dataset/True.csv" \
    --cache-folder "./cache/" \
    --batch-size 10 \
    --epoches 10 \
    --last-states 1 \
    --arch normalized_class_bert

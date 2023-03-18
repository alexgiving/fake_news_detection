# Fake news detection

## How to run training

1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Set PYTHONPATH variable
```bash
export PYTHONPATH=$(pwd)
```

3. Download dataset from Kaggle
Place them into `./dataset` folder
* [Fake.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv)
* [True.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv)


4. Run training pipeline via command below

```bash
python3 script_train.py \
    --device gpu \
    --fake-path "./dataset/Fake.csv" \
    --true-path "./dataset/True.csv" \
    --cache-folder "./cache/" \
    --batch-size 10 \
    --epoches 25 \
    --last-states 1 \
    --arch class_bert
```

## Supported models

1. Classification Bert Model
2. Classification Bert Model with batch normalization
3. Classification Bert Model with batch normalization and hidden linear layer

Run `script_train.py -h` to get help
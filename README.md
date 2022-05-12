# Devnet
Unofficial pytorch implementation of deviation network for table data.

paper of deviation network  
https://arxiv.org/abs/1911.08623

original keras implementation by authors of the paper is here  
https://github.com/GuansongPang/deviation-network

## Setup
install poetry followed by  
https://python-poetry.org/docs/master/#installing-with-the-official-installer

install dependencies
```
poetry install
```

## Usage
train model with train/eval.csv under dataroot
```
poetry python src/main.py dataroot=data/debug epochs=10 eval_interval=10
```

predict score and output result
```
poetry python src/main.py predict_only=true predict_input=data/debug/eval.csv model_path=data/debug/models/example.pth
```

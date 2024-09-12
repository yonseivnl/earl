# Online Continual Learning

## Requirements Install
```bash
pip install -r requirements.txt
```

## Download Datasets
move to the dataset directory
```bash
cd dataset
```

install dataset
- ex. CIFAR-10
```bash
bash cifar10.sh 
```

## Run Experiments
```bash
bash ex.sh
```
<!-- 
- cifar10
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --logdir logs/[logging_dir] --method [method_name] --model [model_name] --random_seed [random_seed] --stream [setup_name]
```


- cifar100
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --logdir logs/[logging_dir] --method [method_name] --model [model_name] --random_seed [random_seed] --stream [setup_name] --dataset cifar100 --memory_size 2000 --num_iters 3
``` --> 
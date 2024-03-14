# Online Continual Learning

## Requirements

- `pytorch>=2.0`
- `torchvision>=0.15`
- `pytorch-ignite>=0.4.12`

## Training

- cifar10
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --logdir logs/[logging_dir] --method [method_name] --model [model_name] --random_seed [random_seed] --stream [setup_name]
```


- cifar100
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --logdir logs/[logging_dir] --method [method_name] --model [model_name] --random_seed [random_seed] --stream [setup_name] --dataset cifar100 --memory_size 2000 --num_iters 3
```
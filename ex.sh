LOG="logs/TRAIN_er_clear10"
MODE="er"
SEEDS="1"
DATASET="clear10" # cifar10, cifar100, tinyimagenet, imagenet
STREAM="disjoint" # disjoint, continuous, iblurry
NUM_CLASSES=1024
USE_DISTILLATION='--use_distillation'
USE_PSEUDO_STREAM='--use_pseudo_stream'
USE_RESIDUAL_ADDITION='--residual_addition'
DOMAIN_INCRE_EVAL='--domain_incre_eval'
EMA_COEFF="0.999"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1 
    EVAL_PERIOD=100 BATCHSIZE=16
    BASEINIT_SAMPLES=30000 FEAT_DIM=8 FEAT_MEM_SIZE=750
    SAMPLES_PER_TASK=10000

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=3 
    EVAL_PERIOD=100 BATCHSIZE=32
    BASEINIT_SAMPLES=30000 FEAT_DIM=8 FEAT_MEM_SIZE=3000
    SAMPLES_PER_TASK=10000

elif [ "$DATASET" == "clear10" ]; then
    MEM_SIZE=500 ONLINE_ITER=3
    EVAL_PERIOD=200 BATCHSIZE=92
    BASEINIT_SAMPLES=9900 FEAT_DIM=14 FEAT_MEM_SIZE=24000
    SAMPLES_PER_TASK=10000

elif [ "$DATASET" == "clear100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=3 
    EVAL_PERIOD=500 BATCHSIZE=43
    BASEINIT_SAMPLES=39000 FEAT_DIM=14 FEAT_MEM_SIZE=48000
    SAMPLES_PER_TASK=10000

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=4000 ONLINE_ITER=3
    EVAL_PERIOD=100 BATCHSIZE=32
    BASEINIT_SAMPLES=60000 FEAT_DIM=16 FEAT_MEM_SIZE=6000
    SAMPLES_PER_TASK=20000

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=20000 ONLINE_ITER=1
    EVAL_PERIOD=1000 BATCHSIZE=256
    BASEINIT_SAMPLES=128117 FEAT_DIM=7 FEAT_MEM_SIZE=1920000 #10% baseinit
    SAMPLES_PER_TASK=256233
else
    echo "Undefined setting"
    exit 1
fi

for RANDOM_SEED in $SEEDS
do
        CUDA_VISIBLE_DEVICES=6 python main.py --logdir $LOG $USE_RESIDUAL_ADDITION $USE_DISTILLATION $USE_PSEUDO_STREAM --pseudo_cls_warmup 5 --pseudo_batch_size 2 --pseudo_cls_interval 2  --knn_top_k 25 --distill_coeff 0.1 --residual_num 50 \
        --baseinit_samples $BASEINIT_SAMPLES --spatial_feat_dim $FEAT_DIM --feat_mem_size $FEAT_MEM_SIZE --ema_coeff $EMA_COEFF --samples_per_task $SAMPLES_PER_TASK \
        --method $MODE --random_seed $RANDOM_SEED --stream $STREAM --dataset $DATASET --batch_size $BATCHSIZE --num_classes $NUM_CLASSES --memory_size $MEM_SIZE --num_iters $ONLINE_ITER --test_freq $EVAL_PERIOD
done
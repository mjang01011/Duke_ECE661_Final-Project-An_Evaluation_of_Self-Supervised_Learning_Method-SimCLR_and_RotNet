## SSL SimCLR Linear Evaluation
python3 run_exp_eh.py  \
    --seed 0 --dataset CIFAR10 --network simclr_resnet50 \
    --num_workers 2 \
    --criterion CrossEntropyLoss \
    --optimizer Adam \
    --lr 1e-3 --weight_decay 1e-6\
    --epochs 10 --batch_size 128 \
    --train True \
    --load_model saved_model/ssl_pretrain_cifar10/model.pth\
    --model_dir saved_model/linear_eval --model_name ssl_lineval\
    --wandb True \

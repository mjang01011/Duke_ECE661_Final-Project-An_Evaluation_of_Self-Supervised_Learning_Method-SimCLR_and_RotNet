
## SSL-SimCLR Training SCRIPT
python3 run_exp_eh.py  \
    --seed 0 --dataset CIFAR10 \
    --num_workers 2 \
    --criterion nt-xent \
    --optimizer Adam \
    --lr 1e-3 --weight_decay 1e-6\
    --network simclr_resnet50\
    --epochs 5000 --batch_size 128\
    --simclr_proj_out 128\
    --train True\
    --model_dir saved_model/ssl_pretrain_cifar10/resnet50 \
    --model_name ssl_p128_b128 \
    --wandb True \

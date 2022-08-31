python main.py \
    -model dualprompt \
    -backbone vit_tiny_patch16_224_in21k \
    -pos_g 1 2  -len_g 5  -pos_e 3 4 5 -len_e 20 \
    -prompt_func prefix_tuning \
    -criterion custom \
    -optimizer adam \
    -lr 0.001 \
    -scheduler const \
    -batch 32 -step 128 -epochs 5 -log 4 -num_task 10 -task_govn CIL -worldsize 1 \
    -dataset CIFAR100 --dataset_path /mnt/e/datasets/cifar100/ \
    -save saved/cifar100/l2p/ \
    -rank 1 --multi --console

python main.py \
    -model l2p \
    -backbone vit_tiny_patch16_224_in21k \
    -criterion custom \
    -optimizer adam \
    -lr 0.03 \
    -scheduler const \
    -batch 32 -step 128 -epochs 5 -log 4 -num_task 10 -task_govn CIL -worldsize 1 \
    -dataset CIFAR100 --dataset_path /mnt/e/datasets/cifar100/ \
    -save saved/cifar100/l2p/ \
    --console

# ours*******************************************************
CUDA_VISIBLE_DEVICES='7' \
        python -m torch.distributed.launch --nproc_per_node 1 --use_env --master_port 24893 \
            train_ours_cnt_seq.py -c config/train_ours_enfs.yml -id name_your_id
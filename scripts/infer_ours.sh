##################################### enfssyn
CUDA_VISIBLE_DEVICES='1' \
        python  infer_ours_cnt.py \
                    --model_path /path/to/model \
                    --data_list /path/to/data.txt  \
                    --infer_mode 1 \
                    --output_path /path/to/output \
                    --scale 2 \ # define the SR scale
                    --seqn 3 \
                    --seql 9 \
                    --step_size 1 \
                    --time_bins 1 \
                    --ori_scale down16 \
                    --mode events \
                    --window 2048 \
                    --sliding_window 1024 \
                    --need_gt_frame \
                    --need_gt_events 

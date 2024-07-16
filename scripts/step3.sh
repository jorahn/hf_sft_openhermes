rm -r /tmp/test-clm; NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node 2 ../utils/run_clm.py \
         --model_name_or_path ../out_tk/ \
         --dataset_name jrahn/OpenHermes-2.5_chatml \
         --do_train \
         --do_eval \
         --output_dir /tmp/test-clm \
         --run_name test-clm \
         --per_device_train_batch_size 4 \
         --warmup_steps 500 \
         --weight_decay 0.01 \
         --logging_dir ../logs \
         --logging_steps 10 \
         --eval_strategy steps \
         --eval_steps 1000 \
         --save_steps 1000 \
         --save_total_limit 5 \
         --bf16 \
         --torch_compile \
         --validation_split_percentage 1 \
         --max_eval_samples 500 \
         --num_train_epochs 2
#         --max_steps 200 

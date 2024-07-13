rm -r /tmp/test-clm; NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node 2 ../utils/run_clm.py \
         --model_name_or_path openai-community/gpt2 \
         --dataset_name wikitext \
         --dataset_config_name wikitext-2-raw-v1 \
         --do_train \
         --do_eval \
         --output_dir ../out \
         --per_device_train_batch_size 16 \
         --max_steps 200

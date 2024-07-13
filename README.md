# hf_sft_openhermes
Instruction SFT of llm.c pre-trained model with OpenHermes2.5 dataset using HF

1. use `utils/export_hf.py` from llm.c to convert pre-trained model.bin to HF format (bfloat16 GPT2LMHeadModel)
2. add special tokens for instruction tuning to tokenizer (see https://huggingface.co/docs/transformers/main/en/chat_templating)
3. use `utils/run_clm.py` from transformers for instruction sft (see https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling, https://huggingface.co/docs/transformers/perf_train_gpu_many)
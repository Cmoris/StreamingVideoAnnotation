export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --nproc_per_node 8 \
    bin/generate_transcripts.py \
    --annotation-dir ./all_results/timestamps \
    --batch-size 4 \
    --target-sr 16000 \
    --asr-model-dir Qwen/Qwen3-ASR-1.7B \
    --output-dir ./all_results/transcripts_with_lang \

python \
    bin/merge.py \
    --data-dir ./all_results/transcripts_with_lang \
    --output-dir ./all_results/transcripts_with_lang \

rm ./all_results/transcripts_with_lang/*.rank*.jsonl
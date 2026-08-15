export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT_DIR=./results/transcripts_with_lang_whisper

torchrun \
    --nproc_per_node 8 \
    bin/generate_transcripts.py \
    --annotation-dir ./results/timestamps \
    --batch-size 4 \
    --target-sr 16000 \
    --asr-model-dir openai/whisper-large-v3 \
    --output-dir ${OUTPUT_DIR} \

python \
    bin/merge.py \
    --data-dir ${OUTPUT_DIR} \
    --output-dir ${OUTPUT_DIR} \

rm ${OUTPUT_DIR}/*.rank*.jsonl
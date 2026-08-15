export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --nproc_per_node=1 \
    bin/generate_timestamps.py \
    --audio_dir ./sources \
    --output_dir ./results/timestamps \
    --batch_size 4 \
    --sampling_rate 16000 \
    --threshold 0.5 \
    --min_silence_duration_ms 250 \
    --min_speech_duration_ms 500 \

python \
    bin/merge.py \
    --data-dir ./results/timestamps \
    --output-dir ./results/timestamps \

rm ./results/timestamps/*.rank*.jsonl

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --nproc_per_node=8 \
    bin/generate_timestamps.py \
    --audio_dir /mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads \
    --output_dir ./all_results/timestamps \
    --batch_size 4 \
    --sampling_rate 16000 \
    --threshold 0.5 \
    --min_silence_duration_ms 250 \
    --min_speech_duration_ms 500 \

python \
    bin/merge.py \
    --data-dir ./all_results/timestamps \
    --output-dir ./all_results/timestamps \

rm ./all_results/timestamps/*.rank*.jsonl

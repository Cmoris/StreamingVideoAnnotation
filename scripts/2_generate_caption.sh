export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun \
    --nproc_per_node 4 \
    bin/generate_caption.py \
    --annotation-dir ./results/timestamps \
    --batch-size 4 \
    --target-sr 1600 \
    --sed-model-dir /mnt/nvme/juchen/qwen/Qwen3-Omni-30B-A3B-Instruct \
    --output-dir ./results/caption \

python \
    bin/merge.py \
    --data-dir ./results/caption \
    --output-dir ./results/caption \

rm ./results/caption/*.rank*.jsonl
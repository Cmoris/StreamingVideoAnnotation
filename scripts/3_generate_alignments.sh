python \
    bin/generate_alignment.py \
    --annotation-dir ./all_results/transcripts_with_lang \
    --batch-size 4 \
    --target-sr 16000 \
    --language-codes en,zh \
    --device cuda:0 \
    --output-dir ./all_results/alignments \
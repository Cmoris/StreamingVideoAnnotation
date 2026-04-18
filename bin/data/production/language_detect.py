import json, tqdm, os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


def language_detect_on_device(device_id: int, duration: int, lines):
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=f'cuda:{device_id}')
    results = []
    local_lines = lines
    path = f'/home/muyun/project/Annotation/all_results/language_detect_results_{device_id}_{duration}.json'
    for line in tqdm.tqdm(local_lines, desc=path):
        conversation = json.loads(line)
        paragraph = ' '.join(w for s, e, w, in conversation[1]['content'][0]['text_stream'])
        result = pipe(paragraph, top_k=1, truncation=True)[0]
        results.append(result)
    json.dump(results, open(path, 'w'))

def gather_and_filter(device_id, duration, lines):
    path = f'/home/muyun/project/Annotation/all_results/language_detect_results_{device_id}_{duration}.json'
    results = json.load(open(path))
    english_lines = [line for line, result in zip(lines, results) if result['label'] == 'en' and result['score'] >= 0.9]
    with open(f'/home/muyun/project/Annotation/all_results/livecc/livecc_english_{duration}.jsonl', 'w') as f:
        for line in english_lines:
            f.write(line)
if __name__ == '__main__':
    # local_mt(range(8), language_detect_on_device, desc='language_detect', num_workers=8)
    for duration in range(10, 110, 10):
        lines = open(f'/home/muyun/project/Annotation/all_results/livecc_merged/livecc_merged_{duration}.jsonl').readlines()
        language_detect_on_device(0, duration, lines)
        gather_and_filter(0, duration, lines)
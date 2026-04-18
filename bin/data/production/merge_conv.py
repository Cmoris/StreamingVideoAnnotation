from pathlib import Path
import json
import re

ERROR_VIDEO_PATHS = ['/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV1Pa411S7Me]纪录片《二毛》是贾玉川执导纪录片，于2019年11月23日首映/video_1920x1080_av01.0.08M.08.0.110.01.01.01.0_758546.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV1oS4y1h74X]纪录片《我们为什么肥胖》【全3集】1080P_P2/video_1920x1080_av01.0.08M.08.0.110.01.01.01.0_683581.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV19UFAzjE3B]挑战极限大冒险 第一季全十集 Boundless/video_1920x1080_av01.0.00M.10.0.110.01.01.01.0_783887.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV1Za411d7mS]治愈！“你看不见的我”纪录片（全5讲）双语字幕|豆瓣高分8.6_P4/video_1920x1080_av01.0.08M.08.0.110.01.01.01.0_467145.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV1oS4y1h74X]纪录片《我们为什么肥胖》【全3集】1080P/video_1920x1080_av01.0.08M.08.0.110.01.01.01.0_765126.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV1yDcYznEcK]神秘的深海 第一季全十集 Mysteries of the Deep/video_1920x1080_av01.0.00M.10.0.110.01.01.01.0_834478.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV1GR2CBtEGg]《华尔街》【央视十集大型金融历史纪录片】/video_1280x720_av01.0.00M.10.0.110.01.01.01.0_465045.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV1Wqr5BUEm8]极北之地 第一季全八集 Hard North_P2/video_1920x1080_av01.0.00M.10.0.110.01.01.01.0_1511410.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV1sx41177Pr]【纪录片】亿万富翁的有钱人生【全6P】_P3/video_720x408_av01.0.04M.08.0.110.01.01.01.0_180939.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV18Z4y1c7eR]【纪录片】《约翰·威尔逊的生活指南》S01E01/video_1920x1080_av01.0.00M.10.0.110.01.01.01.0_808548.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV11y4y1w7tW]【Netflix纪录片】《极简主义》中英双语，帮你摆脱焦虑内耗！避免消费主义陷阱！/video_1920x1080_av01.0.08M.08.0.110.01.01.01.0_610326.mp4', 
                     '/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads/[BV1W3411y7Dc]【纪录片】逻辑的乐趣/video_858x482_av01.0.04M.08.0.110.01.01.01.0_262352.mp4']

def get_highest_resolution_video(audio_path: Path) -> str:
    dir_path = audio_path.parent

    video_files = list(dir_path.glob("video_*.mp4"))

    best_video = None
    best_score = -1

    pattern = re.compile(r"video_(\d+)x(\d+)_")

    for v in video_files:
        match = pattern.search(v.name)
        if not match:
            continue

        w, h = map(int, match.groups())
        score = w * h

        if score > best_score:
            best_score = score
            best_video = v

    return str(best_video) if best_video else None

def merge_conversations(convs, max_gap=0.5, max_duration=10):
    """
    Merge adjacent conversations into longer streams
    """

    merged = []
    buffer = None

    for conv in convs:
        video = conv[0]["content"][0]["video"]
        start = conv[0]["content"][0]["video_start"]
        end = conv[0]["content"][0]["video_end"]

        text_stream = conv[1]["content"][0]["text_stream"]

        if buffer is None:
            buffer = conv
            continue

        b_video = buffer[0]["content"][0]["video"]
        b_start = buffer[0]["content"][0]["video_start"]
        b_end = buffer[0]["content"][0]["video_end"]

        gap = start - b_end

        if video == b_video and gap < max_gap and (end - b_start) < max_duration:

            # merge
            buffer[0]["content"][0]["video_end"] = end

            buffer[1]["content"][0]["text_stream"].extend(text_stream)

        else:
            merged.append(buffer)
            buffer = conv

    if buffer is not None:
        merged.append(buffer)

    return merged

def merge_jsonls(jsonl_files, output_path, max_gap=0.5, max_duration=10):
    all_convs = []
    for jsonl_file in jsonl_files:
        merged_convs = []
        with open(jsonl_file, "r") as f:
            lines = f.readlines()
            for line in lines[:-1]:
                conv = json.loads(line)
                audio = Path(conv[0]["content"][0]["audio"])
                start, end = conv[0]["content"][0]["start"], conv[0]["content"][0]["end"]
                video = get_highest_resolution_video(audio)
                if video in ERROR_VIDEO_PATHS:
                    continue
                conv[0]["content"] = [{"type":"video", "video": video, "video_start": start, "video_end": end}, {"type":"text", "text":"Describe the video."}]
                merged_convs.append(conv)
        merged_convs = list(set(json.dumps(conv) for conv in merged_convs))
        merged_convs = [json.loads(conv) for conv in merged_convs]
        merged_convs = sorted(merged_convs, key=lambda x: (x[0]["content"][0]["video"], x[0]["content"][0]["video_start"]))
        merged_convs = merge_conversations(merged_convs, max_gap=max_gap, max_duration=max_duration)
        all_convs.extend(merged_convs)
    with open(output_path, "w") as f:
        for conv in all_convs:
            f.write(json.dumps(conv, ensure_ascii=True) + "\n")
            
if __name__ == '__main__':
    for duration in range(10, 110, 10):
        jsonl_files = [x for x in Path("/home/muyun/project/Annotation/all_results/alignments").glob("*.jsonl")]
        merge_jsonls(jsonl_files, 
                    f"/home/muyun/project/Annotation/all_results/livecc_merged/livecc_merged_{duration}.jsonl",
                    max_gap=5,
                    max_duration=duration)

import json
from typing import List, Dict, Any

def calculate_stream_duration(conversation: Dict[str, Any]) -> float:
    """计算单个conversation中stream的时长"""
    for content_item in conversation.get("content", []):
        if content_item.get("type") == "video":
            video_start = content_item.get("video_start", 0)
            video_end = content_item.get("video_end", 0)
            return video_end - video_start
    return 0

def merge_short_streams(data: List[Dict], min_duration: float = 5.0) -> List[Dict]:
    """
    合并时长过短的stream到上一个conversation
    
    Args:
        data: 原始对话列表
        min_duration: 最小时长阈值（秒），默认5秒
    
    Returns:
        合并后的对话列表
    """
    if not data:
        return data
    
    merged_data = []
    i = 0
    
    while i < len(data):
        user_conv = data[i][0]
        # 如果是user消息，检查对应的assistant回复时长
        assistant_conv = data[i][1]
        video_path = data[i][0].get("content", [{}])[0].get("video", "")
        duration = calculate_stream_duration(user_conv)
        
        # 如果时长太短，且前面有对话可以合并
        if duration < min_duration and merged_data:
            # 找到上一个user-assistant对
            if len(merged_data) >= 2 and merged_data[-1][1].get("role") == "assistant":
                # 合并video片段和text_stream

                prev_user = merged_data[-1][0]
                prev_assistant = merged_data[-1][1]
                prev_video = prev_user.get("content", [{}])[0].get("video", "")
                
                if prev_video != video_path:

                    # 如果视频路径不同，说明不是同一段视频，不能合并，检查下一条conversation是否可以合并
                    next_user = data[i+1][0] if i + 1 < len(data) else None
                    next_assistant = data[i+1][1] if i + 1 < len(data) else None
                    next_video = next_user.get("content", [{}])[0].get("video", "")
                    if next_video != video_path:
                        i += 1
                        continue
                    for content_item in user_conv.get("content", []):
                        if content_item.get("type") == "video":
                            for current_content in next_user.get("content", []):
                                if current_content.get("type") == "video":
                                    content_item["video_end"] = current_content.get("video_end")
                                    break
                
                    # 合并assistant的text_stream
                    for content_item in assistant_conv.get("content", []):
                        if content_item.get("type") == "text_stream":
                            for current_content in next_assistant.get("content", []):
                                if current_content.get("type") == "text_stream":
                                    content_item["text_stream"].extend(
                                        current_content.get("text_stream", [])
                                    )
                                    break
                    merged_data.append([user_conv, assistant_conv])
                    i += 1
                    continue
                # 更新上一个user的video_end
                for content_item in prev_user.get("content", []):
                    if content_item.get("type") == "video":
                        for current_content in user_conv.get("content", []):
                            if current_content.get("type") == "video":
                                content_item["video_end"] = current_content.get("video_end")
                                break
                
                # 合并assistant的text_stream
                for content_item in prev_assistant.get("content", []):
                    if content_item.get("type") == "text_stream":
                        for current_content in assistant_conv.get("content", []):
                            if current_content.get("type") == "text_stream":
                                content_item["text_stream"].extend(
                                    current_content.get("text_stream", [])
                                )
                                break
                
                # 跳过当前这对conversation
                i += 1
                continue
            
            # 时长足够或没有可合并的对话，正常添加
            merged_data.append([user_conv, assistant_conv])
            i += 1
            continue
        
        # 其他情况直接添加
        merged_data.append([user_conv, assistant_conv])
        i += 1
    
    return merged_data

def main():
    for duration in range(10, 110, 10):
        input_file = f"/home/muyun/project/Annotation/all_results/livecc/livecc_english_{duration}.jsonl"  
        output_file = f"/home/muyun/project/Annotation/all_results/livecc_appended/livecc_english_{duration}_appended.jsonl"  
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        # 设置最小时长阈值（秒）
        MIN_DURATION = 5.0  # 可以根据需要调整这个值
        
        # 合并短stream
        merged_data = merge_short_streams(data, min_duration=MIN_DURATION)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in merged_data:
                json.dump(item, f, ensure_ascii=True)
                f.write('\n')
        
        # 打印统计信息
        print(f"原始对话数量: {len(data)}")
        print(f"合并后对话数量: {len(merged_data)}")
        print(f"合并了 {len(data) - len(merged_data)} 个对话")

if __name__ == "__main__":
    main()
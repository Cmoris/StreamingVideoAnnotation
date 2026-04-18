import pysubs2
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Union

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cn-file",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--ja-file",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./results/transcripts"
    )
    
    parser.add_argument(
        "--fontsize",
        type=int,
        default=40
    )

class SubtitleGenerator:
    """字幕生成器类"""
    
    def __init__(self, fontsize: int = 40):
        """
        初始化字幕生成器
        
        Args:
            fontsize: 字体大小，默认40
        """
        self.fontsize = fontsize
    
    def _load_subtitles(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        加载字幕文件
        
        Args:
            file_path: 字幕文件路径
            
        Returns:
            字幕列表
            
        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"字幕文件不存在: {file_path}")
            
        subtitles = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines[:-1]):
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            subtitle = json.loads(line)
                            subtitles.append(subtitle)
                        except json.JSONDecodeError as e:
                            raise json.JSONDecodeError(
                                f"第{line_num}行JSON格式错误: {e.msg}", 
                                e.doc, 
                                e.pos
                            )
        except Exception as e:
            raise Exception(f"读取文件 {file_path} 时出错: {str(e)}")
            
        return subtitles
    
    def _add_subtitles_to_ssa(self, subs: pysubs2.SSAFile, subtitles: List[Dict], 
                             text_field: str = "translation") -> None:
        """
        将字幕添加到SSA文件中
        
        Args:
            subs: SSA文件对象
            subtitles: 字幕列表
            text_field: 文本字段名
        """
        for subtitle in subtitles:
            try:
                start_ms = int(subtitle["start"]*1000)
                end_ms = int(subtitle["end"]*1000)
                text = subtitle.get(text_field, "")
                subs.append(pysubs2.SSAEvent(
                    start=start_ms, 
                    end=end_ms, 
                    text=text
                ))
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"字幕数据格式错误: {subtitle}, 错误: {str(e)}")
    
    def generate_subtitle(self, 
                         cn_file: Union[str, Path], 
                         ja_file: Optional[Union[str, Path]] = None,
                         output_path: Union[str, Path] = "output.ass") -> None:
        """
        生成字幕文件
        
        Args:
            cn_file: 中文字幕文件路径
            ja_file: 日文字幕文件路径（可选）
            output_path: 输出文件路径
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 字幕数据格式错误
            Exception: 其他错误
        """
        try:
            # 加载中文字幕
            cn_subtitles = []
            if cn_file is not None:
                cn_subtitles = self._load_subtitles(cn_file)
            
            # 如果提供了日文字幕文件，则加载
            ja_subtitles = []
            if ja_file is not None:
                ja_subtitles = self._load_subtitles(ja_file)
            
            # 创建SSA文件
            subs = pysubs2.SSAFile()
            subs.styles["Default"].fontsize = self.fontsize
            
            # 添加中文字幕（使用translation字段）
            if cn_subtitles:
                self._add_subtitles_to_ssa(subs, cn_subtitles, "translation")
            
            # 添加日文字幕（使用text字段）
            if ja_subtitles:
                self._add_subtitles_to_ssa(subs, ja_subtitles, "text")
            
            # 保存文件
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subs.save(str(output_path))
            
            print(f"字幕文件已保存到: {output_path}")
            
        except Exception as e:
            raise Exception(f"生成字幕文件失败: {str(e)}")


def save_subtitle():
    """
    生成字幕文件（函数接口）
    
    Args:
        cn_file: 中文字幕文件路径
        ja_file: 日文字幕文件路径（可选）
        output_path: 输出文件路径
        fontsize: 字体大小
        
    Example:
        >>> save_subtitle("cn_subs.json", "ja_subs.json", "output.ass")
        >>> save_subtitle("cn_subs.json", output_path="chinese_only.ass")
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    generator = SubtitleGenerator(fontsize=args.fontsize)
    generator.generate_subtitle(args.cn_file, args.ja_file, args.output_path)


# 使用示例
if __name__ == "__main__":
    save_subtitle()

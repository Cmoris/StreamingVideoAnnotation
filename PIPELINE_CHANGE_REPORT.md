# 时间戳与转录 Pipeline 修改报告

对照基准：Git HEAD `d315755` 与当前工作区。本文只报告现有修改及本次会话实际完成的验证，不把静态推断写成实测结论。代码行号均指当前版本。

**修改归属说明**

开始检查时，以下修改已经存在：

- `bin/generate_timestamps.py`：文件打开方式由文本改为 `wb`；启动时由 `RANK` 改读 `LOCAL_RANK`。
- `bin/audio/env_model.py`：背景窗口的 `range()` 上限已经增加 `+1`。
- `bin/data/dataset.py`：已经删除两段旧的、完全注释掉的 `read_audio` 实现。

这些不是我最初引入的修改。后续我在此基础上继续修复。因此，`git diff` 中看到的全部差异，不能直接全部归属于我。

**1. 最重要的实际错误：时间单位与采样率混用**

位置：`bin/data/dataset.py:28`、`:186`；`bin/generate_timestamps.py:136`；`bin/data/collator.py:103`；`bin/generate_transcripts.py:182`。

原来的链路：先把音频重采样为 16 kHz，再由 VAD 输出采样点；读取片段时却用源文件的采样率把这些采样点换成秒。

```python
# 修改前：采样点来自重采样音频，除数却来自原文件。
audio_start = float(audio_start) / orig_sr

# 修改后：时间戳明确记录自己的采样率，先换成秒再裁剪。
timestamp_sr = segments.get("sampling_rate", DEFAULT_SAMPLE_RATE)
audio = {"audio": audio_path,
         "start": start / timestamp_sr,
         "end": end / timestamp_sr}
```

例如，16 kHz 时间线上第 `16000` 个采样点表示第 1 秒；原文件若是 48 kHz，旧算法会误读为约 0.333 秒。因此，输出看起来可能有正确的区间数值，但模型实际听到的是另一个位置的声音。

修改后，时间戳 JSONL 新增 `sampling_rate`，Dataset 返回的 `start/end` 统一为秒；Collator 用 `float64` 保留小数；转录写出时不再除一次采样率。

这里必须区分：**原来 Collator 用整数存采样点，以及转录阶段最后除以采样率，本身没有错。** 改成浮点数是配合 Dataset 改为秒的接口调整，否则新接口下会截掉小数或重复换算。

兼容约定：没有 `sampling_rate` 的旧时间戳按 16 kHz 解释。旧文件若实际来自其他采样率，需要补充元数据或重新生成。

**2. 文件内容与字节索引不一致**

位置：`bin/generate_timestamps.py:94`、`bin/generate_transcripts.py:107`、`bin/merge.py:36`。

原代码用 `len(line.encode("utf-8"))` 计算字节长度，但文件以文本模式写入。Windows 下 `\n` 可能写成 `\r\n`，每条记录的实际长度会比计算值多 1 字节。Dataset 随后使用二进制 `seek()`，从第二条开始就可能定位错误。

```python
# Git 原版
f = open(out_file, "w", encoding="utf-8")
lengths.append(len(line.encode("utf-8")))
f.write(line)

# 修改后：计算和写出的都是同一套 UTF-8 字节。
f = open(out_file, "wb")
lengths.append(len(line.encode("utf-8")))
f.write(line.encode("utf-8"))
```

JSONL 最后一行的索引数组也改为编码后写出。`merge.py` 的读取编码显式设为 UTF-8，避免 Windows 系统默认编码影响日文、中文内容。

接手时还有一个直接报错的问题：你已经把时间戳文件打开方式改为 `wb`，但正文和末尾索引仍写入字符串。这会触发二进制文件不接受 `str` 的 `TypeError`。我补齐了两处编码。**这个字符串/二进制混写状态存在于接手时的工作区，不是 Git HEAD 中的原始状态。**

**3. 真实音频发现的问题：伪装成 MP4 的原始 AAC**

位置：`bin/data/dataset.py:22`、`:47`、`:84`。

原实现直接信任 FFprobe 的 `duration`。测试文件 `rajira1.mp4` 的实际封装却是原始 ADTS AAC，估计时长不准确：

| 项目 | 实测值 |
|---|---:|
| FFprobe 估计时长 | 2959.616553 秒 |
| 完整解码时长 | 3000.064 秒 |
| 原流程可能截掉的尾部 | 40.447447 秒 |

我最初尝试统一改为 FFmpeg 输入端 `ss/t` 快速定位。真实对照发现，这对该原始 AAC 文件不可靠，第 30 分钟的切片与从头解码裁剪结果不一致。因此，最终实现分两种情况：

- 普通非原始 AAC 输入：通过 `ss/t` 定位，避免每段都从文件开头解码。
- 原始 AAC：完整解码一次，以实际样本数确定时长，然后按重采样后的采样点切片；用 `lru_cache(maxsize=1)` 缓存最近一个文件。

验证结果：该 AAC 在开头、30 秒、第 30 分钟及尾部的切片与完整解码对应区间一致；独立 FFmpeg 参考切片的最大差值为 `0.0`。

代价：原始 AAC 缓存整个解码波形，会占用内存；它不是流式解码实现。该文件的 16 kHz 单声道 float32 波形约占 192 MB，处理中还会有其他数组和张量。

**4. ASR 依赖、语言识别与 Whisper 输入**

位置：`bin/audio/asr_model.py:9`、`:35`、`:76`、`:91`；`bin/generate_transcripts.py:147`。

| 修改前 | 修改后 | 原问题或修改原因 |
|---|---|---|
| 文件顶层导入 `qwen_asr` | 仅在初始化 Qwen 时导入 | 没装 Qwen 包时，即使只用 Whisper 也无法启动 |
| Whisper 默认 `ja`，Qwen 默认 `Japanese` | 默认自动识别，保留显式 `--language` | 固定日语适合原先的日语场景，不是日语场景下的错误；限制了其他语言输入 |
| 语言字段直接填写配置值或 `unknown` | 使用模型语言检测结果，并传给生成接口 | 避免把配置标签误当成已检测的语言 |
| `detect_language()` 用 `generate(max_new_tokens=1)`，且输入未移到模型设备 | 用 `model.detect_language()`，匹配设备及 dtype | 原方法可能存在设备不匹配，也不可靠地保证取到语言 token |
| 检测输入 `padding=True` | `padding="max_length"`，检测上下文截断到 30 秒 | 实测曾出现 2038 帧输入，而 Whisper 编码器要求 3000 帧 |
| 转录输入不显式处理采样率、attention mask 和长输入 | 非 16 kHz 先重采样；传 mask；不截断转录输入；启用时间戳生成 | 原接口虽接收采样率，处理器却始终按 16 kHz 解释；同时补齐长输入处理配置 |
| Qwen 出错进入 `breakpoint()` | 保留原异常向上传播 | 批处理可能停在调试器里，无法正常失败退出 |

另一个问题是在**我最初改为逐片段自动检测后**发现的：这段 m4a 的几个约 1 秒日语片段被识别为韩语。这不是原来固定日语代码的错误，而是首次修复方案带来的短片段识别不稳定。

最终方案：每个源文件使用前 30 秒上下文检测一次语言并缓存；本次音频不足 30 秒，所以实际使用整段。各片段沿用该语言，并将相同语言传给 Whisper 的生成接口。重跑后，7 条记录的语言均为 `ja`。

限制：当前自动识别策略假设一个文件主要使用同一种语言。长文件前 30 秒与后面语言不同、或频繁切换语言时，仍需另行调整；本次没有验证混合语言场景。

**5. 数据加载、异常处理和空结果**

位置：`bin/data/dataset.py:118`、`:158`；`bin/generate_transcripts.py:91`、`:107`、`:167`；`bin/merge.py:36`。

| 修改前 | 修改后 | 影响 |
|---|---|---|
| 时间戳输入只遍历目录 | 支持目录或单个音频文件 | 可以直接测试指定 m4a |
| ASR 读取目录下全部 JSONL | 排除 `.rank` 分片 | 避免分片和合并文件同时存在时重复转录 |
| 非语音也先解码，之后在 batch 中过滤 | ASR 可提前只保留 speech | 减少无用的环境音解码 |
| 片段为空进入断点，再重复读取 | 抛出含文件及区间信息的异常 | 避免无人值守任务挂住 |
| 无效区间尝试凑出一个采样点 | 显式报错 | 不再掩盖错误的时间边界 |
| 直接 `zip()` 输入与模型输出 | 先检查数量一致 | 避免模型少返回结果时静默丢失片段 |
| 零段合并直接报错 | 支持合法的空索引 `[]` | 纯静音文件也能产生有效空结果 |
| 转录跳过所有静音输入后没有输出文件 | rank 0 为每个输入预建空文件 | 避免整次输入无语音时无法合并 |
| 合并输入不存在时静默结束 | 抛出明确错误 | 更容易发现上游未产出结果 |
| 合并目录只创建最后一层 | `parents=True` | 支持新的多层输出目录 |

**6. 多进程编号、CPU 支持及配置**

位置：`bin/generate_timestamps.py:31`、`:162`；`bin/generate_transcripts.py:47`、`:200`；`bin/audio/asr_model.py:19`。

- 原先把全局 `RANK` 当作 GPU 编号；现在全局 rank 用于数据分片及输出文件编号，`LOCAL_RANK` 用于本机 GPU。多节点时可避免用全局编号访问不存在的本机 GPU。接手时仅把时间戳阶段改读 `LOCAL_RANK`，又会使不同节点的数据分片重复；最终分开传递两个编号。
- 新增 `--device`，只有 CUDA 设备才调用 `torch.cuda.set_device()`，可以显式选择 CPU。
- Whisper 的 CPU 使用 float32，CUDA 使用 bfloat16；输入匹配模型 dtype。
- Whisper 由 `device_map=device` 改为加载后 `.to(device)`，单设备推理不再依赖 Accelerate 的设备映射。
- 固定的 16/8 个 DataLoader workers 改为参数，默认 0。原值并非必然错误，但对 Windows 单文件测试会带来较大的进程启动和内存开销。
- 增加转录进度显示和未知模型后端的明确报错；支持 `--asr-backend`，不再只能靠路径中是否含有 whisper/qwen3 判断。

多节点、CPU Qwen 推理都没有实测；CPU 支持的实际验证对象是 Whisper。

**7. 环境事件检测的边界修正**

位置：`bin/audio/env_model.py:39`、`:76`、`:128`、`:137`。

| 修改 | 原因 | 验证程度 |
|---|---|---|
| 极短波形提前返回空环境事件 | 防止反射 padding 对过短输入报错 | 静态修正；未单独构造真实短音频验证 |
| `onset_frames * 512` → `(onset_frames + 1) * 512` | 差分数组第 i 项对应后一个频谱帧，补回一帧偏移，即 16 kHz 下约 32 ms | 静态修正；未做人工作标注的事件定位评估 |
| 背景窗口上限增加 `+1` | 保留恰好容纳完整窗口的情况 | 接手前已存在的修正，我保留了它 |
| 对 speech 与 environment 区间先取合并结果 | 原先直接累加重叠长度，会把交叠区域重复计数 | 静态修正；真实 m4a 完成环境事件生成，但未独立评价事件准确性 |

这里的环境事件仍是频谱变化启发式检测，不代表已经验证了声音类别或事件语义。

**8. 启动脚本和新增文件**

| 文件 | 前后变化 |
|---|---|
| `scripts/1_generate_timestamps.sh` | 增加失败即停、定位项目目录、环境变量配置和独立 rank 临时目录；不再硬编码可见 GPU 列表 |
| `scripts/2_generate_transcripts.sh` | 同上；进程数由固定 8 改为可配置、默认 1，适用于单卡机器 |
| `scripts/pipeline.sh` | 原文件为空；现在依次执行时间戳与转录脚本 |
| `scripts/pipeline.bat` | 新增 Windows 入口，支持 `all`、`timestamps`、`transcripts` 和可选音频路径；每一步检查退出码 |
| `.gitattributes` | 新增 `*.sh text eol=lf`，避免 Git 把 shell 脚本转换为 CRLF |
| `bin/data/test_dataset_collators.py` | 更新秒级接口断言，补充采样率与原始 AAC 回归用例 |
| `bin/test_pipeline_io.py` | 新增 UTF-8 字节偏移、排序和空索引测试 |
| `bin/test_asr_backend.py` | 新增语言检测、生成语言提示及输入 padding 的模拟回归测试 |

Shell 原来的固定 8 卡配置在 8 卡机器上不一定有错；对本次单卡环境不适用。原 shell 脚本没有失败即停，失败后仍可能继续合并和删除分片，这是实际的运行可靠性问题。

Shell 的独立 rank 目录避免混入旧分片，但不会自动删除旧的已合并输出。BAT 则保留 rank 目录便于检查；不同任务建议使用独立输出位置。BAT 只验证过帮助分支，没有在创建后重新跑一遍完整推理。

会话中曾创建 `scripts/pipeline.ps1`，但本次生成报告时该文件已不在工作区，因此不列为当前交付文件。旧注释代码的删除不影响运行行为，也不计作错误修复。

**9. 实际验证记录与边界**

| 验证对象 | 实际结果 |
|---|---|
| 回归用例 | 会话中累计验证 10 项：7 项 Dataset/Collator、2 项文件合并/索引、1 项 Whisper 模拟测试；随后语言上下文修改后，相关 Whisper 测试再次通过 |
| `rajira1.mp4` | 完成真实解码、切片、时长与尾部完整性验证；没有完成该 50 分钟文件的整段转录 |
| `2023-02-10_20-00-38_33476.m4a` CPU | 时间戳、转录和合并跑通；12 条时间戳中 7 条 speech、5 条 environment_event；生成 7 条转录，检查数量、时间对应关系和字节索引 |
| 同一 m4a GPU | 复用 CPU 阶段已有时间戳；7 条 GPU 转录成功，转录进度统计约 5.3 秒，不含模型加载时间；检查语言、非空文本、时间范围和索引 |
| 编译与格式 | 修改过程中 Python 编译、`git diff --check` 和当时的 PowerShell 语法检查通过 |

CUDA 故障属于环境兼容问题：最初 `torch 2.14.0+cu126` 没有 RTX 5070 所需的 `sm_120` 内核；用户重新安装后实测 `torch 2.15.0+cu134` 包含 `sm_120`，矩阵计算与 BF16 支持检查通过。它不是上述时间戳算法造成的错误。

最近一次环境检测缺少 `torchaudio`，所以**不能把最后一次 GPU 验证说成时间戳和转录全链路 GPU 验证**。本报告未再次重跑环境检测或推理。此前的测试产物可能已被清理，本文记录的是会话中观察到的执行结果。

没有人工参考转录，未计算 WER/CER，也没有确认每个日文词都正确。未实测 Qwen、分布式多节点、长输入 Whisper 转录及全部环境事件边界条件。Caption、alignment 和 subtitle 阶段不在本次完整验证范围内。

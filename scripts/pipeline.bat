@echo off
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"
set "INPUT_AUDIO=%~f2"
cd /d "%~dp0.."

rem Edit these paths when changing the environment, audio, or model.
set "PYTHON=C:\APP\Miniconda\envs\ASR\python.exe"
set "FFMPEG_BIN=C:\APP\ffmpeg-2026-10-19-git-2344518cc5-essentials_build\bin"
set "AUDIO=C:\IDOL\中西阿尔诺\video\noginono\乃木坂46のの_20261108_1800-1830.wav"
set "MODEL=C:\Users\12060\.cache\huggingface\hub\models--openai--whisper-large-v3\snapshots\06f233fe06e710322aca913c1bc4249a0d71fce1"
@REM  set "MODEL=C:\Users\12060\.cache\huggingface\hub\models--Qwen--Qwen3-ASR-1.7B\snapshots\7278e1e70fe206f11671096ffdd38061171dd6e5"
set "DEVICE=cuda:0"
set "FONTSIZE=40"
set "PATH=%FFMPEG_BIN%;%PATH%"
if defined INPUT_AUDIO set "AUDIO=%INPUT_AUDIO%"
for %%F in ("%AUDIO%") do set "OUTPUT=%CD%\results_whisper\%%~nF"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"
if /i "%MODE%"=="timestamps" goto timestamps
if /i "%MODE%"=="transcripts" goto transcripts
if /i "%MODE%"=="subtitles" goto subtitles
if /i "%MODE%"=="all" goto timestamps
echo Usage: %~nx0 [all^|timestamps^|transcripts^|subtitles] [audio_path]
exit /b 2

:timestamps
"%PYTHON%" bin\generate_timestamps.py ^
    --audio_dir "%AUDIO%" ^
    --output_dir "%OUTPUT%\timestamp_ranks" ^
    --batch_size 1 --num_workers 0 --device "%DEVICE%"
if errorlevel 1 goto failed
"%PYTHON%" bin\merge.py --data-dir "%OUTPUT%\timestamp_ranks" --output-dir "%OUTPUT%\timestamps"
if errorlevel 1 goto failed
if /i "%MODE%"=="timestamps" goto done

:transcripts
"%PYTHON%" bin\generate_transcripts.py ^
    --annotation-dir "%OUTPUT%\timestamps" ^
    --output-dir "%OUTPUT%\transcript_ranks" ^
    --language "Japanese" ^
    --asr-model-dir "%MODEL%" ^
    --batch-size 1 --num-workers 0 --device "%DEVICE%"
if errorlevel 1 goto failed
"%PYTHON%" bin\merge.py --data-dir "%OUTPUT%\transcript_ranks" --output-dir "%OUTPUT%\transcripts"
if errorlevel 1 goto failed
if /i "%MODE%"=="transcripts" goto done

:subtitles
"%PYTHON%" bin\generate_subtitle.py ^
    --input-dir "%OUTPUT%\transcripts" ^
    --output-dir "%OUTPUT%\subtitles" ^
    --fontsize "%FONTSIZE%"
if errorlevel 1 goto failed

:done
echo Completed: %OUTPUT%
exit /b 0

:failed
echo Pipeline failed. See the error above.
exit /b 1

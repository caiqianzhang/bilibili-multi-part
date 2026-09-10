# bilibili-multi-part

B 站**分 P（多部分）视频**「全部分集」下载 + LLM 笔记生成脚本。

自包含，零项目依赖：只依赖 `yt-dlp` + `requests`（LLM 笔记生成另需 `openai`），
不 import BiliNote 项目内任何模块，可独立运行或被其它脚本 import。

> 这是后端 `BilibiliDownloader.download_all_parts()` 的可移植副本。后端实现若变动，需同步此文件。

## 功能

给一个 B 站视频链接（可带 `?p=N`，会被忽略），枚举**全部**分集，逐集下载：

- **音频**：每集必下（mp3，yt-dlp；**需要系统安装 ffmpeg** 做转码），默认落盘到
  `<输出目录>/<BV号>/` 按课程分子目录（与批量转写、语义索引的扫描约定一致）；
  单集下载失败会记录并继续其余分集，最后汇总（CLI 退出码非 0）
- **字幕**：有就落盘 `.srt`（B 站 player API 直拉，无需下视频）；
  无字幕/无登录态则跳过，不阻断音频
- **笔记**：有转写结果且指定 `--note` 时，调用 OpenAI 兼容 LLM 生成结构化
  Markdown 笔记（切块 → 逐块生成 → 合并 → 时间标记转跳转链接），落 `.md`
- **搜索**：`--search 关键词` 走网页同款 wbi 签名接口（免登录、稳定、
  元数据齐全、自动翻页），失败时退回 yt-dlp 旧接口

## 安装

```bash
cd bilibili-multi-part

# 最小安装（不生成 LLM 笔记）
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.base.txt

# 完整安装（含 LLM 笔记生成）
pip install -r requirements.txt
# 或
pip install -e .
```

## 用法

```bash
# CLI：下载全部分集到 ./bilibili_out
python bilibili_multi_part.py "https://www.bilibili.com/video/BV1bK411W797"

# 指定输出目录 + 音质 + 用 Netscape cookie 文件登录（拿 AI 字幕必需）
python bilibili_multi_part.py "https://www.bilibili.com/video/BVxxx?p=3" \
    --output /data/bilibili --quality medium \
    --cookie-file /path/to/cookies.txt

# 下载 + 用 LLM 生成笔记（每集一份 .md，含可跳回原片的时间链接）
python bilibili_multi_part.py "https://www.bilibili.com/video/BVxxx" \
    --note --api-key sk-xxx --model gpt-4o-mini \
    --format link,summary --style detailed

# 无字幕分集用 BCUT 免费转写，再生成笔记
python bilibili_multi_part.py "https://www.bilibili.com/video/BVxxx" \
    --transcribe bcut --cookie "$BILIBILI_COOKIE" \
    --note --api-key sk-xxx

# 只搜索不下载
python bilibili_multi_part.py --search "关键词"

# 作为库 import
from bilibili_multi_part import BilibiliDownloader, NoteConfig
results = BilibiliDownloader().download_all_parts(url, output_dir="out")
```

## 语义问答（RAG）

对已生成的字幕语料（`$BILI_OUT_ROOT` 下 48 门课 / 约 2700 万字）建立向量索引，
支持"措辞不同也能命中"的语义检索，以及带 **分集 + 时间点 + 跳转链接** 引用的 LLM 问答。
技术栈：`bge-m3`（本地 GPU 编码，1024 维）+ SQLite 单文件 + numpy 暴力检索（毫秒级），
不需要向量数据库服务。用 `.venv-asr` 运行（复用 torch/transformers，另需 `openai` 仅 ask 模式）：

```bash
# 1. 建索引：默认增量（字幕指纹无变化的课程秒级跳过，不加载模型），--force 全量重嵌
#    首次自动经 hf-mirror 下载 bge-m3 约 2.3GB；48 门课全量约 90 分钟
.venv-asr/bin/python semantic_qa.py build
.venv-asr/bin/python semantic_qa.py build --only-bvid BV1h6m8BWE1T

# 2. 语义检索（不需要 LLM key）：命中结果附 BV号/分集/时间戳/跳转链接
.venv-asr/bin/python semantic_qa.py search "怎么把大问题拆成小问题" -k 8
.venv-asr/bin/python semantic_qa.py search "过拟合" --course BV15J411T7WQ

# 3. LLM 引用问答（需 LLM_API_KEY，同主脚本的环境变量）
.venv-asr/bin/python semantic_qa.py ask "哪几节讲了过拟合？该怎么处理？"

# 4. 索引统计
.venv-asr/bin/python semantic_qa.py stats
```

索引文件默认为 `<BILI_OUT_ROOT>/rag_index.db`；build 默认按字幕指纹增量（重转写后重跑即可，
`stats` 会报告过期课程），课程目录删除后 build 全量扫描时自动清除索引。

## 测试

```bash
.venv-asr/bin/python tests/test_core.py   # 核心回归套件（无网络，秒级；兼容 pytest）
```
每个 chunk 自带 `(bvid, p, 起止秒)` 元数据，答案引用可直接跳回原片对应时间点。

## 环境变量（可代替命令行参数）

| 变量 | 说明 |
|---|---|
| `BILIBILI_COOKIE` | B 站原始 cookie 串（如 `SESSDATA=xxx; ...`，带不带空格/URL 编码均可） |
| `BILIBILI_COOKIE_FILE` | Netscape 格式 cookie 文件路径 |
| `BILIBILI_OUTPUT` | 输出目录默认值 |
| `LLM_API_KEY` | LLM API Key（`--note` 时等效 `--api-key`） |
| `LLM_BASE_URL` | LLM 接口 Base URL（等效 `--base-url`） |
| `LLM_MODEL` | LLM 模型名（等效 `--model`，默认 `gpt-4o-mini`） |
| `OPENAI_MAX_REQUEST_BYTES` | 笔记切块的单请求字节预算（默认 200KB，小上下文模型可调小） |
| `OPENAI_RETRY_ATTEMPTS` | LLM 调用重试次数（默认 3） |
| `OPENAI_RETRY_BACKOFF_SECONDS` | 重试退避基数秒（默认 1.5，指数递增） |

复制 `.env.example` 为 `.env` 并填写即可（脚本启动时自动加载，不覆盖已有环境变量）。

## CLI 参数

```
url                          B 站视频链接（--search 搜索时可省略）
-o, --output DIR             输出目录（默认 ./bilibili_out）
-q, --quality {fast,medium,slow}  音频质量: fast=64/medium=128/slow=320 kbps mp3（默认 fast）
--cookie COOKIE             B 站原始 cookie 串
--cookie-file PATH          Netscape 格式 cookie 文件路径
--manifest PATH             额外把结果写成 JSON manifest
--transcribe {none,bcut}    无字幕分集的音频转写引擎（bcut 免费但需 B 站 cookie）
--note                      生成 LLM Markdown 笔记
--api-key KEY               LLM API Key（--note 时必填）
--base-url URL              LLM 接口 Base URL（OpenAI 兼容）
--model NAME                LLM 模型名（默认 gpt-4o-mini）
--style STYLE               笔记风格：minimal/detailed/academic/tutorial/xiaohongshu/life_journal/task_oriented/business/meeting_minutes
--format FORMATS            笔记格式，逗号分隔：toc/link/screenshot/summary
--tags TAGS                 传给 LLM 的视频标签
--extras TEXT               传给 LLM 的额外指令
--search KEYWORD            只搜索不下载
--workers N                 并行下载线程数（默认 1 串行；建议 2-4，过大易触发风控）
-v, --verbose               调试日志
```

## 模块 API

```python
from bilibili_multi_part import (
    BilibiliDownloader, NoteConfig, NoteGenerator, RequestChunker,
    search_bilibili, transcribe_bcut, BcutTranscriber, BilibiliSubtitleFetcher,
    BilibiliPartResult, TranscriptResult, TranscriptSegment,
    replace_content_markers, prepend_source_link, generate_base_prompt,
)
```

## 依赖说明

- `yt-dlp` / `requests`：基础依赖，下载、字幕直拉、搜索、BCUT 转写都靠它们
- **ffmpeg**：系统二进制依赖（非 pip 包），音频转 mp3 必需，`apt install ffmpeg` 或 `brew install ffmpeg`
- `openai`：**可选**依赖，只在 `--note` 时才 import，不加它不影响原有功能
- `torch` + `funasr`：仅 `funasr_asr_job2.py` 本地批量转写需要（装在 `.venv-asr`）

## 许可

MIT
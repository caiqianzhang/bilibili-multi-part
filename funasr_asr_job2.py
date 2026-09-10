#!/usr/bin/env python3
"""批量转写 v3: 流水线版。字幕直拉先行; 无字幕分集由多路并行下载(DL_WORKERS, 默认 2)喂单 GPU 转写。
音频临时驻留, 转完即删; 逐课程写 _全文.txt 与汇总, 断点可续。"""
import json, logging, os, sys, time, glob
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bilibili_multi_part as bmp

bmp._load_env_file()  # 与主脚本一致：读取脚本同目录 .env（BILIBILI_COOKIE 等），先于下方环境变量读取

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("asr_job")

# 分片: --shard i N 处理 cands[i::N] (双卡各跑一个进程时用)
SHARD, N_SHARDS = 0, 1
if len(sys.argv) >= 3 and sys.argv[1] == "--shard":
    SHARD, N_SHARDS = int(sys.argv[2]), int(sys.argv[3])

COOKIE = os.environ.get("BILIBILI_COOKIE", "")
# 输出根目录可配置(默认保持历史路径, 兼容旧启动方式)
ROOT = os.environ.get("BILI_OUT_ROOT", "/home/you/bilibili_python_out")
# 多进程混跑时可用 ASR_SHARD_TAG 区分汇总/临时目录(默认仍按分片号, 兼容旧启动方式)
TAG = os.environ.get("ASR_SHARD_TAG", str(SHARD))
SCRATCH = f"/tmp/asr_scratch2_g{TAG}"
DL_WORKERS = int(os.environ.get("DL_WORKERS", "2"))
# 跨分片共享的完成记录: 每完成一门课追加一个 BV 号, 重启时自动整门跳过(断点续跑)
DONE_FILE = os.path.join(ROOT, "asr_done.json")

DONE_BVIDS = {"BV1qW4y1a7fU", "BV1rpWjevEip", "BV1sHU9BmEne", "BV1tDsgzxECr", "BV1Ys4y1D72T"}
DUP_BVIDS = {"BV1aGKJ6tEsA", "BV168gh6JE1y", "BV1hY426nEm7", "BV1vkWbzSEUh"}
# 排除集合: 硬编码(历史遗留) ∪ 环境变量 ASR_EXCLUDE_BVIDS(逗号分隔), 两者取并集
EXCLUDE_BVIDS = {"BV1sTAwzTEJL"} | {
    b.strip() for b in os.environ.get("ASR_EXCLUDE_BVIDS", "").split(",") if b.strip()
}


def _load_done() -> set:
    """加载已完成课程记录(跨分片共享, 支持断点续跑)。"""
    try:
        return set(json.load(open(DONE_FILE, encoding="utf-8")))
    except (FileNotFoundError, ValueError):
        return set()


def _mark_done(bvid: str):
    """把已完成的 BV 号追加写入共享记录(原子重写, 多进程安全)。"""
    done = _load_done()
    done.add(bvid)
    with open(DONE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, ensure_ascii=False, indent=0)

# 候选课程清单来源可配置; 缺失时给出可读报错而非裸 traceback
SCAN_FILE = os.environ.get("PY_SCAN_RESULT", "/tmp/py_scan_result.json")
try:
    scan_rows = json.load(open(SCAN_FILE, encoding="utf-8"))
except FileNotFoundError:
    log.error("候选扫描结果不存在: %s（先运行课程扫描生成, 或用 PY_SCAN_RESULT 指定路径）", SCAN_FILE)
    sys.exit(1)
cands = [c for c in scan_rows
         if c["bvid"] not in DONE_BVIDS and c["bvid"] not in DUP_BVIDS
         and c["bvid"] not in EXCLUDE_BVIDS and c["bvid"] not in _load_done()]
# 手动分配: 设 ASR_ONLY_BVIDS=BV1xx,BV2yy 时只处理名单内课程(多进程拆分用)。
# 指定名单时跳过分片切片(否则名单被 [SHARD::N_SHARDS] 二次切掉, 曾导致空队列)
_only = os.environ.get("ASR_ONLY_BVIDS", "")
if _only:
    _want = {b.strip() for b in _only.split(",") if b.strip()}
    cands = [c for c in cands if c["bvid"] in _want]
    cands.sort(key=lambda c: -c["play"])
    log.info("手动指定课程名单: %d 门, 共 %d 集", len(cands), sum(c["parts"] for c in cands))
else:
    cands.sort(key=lambda c: -c["play"])
    cands = cands[SHARD::N_SHARDS]
    log.info("分片 %d/%d: 待处理课程 %d 门, 共 %d 集 (按播放数降序; 续跑跳过 %d 门)",
             SHARD + 1, N_SHARDS, len(cands), sum(c["parts"] for c in cands), len(_load_done()))

if not cands:
    log.info("没有待处理课程, 直接退出（不加载模型）")
    sys.exit(0)

os.makedirs(ROOT, exist_ok=True); os.makedirs(SCRATCH, exist_ok=True)
import torch
from funasr import AutoModel
device = "cuda:0" if torch.cuda.is_available() else "cpu"
log.info("设备: %s", device)
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
                  device=device, disable_update=True, log_level="ERROR")
fetcher = bmp.BilibiliSubtitleFetcher(cookie=COOKIE)
dl = bmp.BilibiliDownloader(cookie=COOKIE)


# SRT / 文本工具已抽到 bilibili_multi_part 模块(单一真源), 此处仅保留兼容别名,
# 避免 v1/v2 各写一份导致漂移
fmt_ts = bmp.fmt_ts
to_srt = bmp.write_srt
parse_srt_text = bmp.parse_srt_text


def fetch_audio(bvid, p):
    """线程池工作函数: 只下载音频, 不碰模型。"""
    try:
        audio = dl.download(f"https://www.bilibili.com/video/{bvid}?p={p}",
                            output_dir=SCRATCH, quality="fast")
        return (p, audio.file_path, None)
    except Exception as e:
        return (p, None, str(e))


# 汇总按 bvid 去重、跨运行保留旧行: 每完成一门课整文件原子重写, 中断不丢历史
SUMMARY_FILE = os.path.join(ROOT, f"asr_v2_summary_gpu{TAG}.json")


def _load_summary_map() -> dict:
    try:
        rows = json.load(open(SUMMARY_FILE, encoding="utf-8"))
        return {r.get("bvid"): r for r in rows if isinstance(r, dict) and r.get("bvid")}
    except (FileNotFoundError, ValueError):
        return {}


summary_map = _load_summary_map()
for idx, c in enumerate(cands, 1):
    bvid = c["bvid"]
    out_dir = os.path.join(ROOT, bvid); os.makedirs(out_dir, exist_ok=True)
    # 断点续跑: 已有 _全文.txt 就整门跳过(其它分片可能已交付)
    if os.path.exists(os.path.join(out_dir, f"{bvid}_全文.txt")):
        log.info("[SKIP] %s 已完成(_全文.txt 存在), 跳过", bvid)
        summary_map[bvid] = {**c, "status": "done", "subtitle_parts": -1, "asr_parts": -1,
                             "failed_parts": -1, "elapsed_s": 0}
        continue
    meta = bmp.fetch_view_data(bvid, cookie=COOKIE)
    pages = (meta or {}).get("pages") or []
    if not pages:
        log.warning("[SKIP] %s view API 无分P", bvid)
        summary_map[bvid] = {**c, "status": "no_pages"}; continue
    titles = {i + 1: ((pg.get("part") or f"P{i+1}").strip()) for i, pg in enumerate(pages)}
    log.info("===== [%d/%d] %s | %s播放 %d集 =====", idx, len(cands), bvid, f"{c['play']:,}", len(pages))

    n_sub = n_asr = n_fail = 0
    texts = {}  # p -> 文本行列表 / None=失败
    t0 = time.time()

    # 阶段1: 字幕直拉(并行) + 断点续跑(跳过已有 .srt 的分集)
    asr_todo = []

    def _fetch_one(bvid, p, cid):
        try:
            return (p, fetcher.fetch_subtitles_for_cid(bvid, cid), None)
        except Exception as e:
            return (p, None, str(e)[:70])

    with ThreadPoolExecutor(max_workers=max(1, DL_WORKERS)) as pool:
        futures = {}
        for p, page in enumerate(pages, 1):
            existing = sorted(glob.glob(os.path.join(out_dir, f"{bvid}_p{p}.*.srt")), key=bmp.srt_lang_rank)
            if existing:
                n_sub += 1
                texts[p] = bmp.parse_srt_text(existing[0])
                continue
            futures[pool.submit(_fetch_one, bvid, p, page.get("cid"))] = p
        for fut in as_completed(futures):
            p, tr, err = fut.result()
            if err:
                log.warning("p%d 字幕接口异常: %s", p, err); tr = None
            if tr:
                n_sub += 1
                dl._write_subtitle_file(bvid, p, tr, out_dir)
                texts[p] = [s.text for s in tr.segments]
            else:
                asr_todo.append(p)
    log.info("字幕直拉 %d 集(含续跑跳过), 待本地转写 %d 集", n_sub, len(asr_todo))

    # 阶段2: 3 路并行下载 + 主线程 GPU 转写(流水线)
    done = 0
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as pool:
        futures = {pool.submit(fetch_audio, bvid, p): p for p in asr_todo}
        for fut in as_completed(futures):
            p, path, err = fut.result()
            if err:
                n_fail += 1; texts[p] = None
                log.warning("p%d 音频下载失败: %s | %s", p, err[:50], titles[p][:24])
                continue
            try:
                res = model.generate(input=path, batch_size_s=300, sentence_timestamp=True)[0]
                info = res.get("sentence_info") or []
                segs = [{"start": s["start"] / 1000.0, "end": s["end"] / 1000.0,
                         "text": (s["text"] or "").strip()}
                        for s in info if (s.get("text") or "").strip()]
                if not segs and (res.get("text") or "").strip():
                    segs = [{"start": 0.0, "end": 0.0, "text": res["text"]}]
                if not segs:
                    raise RuntimeError("转写结果为空")
                bmp.write_srt(segs, os.path.join(out_dir, f"{bvid}_p{p}.asr_zh.srt"))
                texts[p] = [s["text"] for s in segs]
                n_asr += 1
            except Exception as e:
                n_fail += 1; texts[p] = None
                log.warning("p%d 转写失败: %s | %s", p, str(e)[:50], titles[p][:24])
            finally:
                for leftover in glob.glob(os.path.join(SCRATCH, f"{bvid}_p{p}.*")):
                    try: os.remove(leftover)
                    except OSError: pass
            done += 1
            if done % 20 == 0 or done == len(asr_todo):
                log.info("转写进度 %d/%d (字幕%d 转写%d 失败%d) 用时%ds",
                         done, len(asr_todo), n_sub, n_asr, n_fail, round(time.time() - t0))

    txt_path = os.path.join(out_dir, f"{bvid}_全文.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in range(1, len(pages) + 1):
            f.write(f"\n\n===== P{p} {titles[p]} =====\n")
            f.write("\n".join(texts[p]) if texts.get(p) else "(无文字: 无字幕且转写失败)")
    summary_map[bvid] = {**c, "subtitle_parts": n_sub, "asr_parts": n_asr, "failed_parts": n_fail,
                         "elapsed_s": round(time.time() - t0), "status": "ok"}
    _mark_done(bvid)   # 持久化完成记录, 重启时整门跳过(断点续跑)
    log.info("[OK] %s: %d集 = 字幕%d + 转写%d + 失败%d, 用时%ds", bvid, len(pages), n_sub, n_asr, n_fail, round(time.time() - t0))
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(list(summary_map.values()), f, ensure_ascii=False, indent=2)
    time.sleep(8)

log.info("===== 全部完成: %d/%d 门课程成功 =====",
         sum(1 for s in summary_map.values() if s.get("status") == "ok"), len(cands))

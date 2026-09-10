#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字幕语料语义问答（RAG）：为 bilibili_python_out 下的 .srt 建向量索引，语义检索 + LLM 引用问答。

数据规模（2026-09）：48 门课 / 5691 个 srt / 纯文本约 2700 万字 → 约 6 万 chunk，
bge-m3(1024 维) + SQLite 单文件 + numpy 暴力检索即可毫秒级返回，无需向量数据库服务。

每个 chunk 自带 (bvid, p, 起止秒) 元数据，检索结果可直接生成 ?p=N&t=秒 跳转链接。

用法（用 .venv-asr，复用 torch/transformers）：
    # 建索引：首次自动经 hf-mirror 下载 bge-m3（约 2.3GB）；默认全量，可只建指定课程
    HF_ENDPOINT=https://hf-mirror.com .venv-asr/bin/python semantic_qa.py build
    .venv-asr/bin/python semantic_qa.py build --only-bvid BV1h6m8BWE1T
    # 语义检索（不需要 LLM key）
    .venv-asr/bin/python semantic_qa.py search "过拟合怎么办" -k 8
    # LLM 引用问答（需 LLM_API_KEY，读 .env/环境变量，OpenAI 兼容）
    .venv-asr/bin/python semantic_qa.py ask "哪几节讲了过拟合？该怎么处理？"
    .venv-asr/bin/python semantic_qa.py stats
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np
import requests

# HF 访问环境：默认走 hf-mirror（huggingface.co 直连不稳），禁用 Xet 后端（mirror 不代理其 CAS 会 401）。
# 必须在 transformers/huggingface_hub 首次导入前设置，故放在模块顶部。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 语料布局约定/跳转链接格式收编自 bilibili_multi_part（单一真源）
from bilibili_multi_part import bilibili_jump_url as jump_url, pick_course_srts

MODEL_NAME = "BAAI/bge-m3"
EMB_DIM = 1024
# bge-m3 支持 8192；中文在 XLM-R 分词下约 1.3 token/字，450 字块给 768 防截断
MAX_LEN = 768
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
VIEW_API = "https://api.bilibili.com/x/web-interface/view"
TITLE_RE = re.compile(r"^===== P(\d+) (.*) =====$")
TS_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def log(msg, *args):
    print(f"{datetime.now():%H:%M:%S} {msg % args if args else msg}", flush=True)


def default_root() -> str:
    return os.environ.get("BILI_OUT_ROOT", "/home/you/bilibili_python_out")


def default_db(root: str) -> str:
    return os.path.join(root, "rag_index.db")


# ---------------------------------------------------------------------------
# 语料解析：srt → (start, end, text) 分段 → 按字数聚合成带时间边界的 chunk
# ---------------------------------------------------------------------------
def parse_srt(path: str):
    """解析 srt 为 [(start_sec, end_sec, text)]；跳过无法解析的行。"""
    segs = []
    cur_ts, cur_text = None, []

    def flush():
        nonlocal cur_ts, cur_text
        if cur_ts and cur_text:
            text = " ".join(x.strip() for x in cur_text if x.strip())
            if text:
                segs.append((cur_ts[0], cur_ts[1], text))
        cur_ts, cur_text = None, []

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = TS_RE.search(line)
            if m:
                flush()
                g = [int(x) for x in m.groups()]
                cur_ts = (g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0,
                          g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0)
            elif line.strip() and not line.strip().isdigit():
                if cur_ts is not None:
                    cur_text.append(line.strip())
    flush()
    return segs


def load_part_titles(course_dir: str, bvid: str) -> dict:
    """从历史产物 _全文.txt 的 `===== P{n} 标题 =====` 头恢复分集标题。"""
    titles = {}
    txt = os.path.join(course_dir, f"{bvid}_全文.txt")
    if os.path.exists(txt):
        with open(txt, encoding="utf-8") as f:
            for line in f:
                m = TITLE_RE.match(line.strip())
                if m:
                    titles[int(m.group(1))] = m.group(2).strip()
    return titles


def make_chunks(segs, chunk_chars: int):
    """相邻分段按字数聚合成 chunk，记录起止秒：[(text, start, end, n_chars)]。"""
    chunks, buf, start, n = [], [], None, 0
    for s, e, text in segs:
        if not buf:
            start = s
        buf.append(text)
        n += len(text)
        if n >= chunk_chars:
            chunks.append((" ".join(buf), start, e, n))
            buf, n = [], 0
    if buf:
        chunks.append((" ".join(buf), start, segs[-1][1], n))
    return chunks


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def fetch_course_title(bvid: str, cookie: str = "") -> str:
    try:
        headers = {"User-Agent": UA, "Referer": "https://www.bilibili.com"}
        if cookie:
            headers["Cookie"] = cookie
        data = requests.get(VIEW_API, params={"bvid": bvid}, headers=headers, timeout=10).json()
        if data.get("code") == 0:
            return (data.get("data", {}).get("title") or "").strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# 向量化：bge-m3 CLS 池化 + L2 归一化（点积即余弦相似度）
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log("加载 %s（首次自动经 HF_ENDPOINT 下载约 2.3GB）... 设备: %s", MODEL_NAME, self.device)
        try:
            self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME)
        except Exception as exc:
            # 已有本地缓存时，仓库元数据联网校验失败不应阻断（离线兜底）
            log("在线加载失败(%s)，改用本地缓存离线加载", type(exc).__name__)
            self.tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
            self.model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
        self.model = self.model.to(self.device).eval()
        self.batch_size = batch_size

    @staticmethod
    def _cls_pool(hidden):
        return hidden[:, 0]

    def encode(self, texts) -> np.ndarray:
        """→ (n, 1024) float32 L2 归一化矩阵。"""
        torch = self.torch
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # 长文本先行, 减少padding
        out = np.zeros((len(texts), EMB_DIM), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                batch = [texts[j] for j in idx]
                enc = self.tok(batch, padding=True, truncation=True, max_length=MAX_LEN,
                               return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                vec = self._cls_pool(hidden)
                vec = torch.nn.functional.normalize(vec, p=2, dim=1).cpu().numpy()
                for j, v in zip(idx, vec):
                    out[j] = v
        return out


# ---------------------------------------------------------------------------
# 索引库：SQLite 单文件（chunks + embeddings fp16 BLOB + 课程/元数据表）
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  bvid TEXT NOT NULL, p INTEGER NOT NULL, part_title TEXT,
  start REAL, end REAL, n_chars INTEGER, text TEXT,
  embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS idx_chunks_course ON chunks(bvid, p, start);
CREATE TABLE IF NOT EXISTS courses(
  bvid TEXT PRIMARY KEY, title TEXT, n_parts INTEGER, n_chunks INTEGER,
  n_chars INTEGER, indexed_at TEXT, fingerprint TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    # WAL: 建索引的长事务期间允许并发读（默认 delete 模式下读会撞 database is locked）
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def course_fingerprint(srts: dict) -> str:
    """课程字幕集合指纹（文件数 + 最新 mtime）：变化才重嵌，否则秒级跳过。"""
    if not srts:
        return "0:0"
    mt = max(os.path.getmtime(p) for p in srts.values())
    return f"{len(srts)}:{int(mt)}"


def ensure_schema(con: sqlite3.Connection, root: str) -> None:
    """旧库升级：补 fingerprint 列并按当前磁盘状态回填，避免升级后触发全量重嵌。"""
    cols = {r[1] for r in con.execute("PRAGMA table_info(courses)")}
    if "fingerprint" in cols:
        return
    con.execute("ALTER TABLE courses ADD COLUMN fingerprint TEXT")
    for (bvid,) in con.execute("SELECT bvid FROM courses").fetchall():
        cdir = os.path.join(root, bvid)
        if os.path.isdir(cdir):
            con.execute("UPDATE courses SET fingerprint=? WHERE bvid=?",
                        (course_fingerprint(pick_course_srts(cdir)), bvid))
    con.commit()
    log("已升级索引结构: 补充 fingerprint 列并回填 %d 门课", con.total_changes)


def build(args):
    root = args.root or default_root()
    db_path = args.db or default_db(root)
    only = {b.strip() for b in (args.only_bvid or "").split(",") if b.strip()}
    cookie = os.environ.get("BILIBILI_COOKIE", "")

    course_dirs = sorted(d for d in glob.glob(os.path.join(root, "BV*")) if os.path.isdir(d))
    if only:
        course_dirs = [d for d in course_dirs if os.path.basename(d) in only]
    if not course_dirs:
        log("在 %s 下没有找到课程目录", root)
        return 1

    con = connect(db_path)
    ensure_schema(con, root)

    # 指纹比对：只重嵌字幕有变化的课程（--force 全量）
    existing_fp = dict(con.execute("SELECT bvid, fingerprint FROM courses"))
    jobs, skipped = [], 0
    for cdir in course_dirs:
        bvid = os.path.basename(cdir)
        fp = course_fingerprint(pick_course_srts(cdir))
        if not args.force and fp != "0:0" and existing_fp.get(bvid) == fp:
            skipped += 1
            continue
        jobs.append(cdir)
    if not only:  # 全量扫描时顺带 GC：磁盘上已消失的课程清除索引（幽灵课程）
        removed = 0
        for (bvid,) in con.execute("SELECT bvid FROM courses").fetchall():
            if not os.path.isdir(os.path.join(root, bvid)):
                con.execute("DELETE FROM chunks WHERE bvid=?", (bvid,))
                con.execute("DELETE FROM courses WHERE bvid=?", (bvid,))
                removed += 1
                log("[GC] %s 课程目录已不存在，清除索引", bvid)
        if removed:
            con.commit()
    log("待建 %d 门，未变化跳过 %d 门", len(jobs), skipped)
    if not jobs:
        log("索引已是最新，无需重建")
        return 0

    emb = Embedder(batch_size=args.batch)
    total_chunks = 0
    t0 = time.time()

    for ci, cdir in enumerate(jobs, 1):
        bvid = os.path.basename(cdir)
        srts = pick_course_srts(cdir)
        if not srts:
            log("[%d/%d] %s 无可用 srt，跳过", ci, len(jobs), bvid)
            continue
        fp = course_fingerprint(srts)
        titles = load_part_titles(cdir, bvid)

        texts, meta_rows = [], []
        for p in sorted(srts):
            segs = parse_srt(srts[p])
            for text, s, e, n in make_chunks(segs, args.chunk_chars):
                meta_rows.append((bvid, p, titles.get(p) or f"P{p}", s, e, n))
                texts.append(text)
        if not texts:
            log("[%d/%d] %s 全部字幕为空，跳过", ci, len(jobs), bvid)
            continue

        vectors = emb.encode(texts)
        con.execute("DELETE FROM chunks WHERE bvid=?", (bvid,))
        con.executemany(
            "INSERT INTO chunks(bvid,p,part_title,start,end,n_chars,text,embedding) "
            "VALUES(?,?,?,?,?,?,?,?)",
            [(bvid, mp, mt, ms, me, mn, mt_text, vec.astype(np.float16).tobytes())
             for (bvid, mp, mt, ms, me, mn), mt_text, vec
             in zip(meta_rows, texts, vectors)])

        course_title = "" if args.offline else fetch_course_title(bvid, cookie)
        con.execute("INSERT OR REPLACE INTO courses VALUES(?,?,?,?,?,?,?)",
                    (bvid, course_title, len(srts), len(texts),
                     sum(r[5] for r in meta_rows),
                     datetime.now().isoformat(timespec="seconds"), fp))
        con.commit()
        total_chunks += len(texts)
        log("[%d/%d] %s: %d 集 → %d chunk (%.0f 万字) 累计耗时 %ds",
            ci, len(jobs), bvid, len(srts), len(texts),
            sum(r[5] for r in meta_rows) / 1e4, time.time() - t0)

    con.execute("INSERT OR REPLACE INTO meta VALUES('model',?)", (MODEL_NAME,))
    con.execute("INSERT OR REPLACE INTO meta VALUES('chunk_chars',?)", (str(args.chunk_chars),))
    con.execute("INSERT OR REPLACE INTO meta VALUES('built_at',?)", (datetime.now().isoformat(timespec="seconds"),))
    con.commit()
    log("建索引完成: %d 课程, %d chunk, 耗时 %.0fs → %s",
        len(jobs), total_chunks, time.time() - t0, db_path)
    return 0


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
def load_matrix(con) -> tuple:
    rows = con.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.zeros((len(rows), EMB_DIM), dtype=np.float32)
    for i, (_, blob) in enumerate(rows):
        mat[i] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    return ids, mat


def semantic_search(args) -> int:
    root = args.root or default_root()
    db_path = args.db or default_db(root)
    if not os.path.exists(db_path):
        log("索引不存在: %s（先运行 build）", db_path)
        return 1
    con = connect(db_path)
    t0 = time.time()
    ids, mat = load_matrix(con)
    emb = Embedder(batch_size=8)
    q = emb.encode([args.query])[0]
    scores = mat @ q
    if args.course:
        id2key = dict(con.execute("SELECT id, bvid FROM chunks").fetchall())
        mask = np.array([args.course.lower() in id2key[i].lower() for i in ids])
        scores = np.where(mask, scores, -1.0)
    top = np.argsort(-scores)[:args.k]
    dt = (time.time() - t0) * 1000

    print(f"\n===== 「{args.query}」 语义检索 top-{args.k}（{dt:.0f}ms, 语料 {len(ids)} chunks）=====")
    shown = 0
    for rank, i in enumerate(top, 1):
        if scores[i] < 0:
            break
        shown = rank
        bvid, p, part, s, text = con.execute(
            "SELECT bvid,p,part_title,start,text FROM chunks WHERE id=?", (int(ids[i]),)).fetchone()
        course = con.execute("SELECT title FROM courses WHERE bvid=?", (bvid,)).fetchone()
        cname = (course[0] if course and course[0] else bvid)[:24]
        snippet = text[:110] + ("…" if len(text) > 110 else "")
        print(f"\n[{rank}] 相似度 {scores[i]:.3f} | {cname} | P{p} {part} @ {fmt_ts(s)}")
        print(f"    {snippet}")
        print(f"    → {jump_url(bvid, p, s)}")
    if not shown:
        log("无命中结果")
        return 1
    return 0


# ---------------------------------------------------------------------------
# LLM 引用问答
# ---------------------------------------------------------------------------
ASK_PROMPT = """你是 B 站课程字幕知识库的问答助手。仅依据下面检索到的字幕片段回答问题。

要求：
- 用中文回答，先给结论再展开；在关键结论后标注引用编号，如 [1][2]
- 引用编号只能对应提供的片段，不得编造；片段不足以回答时明确说明"语料中未找到"
- 保留必要的英文术语

问题：{question}

检索到的片段：
{context}
"""


def retrieve(con, query: str, k: int, course: str = ""):
    ids, mat = load_matrix(con)
    emb = Embedder(batch_size=8)
    q = emb.encode([query])[0]
    scores = mat @ q
    if course:
        id2key = dict(con.execute("SELECT id, bvid FROM chunks").fetchall())
        mask = np.array([course.lower() in id2key[i].lower() for i in ids])
        scores = np.where(mask, scores, -1.0)
    top = np.argsort(-scores)[:k]
    hits = []
    for i in top:
        if scores[i] < 0:
            break
        row = con.execute(
            "SELECT c.bvid,c.p,c.part_title,c.start,c.text,co.title FROM chunks c "
            "LEFT JOIN courses co ON co.bvid=c.bvid WHERE c.id=?", (int(ids[i]),)).fetchone()
        hits.append({"score": float(scores[i]), "bvid": row[0], "p": row[1],
                     "part": row[2], "start": row[3], "text": row[4], "course": row[5] or row[0]})
    return hits


def ask(args) -> int:
    root = args.root or default_root()
    db_path = args.db or default_db(root)
    if not os.path.exists(db_path):
        log("索引不存在: %s（先运行 build）", db_path)
        return 1
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        log("未配置 LLM_API_KEY（.env 或环境变量）。只检索不回答可改用: search %r", args.question)
        return 1
    try:
        from openai import OpenAI
    except ImportError:
        log("缺少 openai 包: .venv-asr/bin/pip install openai")
        return 1

    con = connect(db_path)
    t0 = time.time()
    hits = retrieve(con, args.question, args.k, args.course)
    if not hits:
        log("检索无结果")
        return 1
    log("检索 %d 条命中（%ds），调用 LLM %s ...", len(hits), time.time() - t0, args.model)

    context = "\n\n".join(
        f"[{i}] 课程《{h['course'][:40]}》P{h['p']} {h['part']} @ {fmt_ts(h['start'])}\n{h['text']}"
        for i, h in enumerate(hits, 1))
    client = OpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL"))
    messages = [{"role": "user", "content": ASK_PROMPT.format(question=args.question, context=context)}]
    try:
        resp = client.chat.completions.create(model=args.model, messages=messages, temperature=0.3)
    except Exception as exc:
        raw = str(exc).lower()
        if "temperature" in raw and any(t in raw for t in
                                        ("does not support", "unsupported", "only the default")):
            # 部分（推理类）模型不支持自定义 temperature，与主脚本 NoteGenerator 同款兜底
            log("模型 %s 不支持自定义 temperature，改用默认值重试", args.model)
            try:
                resp = client.chat.completions.create(model=args.model, messages=messages)
            except Exception as exc2:
                log("LLM 调用失败: %s", exc2)
                return 1
        else:
            log("LLM 调用失败: %s", exc)
            return 1
    answer = (resp.choices[0].message.content or "").strip()

    print(f"\n===== 问：{args.question} =====\n")
    print(answer)
    print(f"\n===== 来源（模型: {args.model}）=====")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['course'][:30]} | P{h['p']} {h['part']} @ {fmt_ts(h['start'])} "
              f"(相似度 {h['score']:.3f})")
        print(f"    {jump_url(h['bvid'], h['p'], h['start'])}")
    return 0


def stats(args) -> int:
    root = args.root or default_root()
    db_path = args.db or default_db(root)
    if not os.path.exists(db_path):
        log("索引不存在: %s（先运行 build）", db_path)
        return 1
    con = connect(db_path)
    ensure_schema(con, root)
    n_chunk, n_chars = con.execute("SELECT COUNT(*), COALESCE(SUM(n_chars),0) FROM chunks").fetchone()
    print(f"索引: {db_path} ({os.path.getsize(db_path)/1e6:.0f} MB)")
    for k, v in con.execute("SELECT key, value FROM meta"):
        print(f"  {k}: {v}")
    print(f"chunk 总数: {n_chunk:,} | 覆盖文本: {n_chars/1e4:.0f} 万字 | 课程: "
          f"{con.execute('SELECT COUNT(*) FROM courses').fetchone()[0]}")
    # 索引新鲜度：磁盘字幕与指纹不一致 = 过期；目录消失 = 幽灵课程
    stale, missing = [], []
    for bvid, fp in con.execute("SELECT bvid, fingerprint FROM courses").fetchall():
        cdir = os.path.join(root, bvid)
        if not os.path.isdir(cdir):
            missing.append(bvid)
        elif fp and course_fingerprint(pick_course_srts(cdir)) != fp:
            stale.append(bvid)
    if stale:
        print(f"⚠ {len(stale)} 门课字幕已变化（索引过期）: {', '.join(stale)}")
        print(f"  建议: build --only-bvid {','.join(stale)}")
    if missing:
        print(f"⚠ {len(missing)} 门课目录已消失（build 时会自动清除）: {', '.join(missing)}")
    for bvid, title, np_, nc, nch in con.execute(
            "SELECT bvid,title,n_parts,n_chunks,n_chars FROM courses ORDER BY n_chars DESC"):
        print(f"  {bvid} | {(title or '（未取到标题）')[:36]:<38} | {np_} 集 | {nc} chunk | {nch/1e4:.0f} 万字")
    return 0


def main(argv=None):
    # 与主脚本一致：先加载脚本同目录 .env（ask 模式要读 LLM_API_KEY 等）
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bilibili_multi_part import _load_env_file
        _load_env_file()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="字幕语料语义问答（bge-m3 + SQLite + numpy）")
    parser.add_argument("--root", default=None, help="语料根目录（默认 $BILI_OUT_ROOT）")
    parser.add_argument("--db", default=None, help="索引 db 路径（默认 <root>/rag_index.db）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        # 公共参数在子命令后也可用（SUPPRESS: 未提供时不覆盖顶层解析值）
        p.add_argument("--root", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--db", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    b = sub.add_parser("build", help="扫描语料并建向量索引（默认增量：只建有变化的课程）")
    add_common(b)
    b.add_argument("--only-bvid", default="", help="只建指定课程，逗号分隔")
    b.add_argument("--force", action="store_true", help="忽略指纹全量重嵌")
    b.add_argument("--chunk-chars", type=int, default=450, help="chunk 目标字数（默认 450）")
    b.add_argument("--batch", type=int, default=64, help="编码批大小")
    b.add_argument("--offline", action="store_true", help="不调 view API 取课程标题")

    s = sub.add_parser("search", help="语义检索（无需 LLM）")
    add_common(s)
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--course", default="", help="限定课程（bvid 子串）")

    a = sub.add_parser("ask", help="LLM 引用问答（需 LLM_API_KEY）")
    add_common(a)
    a.add_argument("question")
    a.add_argument("-k", type=int, default=6)
    a.add_argument("--course", default="", help="限定课程（bvid 子串）")
    a.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))

    st = sub.add_parser("stats", help="索引统计")
    add_common(st)

    args = parser.parse_args(argv)
    if args.cmd == "build":
        return build(args)
    if args.cmd == "search":
        return semantic_search(args)
    if args.cmd == "ask":
        return ask(args)
    return stats(args)


if __name__ == "__main__":
    sys.exit(main())

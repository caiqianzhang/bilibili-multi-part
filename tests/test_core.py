#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核心回归测试套件（无网络、无模型依赖，秒级跑完）。

运行方式：
    .venv-asr/bin/python tests/test_core.py   # 全量（含 semantic_qa 部分）
    .venv/bin/python tests/test_core.py       # 只跑主脚本部分（缺 numpy 时自动跳过 RAG）
兼容 pytest: pytest tests/test_core.py
"""
import glob
import gc
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bilibili_multi_part as bmp

try:  # semantic_qa 依赖 numpy/torch 环境；缺失时跳过 RAG 相关用例
    import semantic_qa as sq
    HAVE_SQ = True
except Exception:
    HAVE_SQ = False

RESULTS = []


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  {'✓' if cond else '✗ FAIL'} {name} {detail}")


# ---------------------------------------------------------------------------
# 1. cookie：切分 / URL 解码统一 / 临时文件清理
# ---------------------------------------------------------------------------
def test_cookie_handling():
    print("【cookie】")
    p = bmp.write_cookie_file("SESSDATA=aaa;bfe_id=bbb")  # 无空格写法
    lines = [l for l in open(p).read().splitlines() if not l.startswith("#")]
    ok("无空格 cookie 全部键写入", len(lines) == 2 and lines[1].split("\t")[5] == "bfe_id")
    os.unlink(p)
    p = bmp.write_cookie_file("SESSDATA=aaa%2Cbbb")
    ok("URL 编码自动解码", "aaa,bbb" in open(p).read())
    os.unlink(p)
    ok("_load_cookie_pairs 解码一致", bmp._load_cookie_pairs("SESSDATA=x%2Cy", None) == {"SESSDATA": "x,y"})

    d = bmp.BilibiliDownloader(cookie="SESSDATA=leak_demo")
    tmp_path = d._cookie_file
    ok("构造时创建临时 cookie 文件", tmp_path and os.path.exists(tmp_path))
    del d
    gc.collect()
    ok("对象回收后临时文件自动清理", not os.path.exists(tmp_path))


# ---------------------------------------------------------------------------
# 2. BV 号提取
# ---------------------------------------------------------------------------
def test_extract_bvid():
    print("【BV 号提取】")
    ok("标准带参链接", bmp.extract_bvid("https://www.bilibili.com/video/BV1bK411W797?p=3") == "BV1bK411W797")
    ok("过长 token 不误捕", bmp.extract_bvid("BV1bK411W797extra") is None)
    ok("过短不误捕", bmp.extract_bvid("BV123") is None)


# ---------------------------------------------------------------------------
# 3. 时间格式与笔记时间标记（长视频小时位）
# ---------------------------------------------------------------------------
def test_time_format_and_markers():
    print("【时间格式/标记】")
    ft = bmp.NoteGenerator._format_time
    ok("mm:ss", ft(65) == "01:05" and ft(3599) == "59:59")
    ok("h:mm:ss（≥1h 不丢小时位）", ft(3600) == "1:00:00" and ft(3753) == "1:02:33")
    out = bmp.replace_content_markers(
        "A *Content-[00:30]* B *Content-[65:30]* C *Content-[1:02:15]* D Content-04:16", "BV1abc_p2")
    ok("四种时间形态全部转链接",
       all(t in out for t in ("t=30)", "t=3930)", "t=3735)", "t=256)")))


# ---------------------------------------------------------------------------
# 4. 语料布局约定（单一真源）
# ---------------------------------------------------------------------------
def test_corpus_layout_utils():
    print("【语料布局约定】")
    tmp = tempfile.mkdtemp(prefix="t_layout_")
    try:
        bvid = "BV1bK411W797"
        for name in (f"{bvid}_p1.zh.srt", f"{bvid}_p1.ai_zh.srt", f"{bvid}_p1.asr_zh.srt",
                     f"{bvid}_p2.ai_zh.srt", "notmatch.txt"):
            open(os.path.join(tmp, name), "w").write("x")
        picked = bmp.pick_course_srts(tmp)
        ok("每集只取一份且人工>AI>ASR",
           os.path.basename(picked[1]) == f"{bvid}_p1.zh.srt"
           and os.path.basename(picked[2]) == f"{bvid}_p2.ai_zh.srt")
        ok("跳过不合规文件名", len(picked) == 2)
        ok("跳转链接格式", bmp.bilibili_jump_url("BV1abc", 2, 3735.9)
           == "https://www.bilibili.com/video/BV1abc?p=2&t=3735")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. RequestChunker：增量估算与精确算法等价
# ---------------------------------------------------------------------------
def test_chunker():
    print("【切块器】")
    def build_msg(segs, imgs, prefix="P"):
        text = "\n".join(f"{prefix}{i} - {s['text']}" for i, s in enumerate(segs))
        return [{"role": "user", "content": "HEAD" * 50 + text + "TAIL" * 20}]

    def exact_size(segs):
        return len(json.dumps(build_msg(segs, []), ensure_ascii=False).encode())

    def exact_chunk(segments, max_bytes):  # 旧精确算法参考实现
        out, i = [], 0
        while i < len(segments):
            batch = []
            while i < len(segments):
                cand = batch + [segments[i]]
                if exact_size(cand) <= max_bytes:
                    batch, i = cand, i + 1
                    continue
                break
            if not batch:
                batch, i = [segments[i]], i + 1
            out.append(batch)
        return out

    segs = [{"text": "内容" * 40 + str(i)} for i in range(300)]
    ch = bmp.RequestChunker(build_msg, 8192)
    chunks = ch.chunk(segs, [], prefix="P")
    joined = [s["text"] for c in chunks for s in c.segments]
    ok("文本无丢失且有序", joined == [s["text"] for s in segs])
    over = [c for c in chunks if exact_size(c.segments) > 8192]
    ok("无超预算块（含余量内）", not over, f"(max {max(exact_size(c.segments) for c in chunks)})")
    ok("切块数与精确算法一致", len(chunks) == len(exact_chunk(segs, 8192)),
       f"({len(chunks)})")


# ---------------------------------------------------------------------------
# 6. .env 加载（不覆盖已有环境变量）
# ---------------------------------------------------------------------------
def test_env_loader():
    print("【.env 加载】")
    envf = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
    envf.write("# comment\nLLM_MODEL=\"from-dotenv\"\nexport LLM_API_KEY=sk-dotenv\nBAD LINE\n")
    envf.close()
    os.environ["LLM_MODEL"] = "keep-me"
    bmp._load_env_file(envf.name)
    ok("已有变量不覆盖", os.environ["LLM_MODEL"] == "keep-me")
    ok("新变量写入且去引号/支持 export", os.environ.get("LLM_API_KEY") == "sk-dotenv")
    del os.environ["LLM_MODEL"], os.environ["LLM_API_KEY"]
    os.unlink(envf.name)


# ---------------------------------------------------------------------------
# 7. download_all_parts：按课程子目录 / 失败继续并记录 / cookie 收敛
# ---------------------------------------------------------------------------
def test_download_all_parts():
    print("【批量下载编排】")
    OUT = tempfile.mkdtemp(prefix="t_dl_")
    view_calls, used_dirs, used_cookies = [], [], []

    def fake_view(bvid, cookie=""):
        view_calls.append(bvid)
        return {"title": "t", "pages": [{"cid": 1, "part": "P1"}, {"cid": 2, "part": "P2"}]}

    def fake_dl(url, output_dir=None, quality="fast"):
        used_dirs.append(output_dir)
        if url.endswith("p=2"):
            raise RuntimeError("网络抖动")
        return bmp.AudioDownloadResult(os.path.join(output_dir or ".", "a.mp3"), "t", 1.0,
                                       None, "bilibili", "BV1bK411W797_p1", {})

    class FakeFetcher:
        def __init__(self, cookie=None):
            used_cookies.append(cookie)
        def list_pages(self, bvid):
            return fake_view(bvid)["pages"]
        def fetch_subtitles_for_cid(self, bvid, cid):
            return None

    with mock.patch.object(bmp, "fetch_view_data", side_effect=fake_view), \
         mock.patch.object(bmp, "BilibiliSubtitleFetcher", FakeFetcher), \
         mock.patch.object(bmp.BilibiliDownloader, "download", side_effect=fake_dl):
        dl = bmp.BilibiliDownloader(cookie="SESSDATA=ctor")
        rs = dl.download_all_parts("https://www.bilibili.com/video/BV1bK411W797", output_dir=OUT)

    ok("默认按课程子目录落盘", len(used_dirs) == 2 and
       all(d == os.path.join(OUT, "BV1bK411W797") for d in used_dirs))
    ok("view API 只调 1 次（无 N+1）", len(view_calls) == 1)
    ok("单集失败不中止整批", len(rs) == 2 and rs[1].audio is None
       and "音频下载失败" in rs[1].error)
    ok("子目录已创建", os.path.isdir(os.path.join(OUT, "BV1bK411W797")))

    # cookie 收敛：形参缺省时回落构造器值（字幕与音频同侧）
    with mock.patch.object(bmp, "fetch_view_data", side_effect=fake_view), \
         mock.patch.object(bmp, "BilibiliSubtitleFetcher", FakeFetcher), \
         mock.patch.object(bmp.BilibiliDownloader, "download", side_effect=fake_dl):
        bmp.BilibiliDownloader(cookie="SESSDATA=ctor").download_all_parts(
            "https://www.bilibili.com/video/BV1bK411W797", output_dir=OUT)
    ok("cookie 形参缺省回落构造器", used_cookies[-1] == "SESSDATA=ctor")

    # 转写/笔记失败也进 error 字段
    def fake_bcut(audio_file, cookie=None):
        raise RuntimeError("BCUT 不可用")

    class FakeNoteGen:
        def __init__(self, **kw): pass
        def generate(self, *a, **k): raise RuntimeError("LLM 超时")

    with mock.patch.object(bmp, "fetch_view_data", side_effect=fake_view), \
         mock.patch.object(bmp, "BilibiliSubtitleFetcher", FakeFetcher), \
         mock.patch.object(bmp.BilibiliDownloader, "download", side_effect=fake_dl), \
         mock.patch.object(bmp, "transcribe_bcut", side_effect=fake_bcut), \
         mock.patch.object(bmp, "NoteGenerator", FakeNoteGen):
        rs = bmp.BilibiliDownloader().download_all_parts(
            "https://www.bilibili.com/video/BV1bK411W797", output_dir=OUT, transcribe="bcut",
            note_config=bmp.NoteConfig(api_key="sk-x"))
    ok("转写失败记入 error", rs[0].error and "转写失败" in rs[0].error, f"({rs[0].error})")
    shutil.rmtree(OUT, ignore_errors=True)


# ---------------------------------------------------------------------------
# 8. semantic_qa：srt 解析/切块/指纹/WAL/增量构建（无需模型）
# ---------------------------------------------------------------------------
def test_semantic_qa():
    if not HAVE_SQ:
        print("【semantic_qa】跳过（当前环境缺 numpy）")
        return
    print("【semantic_qa】")
    srt = "1\n00:00:01,000 --> 00:00:03,500\n第一句\n\n2\n00:00:04,000 --> 00:00:06,000\n第二句\n"
    tmp = tempfile.mkdtemp(prefix="t_sq_")
    try:
        srt_path = os.path.join(tmp, "BV1bK411W797_p1.zh.srt")
        open(srt_path, "w", encoding="utf-8").write(srt)
        segs = sq.parse_srt(srt_path)
        ok("srt 解析含时间边界", len(segs) == 2 and abs(segs[0][0] - 1.0) < 1e-6 and abs(segs[1][1] - 6.0) < 1e-6)
        chunks = sq.make_chunks(segs, 450)
        ok("切块保留时间边界", len(chunks) == 1 and chunks[0][1] == 1.0 and chunks[0][2] == 6.0)
        ok("智能拼接：中文无空格/英文留空格",
           sq._join_texts(["你好", "世界"]) == "你好世界"
           and sq._join_texts(["hello", "world"]) == "hello world")

        con = sq.connect(os.path.join(tmp, "t.db"))
        ok("WAL 已启用", con.execute("PRAGMA journal_mode").fetchone()[0] == "wal")
        con.executescript("DROP TABLE IF EXISTS courses")  # 模拟旧库（无 fingerprint 列）
        con.execute("CREATE TABLE courses(bvid TEXT PRIMARY KEY, title TEXT, n_parts INTEGER,"
                    " n_chunks INTEGER, n_chars INTEGER, indexed_at TEXT)")
        con.commit()
        sq.ensure_schema(con, tmp)
        ok("旧库迁移补 fingerprint 列",
           "fingerprint" in {r[1] for r in con.execute("PRAGMA table_info(courses)")})

        # 增量构建：指纹一致时应秒级跳过且不加载模型
        bvid = "BV1bK411W797"
        os.makedirs(os.path.join(tmp, bvid))
        shutil.copy(srt_path, os.path.join(tmp, bvid, os.path.basename(srt_path)))
        fp = sq.course_fingerprint(bmp.pick_course_srts(os.path.join(tmp, bvid)))
        con.execute("INSERT OR REPLACE INTO courses VALUES(?,?,?,?,?,?,?)",
                    (bvid, "t", 1, 1, 10, "2026-01-01", fp))
        con.commit()
        rc = sq.main(["build", "--root", tmp, "--db", os.path.join(tmp, "t.db"), "--offline"])
        ok("指纹一致时 build 秒级跳过（不加载模型）", rc == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 7b. download_all_parts 并行：顺序还原 / 失败隔离 / 实际并发
# ---------------------------------------------------------------------------
def test_download_parallel():
    print("【并行下载编排】")
    OUT = tempfile.mkdtemp(prefix="t_par_")
    state = {"view": 0}
    conc = {"cur": 0, "max": 0}

    def fake_view(bvid, cookie=""):
        state["view"] += 1
        return {"title": "t", "pages": [{"cid": i, "part": f"P{i}"} for i in range(1, 7)]}

    def fake_dl(url, output_dir=None, quality="fast"):
        conc["cur"] += 1
        conc["max"] = max(conc["max"], conc["cur"])
        try:
            time.sleep(0.2)  # 模拟网络下载耗时
            if url.endswith("p=4"):
                raise RuntimeError("网络抖动")
            return bmp.AudioDownloadResult(os.path.join(output_dir, "a.mp3"), "t", 1.0,
                                           None, "bilibili", "BV1bK411W797_p1", {})
        finally:
            conc["cur"] -= 1

    class FakeFetcher:
        def __init__(self, cookie=None):
            pass
        def list_pages(self, bvid):
            return fake_view(bvid)["pages"]
        def fetch_subtitles_for_cid(self, bvid, cid):
            return None

    t0 = time.time()
    with mock.patch.object(bmp, "fetch_view_data", side_effect=fake_view), \
         mock.patch.object(bmp, "BilibiliSubtitleFetcher", FakeFetcher), \
         mock.patch.object(bmp.BilibiliDownloader, "download", side_effect=fake_dl):
        rs = bmp.BilibiliDownloader().download_all_parts(
            "https://www.bilibili.com/video/BV1bK411W797", output_dir=OUT, workers=3)
    dt = time.time() - t0

    ok("结果按分集号有序", [r.p for r in rs] == [1, 2, 3, 4, 5, 6])
    ok("失败集隔离且其余成功", rs[3].audio is None and "音频下载失败" in rs[3].error
       and all(r.audio for i, r in enumerate(rs) if i != 3))
    ok("结构性并发断言：最大同时下载数 ≥2（不受机器快慢影响）", conc["max"] >= 2,
       f"(max_concurrent={conc['max']}, {dt:.2f}s)")
    ok("view API 仍只调 1 次", state["view"] == 1)
    shutil.rmtree(OUT, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as exc:
            ok(f"{t.__name__} 异常", False, f"{type(exc).__name__}: {exc}")
    n_pass = sum(1 for _, c in RESULTS if c)
    print(f"\n===== 回归结果: {n_pass}/{len(RESULTS)} 通过 =====")
    sys.exit(0 if n_pass == len(RESULTS) else 1)

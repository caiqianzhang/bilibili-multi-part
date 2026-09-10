#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B 站分 P 视频「全部分集」下载 + LLM 笔记生成脚本（自包含，零项目依赖）。

这是后端 BilibiliDownloader.download_all_parts() 的可移植副本，逻辑一致，
但只依赖 yt-dlp + requests（LLM 笔记生成另需 openai）、不 import 项目内任何
模块，可独立运行或被其它脚本 import。后端实现若变动，需同步此文件。

做什么
------
给一个 B 站视频链接（可带 ?p=N，会被忽略），枚举**全部**分集，逐集下载：
  - 音频：每集必下（mp3，yt-dlp）
  - 字幕：有就落盘 .srt（B 站 player API 直拉，无需下视频）；
          无字幕/无登录态则跳过，不阻断音频。
  - 笔记：有转写结果且指定 --note 时，调用 OpenAI 兼容 LLM 生成结构化
          Markdown 笔记（切块→逐块生成→合并→时间标记转跳转链接），落 .md。

用法
----
    # CLI：下载全部分集到 ./bilibili_out/<BV号>/（按课程分子目录）
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

    # 作为库 import
    from bilibili_multi_part import BilibiliDownloader, NoteConfig
    results = BilibiliDownloader().download_all_parts(url, output_dir="out")

环境变量（可代替命令行参数）
----------------------------
启动时自动读取脚本同目录的 .env（KEY=VALUE，不覆盖已有环境变量）。

    BILIBILI_COOKIE       B 站原始 cookie 串（如 "SESSDATA=xxx; ..."）
    BILIBILI_COOKIE_FILE  Netscape 格式 cookie 文件路径
    BILIBILI_OUTPUT       输出目录默认值
    LLM_API_KEY           LLM API Key（--note 时等效 --api-key）
    LLM_BASE_URL          LLM 接口 Base URL（等效 --base-url）
    LLM_MODEL             LLM 模型名（等效 --model，默认 gpt-4o-mini）
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
import tempfile
import time
import urllib.parse
import weakref
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

import requests
import yt_dlp

logger = logging.getLogger("bilibili_multi_part")

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
VIEW_API = "https://api.bilibili.com/x/web-interface/view"
PLAYER_API = "https://api.bilibili.com/x/player/wbi/v2"

# --quality 档位 → mp3 码率 (kbps)。fast 文件最小，够 ASR 转写/字幕对齐用
QUALITY_BITRATES = {"fast": "64", "medium": "128", "slow": "320"}


# ---------------------------------------------------------------------------
# 极简数据模型（自包含，不依赖项目内的 models 模块）
# ---------------------------------------------------------------------------
@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    language: Optional[str]
    full_text: str
    segments: List[TranscriptSegment]
    raw: Optional[dict] = None


@dataclass
class AudioDownloadResult:
    file_path: str
    title: str
    duration: float
    cover_url: Optional[str]
    platform: str
    video_id: str
    raw_info: dict


@dataclass
class BilibiliPartResult:
    """单集下载结果。audio=None 表示该集音频下载失败（error 记录原因）。"""
    p: int
    bvid: str
    cid: Optional[int]
    title: str
    audio: Optional[AudioDownloadResult]
    transcript: Optional[TranscriptResult] = None
    subtitle_path: Optional[str] = None
    note_path: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SRT / 文本工具函数（v1/v2 批量任务脚本共用，避免各写一份导致漂移）
# ---------------------------------------------------------------------------
def _seg_field(seg, key, default=0):
    """兼容 dict 与 dataclass 两种 segment 表达（批量脚本用 dict，库内用 dataclass）。"""
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def fmt_ts(seconds: float) -> str:
    """SRT 时间戳: HH:MM:SS,mmm（与 BilibiliDownloader._fmt_ts 一致）。"""
    ms = int(round(float(seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, path: str) -> str:
    """把 segment 列表写成 .srt，返回写入路径。"""
    lines = []
    for i, seg in enumerate(segments, 1):
        text = (_seg_field(seg, "text", "") or "").strip()
        lines += [str(i),
                  f"{fmt_ts(_seg_field(seg, 'start', 0))} --> {fmt_ts(_seg_field(seg, 'end', 0))}",
                  text, ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def parse_srt_text(path: str) -> List[str]:
    """从 .srt 提取纯文本行（跳过序号/时间轴/空行），断点续跑时复用已有字幕。"""
    return [line.rstrip("\n") for line in open(path, encoding="utf-8")
            if line.strip() and not line.strip().isdigit() and "-->" not in line]


# ---------------------------------------------------------------------------
# 语料布局约定（下载器 / funasr 批量任务 / semantic_qa 索引三环共用的"单一真源"）
# ---------------------------------------------------------------------------
# 字幕文件命名契约: {bvid}_p{n}.{lang}.srt
SRT_FILE_RE = re.compile(r"^(BV[0-9A-Za-z]{10})_p(\d+)\.(.+)\.srt$")


def srt_lang_rank(path: str):
    """同分集多份字幕的择优序：人工字幕 > AI 字幕 > 本地 ASR 产物。"""
    b = os.path.basename(path)
    return (".asr" in b, ".ai" in b, b)


def pick_course_srts(course_dir: str) -> dict:
    """扫描课程目录，返回 {p: 最优字幕路径}（每集只取择优后的一份）。"""
    best = {}
    for f in glob.glob(os.path.join(course_dir, "*.srt")):
        m = SRT_FILE_RE.match(os.path.basename(f))
        if not m:
            continue
        p = int(m.group(2))
        if p not in best or srt_lang_rank(f) < srt_lang_rank(best[p]):
            best[p] = f
    return best


def bilibili_jump_url(bvid: str, p: int, seconds: float) -> str:
    """原片时间点跳转链接（笔记时间标记、语义问答引用共用此格式）。"""
    return f"https://www.bilibili.com/video/{bvid}?p={p}&t={int(seconds)}"


# ---------------------------------------------------------------------------
# yt-dlp 补丁：B 站 wbi/playurl 网关需要 dm_img_* / web_location 参数
# （内联在此以保持脚本自包含，不依赖项目内的 bilibili_dm_patch 模块）
# ---------------------------------------------------------------------------
def apply_bilibili_dm_img_patch() -> bool:
    try:
        from yt_dlp.extractor.bilibili import BilibiliBaseIE
    except Exception as e:  # pragma: no cover - yt-dlp 缺失或内部结构变更
        logger.warning("跳过 dm_img 补丁，无法导入 BilibiliBaseIE: %s", e)
        return False

    if getattr(BilibiliBaseIE._download_playinfo, "_bili_dm_patched", False):
        return True

    def build():
        return {
            "web_location": 1550101,
            "dm_img_list": "[]",
            "dm_img_str": base64.b64encode(
                "".join(random.choices(string.printable, k=random.randint(16, 64))).encode()
            )[:-2].decode(),
            "dm_cover_img_str": base64.b64encode(
                "".join(random.choices(string.printable, k=random.randint(32, 128))).encode()
            )[:-2].decode(),
            "dm_img_inter": '{"ds":[],"wh":[6093,6631,31],"of":[430,760,380]}',
        }

    original = BilibiliBaseIE._download_playinfo

    def _patched(self, bvid, cid, headers=None, query=None, **kwargs):
        # 只合并 dm_img_* / web_location；fatal 等原的关键字参数原样透传，否则真实下载会崩
        return original(self, bvid, cid, headers=headers,
                        query={**build(), **(query or {})}, **kwargs)

    _patched._bili_dm_patched = True
    BilibiliBaseIE._download_playinfo = _patched
    return True


# ---------------------------------------------------------------------------
# URL 解析
# ---------------------------------------------------------------------------
def extract_bvid(url: str) -> Optional[str]:
    """从 B 站链接提取 BV 号（含 b23.tv 短链解析）。BV 号固定为 BV + 10 位字母数字。"""
    if "b23.tv" in url:
        try:
            url = requests.head(url, allow_redirects=True, timeout=10).url
        except Exception as e:
            logger.warning("b23.tv 短链解析失败: %s", e)
    m = re.search(r"(?<![0-9A-Za-z])BV([0-9A-Za-z]{10})(?![0-9A-Za-z])", url)
    return f"BV{m.group(1)}" if m else None


# ---------------------------------------------------------------------------
# Cookie 工具（搜索 / 下载共用）
# ---------------------------------------------------------------------------
def normalize_cookie(cookie: Optional[str]) -> str:
    """统一 cookie 规范化：去首尾空白 + URL 解码（SESSDATA 等常带 %2C 类编码）。

    所有消费 cookie 的入口（yt-dlp 文件注入 / 请求头 / 搜索签名）都必须走这里，
    保证同一串 cookie 在下载、字幕、BCUT、搜索四处行为一致。
    """
    if not cookie:
        return ""
    return urllib.parse.unquote(cookie.strip())


def _remove_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def write_cookie_file(cookie: Optional[str]) -> Optional[str]:
    """把原始 cookie 串写成 Netscape 格式临时文件，返回文件路径（调用方负责删除）。

    yt-dlp 没有 ydl_opts["cookie"] 参数（未知参数会被静默忽略），原始 cookie 串
    只能经 Netscape 文件注入。cookie 为空返回 None。
    """
    cookie = normalize_cookie(cookie)
    if not cookie:
        return None
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write("# Netscape HTTP Cookie File\n")
    # 按裸分号切分并去空白：带空格（"a=1; b=2"）与不带空格（"a=1;b=2"）两种写法都兼容
    for pair in cookie.split(";"):
        if "=" in pair:
            k, v = (s.strip() for s in pair.split("=", 1))
            if k:
                tmp.write(f".bilibili.com\tTRUE\t/\tFALSE\t0\t{k}\t{v}\n")
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# 字幕直拉（B 站 player API，无需下载视频）
# ---------------------------------------------------------------------------
def fetch_view_data(bvid: str, cookie: str = "") -> Optional[dict]:
    """调 view API 拿视频元数据（分 P 列表 / UP 主 / 时长等），无需登录。

    BilibiliSubtitleFetcher 与 search_bilibili 共用；请求失败或 code!=0 返回 None。
    """
    headers = {"User-Agent": UA, "Referer": "https://www.bilibili.com"}
    cookie = normalize_cookie(cookie)
    if cookie:
        headers["Cookie"] = cookie
    try:
        resp = requests.get(VIEW_API, params={"bvid": bvid}, headers=headers, timeout=10)
        data = resp.json()
    except Exception as e:
        logger.warning("view API 请求失败: %s", e)
        return None
    if data.get("code") != 0:
        logger.warning("view API 返回错误: code=%s msg=%s", data.get("code"), data.get("message"))
        return None
    return data.get("data") or {}


class BilibiliSubtitleFetcher:
    def __init__(self, cookie: Optional[str] = None):
        self._cookie = normalize_cookie(cookie)

    def _headers(self) -> dict:
        h = {"User-Agent": UA, "Referer": "https://www.bilibili.com"}
        if self._cookie:
            h["Cookie"] = self._cookie
        return h

    def _view(self, bvid: str) -> Optional[dict]:
        return fetch_view_data(bvid, self._cookie)

    def list_pages(self, bvid: str) -> List[dict]:
        """全部分集（分 P 视频返回所有集；单集返回长度 1 的列表）。"""
        data = self._view(bvid)
        if not data:
            return []
        return data.get("pages", []) or []

    def _list_subtitles(self, bvid: str, cid: int) -> List[dict]:
        try:
            resp = requests.get(PLAYER_API, params={"bvid": bvid, "cid": cid},
                                headers=self._headers(), timeout=10)
            data = resp.json()
        except Exception as e:
            logger.warning("player API 请求失败: %s", e)
            return []
        if data.get("code") != 0:
            logger.warning("player API 返回错误: code=%s msg=%s", data.get("code"), data.get("message"))
            return []
        return data.get("data", {}).get("subtitle", {}).get("subtitles", []) or []

    @staticmethod
    def _pick(subtitles: List[dict]) -> Optional[dict]:
        if not subtitles:
            return None

        def is_zh(s):
            lan = (s.get("lan") or "").lower()
            return lan.startswith("zh") or lan == "ai-zh"

        for s in subtitles:  # 人工中文
            if is_zh(s) and not s.get("ai_type"):
                return s
        for s in subtitles:  # AI 中文
            if is_zh(s):
                return s
        return subtitles[0]  # 任意非空

    @staticmethod
    def _norm(url: str) -> str:
        return "https:" + url if url.startswith("//") else url

    def _fetch_body(self, subtitle_url: str) -> Optional[List[dict]]:
        try:
            resp = requests.get(self._norm(subtitle_url), headers=self._headers(), timeout=15)
            return resp.json().get("body") or []
        except Exception as e:
            logger.warning("字幕 body 下载失败: %s", e)
            return None

    def fetch_subtitles(self, video_url: str) -> Optional[TranscriptResult]:
        bvid = extract_bvid(video_url)
        if not bvid:
            logger.info("无法从 URL 提取 BV id: %s", video_url)
            return None
        m = re.search(r"[?&]p=(\d+)", video_url)
        p = int(m.group(1)) if m else None

        data = self._view(bvid)
        if not data:
            return None
        pages = data.get("pages", []) or []
        if pages and p and 1 <= p <= len(pages):
            cid = pages[p - 1].get("cid")
        elif pages:
            cid = pages[0].get("cid")
        else:
            cid = data.get("cid")
        if not cid:
            logger.info("%s (p=%s) 未取到 cid", bvid, p)
            return None

        result = self.fetch_subtitles_for_cid(bvid, cid)
        if result:
            logger.info("字幕下载成功: %s p=%s lan=%s %d 段", bvid, p, result.language, len(result.segments))
        return result

    def fetch_subtitles_for_cid(self, bvid: str, cid: int) -> Optional[TranscriptResult]:
        """按已知 cid 直拉字幕（调用方已持有分 P 列表时可省一次 view API）。"""
        if not cid:
            return None
        subtitles = self._list_subtitles(bvid, cid)
        if not subtitles:
            logger.info("%s (cid=%s) 无字幕轨", bvid, cid)
            return None
        track = self._pick(subtitles)
        if not track or not track.get("subtitle_url"):
            logger.info("%s 字幕轨无 subtitle_url（可能需登录）", bvid)
            return None

        body = self._fetch_body(track["subtitle_url"])
        if not body:
            return None
        segments = []
        for item in body:
            text = (item.get("content") or "").strip()
            if text:
                segments.append(TranscriptSegment(
                    start=float(item.get("from", 0)),
                    end=float(item.get("to", 0)),
                    text=text,
                ))
        if not segments:
            return None
        lan = track.get("lan") or "zh"
        return TranscriptResult(
            language=lan,
            full_text=" ".join(s.text for s in segments),
            segments=segments,
            raw={"source": "bilibili_player_api", "bvid": bvid, "cid": cid, "lan": lan},
        )


# ---------------------------------------------------------------------------
# BCUT（必剪）语音识别：把音频文件上传到 B 站 member.bilibili.com 的
# rubick-interface，异步转写，轮询取结果。免费，但需要 B 站登录态 cookie。
# ---------------------------------------------------------------------------
BCUT_API = "https://member.bilibili.com/x/bcut/rubick-interface"
BCUT_REQ_UPLOAD = BCUT_API + "/resource/create"
BCUT_COMMIT_UPLOAD = BCUT_API + "/resource/create/complete"
BCUT_CREATE_TASK = BCUT_API + "/task"
BCUT_QUERY_RESULT = BCUT_API + "/task/result"


class BcutTranscriber:
    """必剪语音识别（B 站官方 ASR，免费，需登录态 cookie）。"""

    def __init__(self, cookie: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Bilibili/1.0.0 (https://www.bilibili.com)",
            "Content-Type": "application/json",
        })
        cookie = normalize_cookie(cookie)
        if cookie:
            self.session.headers["Cookie"] = cookie

    def _upload(self, file_path: str) -> dict:
        # 只取文件大小，不整读文件：全量字节由 transcript() 读一次，避免双份内存
        size = os.path.getsize(file_path)
        if size <= 0:
            raise ValueError("无法读取文件数据")

        payload = json.dumps({
            "type": 2, "name": "audio.mp3", "size": size,
            "ResourceFileType": "mp3", "model_id": "8",
        })
        resp = self.session.post(BCUT_REQ_UPLOAD, data=payload)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"申请上传失败: {data.get('message')}")
        return data["data"]

    @staticmethod
    def _upload_parts(session, binary: bytes, upload_urls: List[str], per_size: int) -> str:
        etags = []
        for i, url in enumerate(upload_urls):
            start = i * per_size
            end = min((i + 1) * per_size, len(binary))
            r = session.put(url, data=binary[start:end],
                            headers={"Content-Type": "application/octet-stream"})
            r.raise_for_status()
            etags.append(r.headers.get("Etag", "").strip('"'))
        return ",".join(etags)

    def _commit_upload(self, meta: dict, etags: str) -> str:
        payload = json.dumps({
            "InBossKey": meta["in_boss_key"],
            "ResourceId": meta["resource_id"],
            "Etags": etags,
            "UploadId": meta["upload_id"],
            "model_id": "8",
        })
        resp = self.session.post(BCUT_COMMIT_UPLOAD, data=payload)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"提交上传失败: {data.get('message')}")
        return data["data"]["download_url"]

    def _create_task(self, download_url: str) -> str:
        resp = self.session.post(BCUT_CREATE_TASK,
                                 json={"resource": download_url, "model_id": "8"})
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"创建任务失败: {data.get('message')}")
        return data["data"]["task_id"]

    def _query_result(self, task_id: str) -> dict:
        resp = self.session.get(BCUT_QUERY_RESULT,
                                params={"model_id": 7, "task_id": task_id})
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"查询结果失败: {data.get('message')}")
        return data["data"]

    def transcript(self, file_path: str) -> TranscriptResult:
        """上传音频 → 创建转写任务 → 轮询 → 返回 TranscriptResult。"""
        with open(file_path, "rb") as f:
            binary = f.read()

        meta = self._upload(file_path)
        etags = self._upload_parts(self.session, binary, meta["upload_urls"], meta["per_size"])
        download_url = self._commit_upload(meta, etags)
        task_id = self._create_task(download_url)

        result_data = None
        for _ in range(500):
            result_data = self._query_result(task_id)
            state = result_data.get("state")
            if state == 4:      # 完成
                break
            if state == 3:      # 失败
                raise RuntimeError(f"B站 ASR 任务失败: state=3")
            time.sleep(1)
        else:
            raise RuntimeError("B站 ASR 任务超时未完成")

        raw = json.loads(result_data.get("result") or "{}")
        segments = []
        full_text = ""
        for u in raw.get("utterances", []):
            text = (u.get("transcript") or "").strip()
            if not text:
                continue
            full_text += text + " "
            segments.append(TranscriptSegment(
                start=float(u.get("start_time", 0)) / 1000.0,
                end=float(u.get("end_time", 0)) / 1000.0,
                text=text,
            ))
        return TranscriptResult(
            language=raw.get("language", "zh"),
            full_text=full_text.strip(),
            segments=segments,
            raw=raw,
        )


def transcribe_bcut(audio_file: str, cookie: Optional[str] = None) -> TranscriptResult:
    """便捷函数：用 BCUT 转写音频文件。"""
    return BcutTranscriber(cookie=cookie).transcript(audio_file)


# ---------------------------------------------------------------------------
# B 站视频搜索（免费，无需 cookie）
# ---------------------------------------------------------------------------
@dataclass
class BilibiliSearchResult:
    """单条搜索结果。"""
    bvid: str
    title: str
    uploader: str
    duration: float
    cover: Optional[str]
    url: str


# ---------------------------------------------------------------------------
# B 站搜索主路径：网页同款 wbi 签名接口（免登录可用，元数据齐全）
#
# 与网页搜索一致的四步：
#   1. 访问首页让 B 站签发真实设备 cookie（buvid3/b_nut）——yt-dlp 是本地随机
#      伪造 buvid3，查无此设备，这是脚本搜索 412 而网页不 412 的关键差别
#   2. nav 接口拿 wbi 密钥（img_key/sub_key 藏在头像 url 文件名里）
#   3. 对参数做 wts + w_rid 签名（固定混淆表重排密钥后 md5）
#   4. 调 wbi/search/all/v2，视频结果在 result_type=="video" 分块里
# ---------------------------------------------------------------------------
WBI_MIXIN_TAB = [46,47,18,2,53,8,23,32,15,50,10,31,58,3,45,35,27,43,5,49,
                 33,9,42,19,29,28,14,39,12,38,41,13,37,48,7,16,24,55,40,61,
                 26,17,0,1,60,51,30,4,22,25,54,21,56,59,6,63,57,62,11,36,20,34,44,52]


def _load_cookie_pairs(cookie: Optional[str], cookie_file: Optional[str]) -> dict:
    """把原始 cookie 串 / Netscape cookie 文件解析成 {name: value}（B 站域）。"""
    pairs: dict = {}
    cookie = normalize_cookie(cookie)
    if cookie:
        for pair in cookie.split(";"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                pairs[k.strip()] = v.strip()
    if cookie_file and os.path.exists(cookie_file):
        with open(cookie_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("#HttpOnly_"):
                    line = line[len("#HttpOnly_"):]
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) >= 7 and "bilibili.com" in fields[0]:
                    pairs[fields[5]] = fields[6]
    return pairs


def _sign_wbi(params: dict, mixin_key: str) -> dict:
    """wbi 参数签名：过滤特殊字符 → 加 wts → 按 key 排序 → urlencode → md5。"""
    params = {k: "".join(ch for ch in str(v) if ch not in "!'()*")
              for k, v in params.items()}
    params["wts"] = int(time.time())
    query = urllib.parse.urlencode(dict(sorted(params.items())), quote_via=urllib.parse.quote)
    params["w_rid"] = hashlib.md5((query + mixin_key).encode()).hexdigest()
    return params


def _duration_to_seconds(value) -> float:
    """搜索接口时长是 'mm:ss' / 'hh:mm:ss' 字符串（mm 段可超过 59），也兼容秒数。"""
    if isinstance(value, (int, float)):
        return float(value)
    parts = [int(p) for p in str(value or "").split(":") if p.strip().isdigit()]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return float(seconds)


def search_bilibili_webapi(query: str, max_results: int = 20,
                           cookie: Optional[str] = None,
                           cookie_file: Optional[str] = None,
                           retries: int = 2) -> List[BilibiliSearchResult]:
    """网页同款 wbi 签名搜索，返回条目自带 UP 主/时长/封面，无需再补全。"""
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Referer": "https://www.bilibili.com/"})
    pairs = _load_cookie_pairs(cookie, cookie_file)
    for name, value in pairs.items():
        session.cookies.set(name, value, domain=".bilibili.com")
    if "buvid3" not in pairs:
        session.get("https://www.bilibili.com/", timeout=10)  # 领取设备 cookie

    nav = session.get("https://api.bilibili.com/x/web-interface/nav", timeout=10).json()
    wbi = (nav.get("data") or {}).get("wbi_img") or {}
    img_key = (wbi.get("img_url") or "").rsplit("/", 1)[-1].split(".")[0]
    sub_key = (wbi.get("sub_url") or "").rsplit("/", 1)[-1].split(".")[0]
    if not img_key or not sub_key:
        raise RuntimeError(f"nav 接口未返回 wbi 密钥: code={nav.get('code')}")
    mixin_key = "".join((img_key + sub_key)[i] for i in WBI_MIXIN_TAB)[:32]

    last_err = None
    for attempt in range(retries):
        try:
            # 分页拉取直到凑满 max_results / 翻完（单页约 20 条视频）
            results: List[BilibiliSearchResult] = []
            max_pages = min(10, -(-max_results // 20) + 1)
            for page in range(1, max_pages + 1):
                data = session.get(
                    "https://api.bilibili.com/x/web-interface/wbi/search/all/v2",
                    params=_sign_wbi({"keyword": query, "page": page}, mixin_key),
                    headers={"Referer": "https://www.bilibili.com/search?keyword="
                                        + urllib.parse.quote(query)},
                    timeout=10,
                ).json()
                if data.get("code") != 0:
                    raise RuntimeError(f"搜索接口错误: code={data.get('code')} msg={data.get('message')}")
                blocks = (data.get("data") or {}).get("result") or []
                video_block = next((b for b in blocks if b.get("result_type") == "video"), None)
                items = (video_block or {}).get("data", [])
                if not items:
                    break  # 没有更多结果
                for item in items:
                    bvid = item.get("bvid")
                    if not bvid:
                        continue
                    results.append(BilibiliSearchResult(
                        bvid=bvid,
                        # 标题里的 <em class="keyword"> 是命中高亮标签，去掉
                        title=re.sub(r"<[^>]+>", "", item.get("title") or "").strip(),
                        uploader=(item.get("author") or "").strip(),
                        duration=_duration_to_seconds(item.get("duration")),
                        cover=item.get("pic"),
                        url=f"https://www.bilibili.com/video/{bvid}",
                    ))
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break
            return results
        except Exception as exc:
            last_err = exc
            logger.warning("wbi 搜索失败 (attempt %d/%d): %s", attempt + 1, retries, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"wbi 搜索失败（已重试 {retries} 次）: {last_err}")


def search_bilibili(query: str, max_results: int = 20, retries: int = 3,
                    cookie: Optional[str] = None,
                    cookie_file: Optional[str] = None) -> List[BilibiliSearchResult]:
    """
    搜索 B 站视频。

    优先走网页同款 wbi 签名接口（search_bilibili_webapi，免登录、稳、元数据齐全）；
    失败（如 B 站风控策略变动）时退回 yt-dlp 的 BiliBiliSearchIE 旧接口，
    并用 view API 补全 UP 主 / 时长。

    :param query: 搜索关键词
    :param max_results: 最多返回条数（wbi 主路径自动翻页，每页约 20 条）
    :param retries: 失败重试次数（含退避），wbi 主路径与 yt-dlp 兜底路径共用
    :param cookie: B 站原始 cookie 串（如 "SESSDATA=xxx; ..."），可选
    :param cookie_file: Netscape 格式 cookie 文件路径（与 cookie 二选一即可）
    :return: 搜索结果列表
    """
    try:
        return search_bilibili_webapi(query, max_results=max_results,
                                      cookie=cookie, cookie_file=cookie_file,
                                      retries=retries)
    except Exception as exc:
        logger.warning("wbi 搜索不可用，退回 yt-dlp 路径: %s", exc)

    # yt-dlp 只认 cookiefile 选项；cookie_file 优先，否则把原始串转临时文件
    # （write_cookie_file 内部统一做 URL 解码，此处不再重复处理）
    tmp_cookie_file = None
    if cookie_file is None and cookie:
        tmp_cookie_file = write_cookie_file(cookie)

    last_err = None
    try:
        for attempt in range(retries):
            try:
                ydl_opts = {
                    "quiet": True, "no_warnings": True,
                    # 只要 BV 号列表（extract_flat），不做完整展开：完整展开会把
                    # 多 P 课程逐分 P 提取（极慢、极易触发 412），且付费课程等
                    # 无 extractor 的条目会让整个搜索报错。元数据由下方 view API 补全。
                    "extract_flat": "in_playlist",
                }
                if cookie_file or tmp_cookie_file:
                    ydl_opts["cookiefile"] = cookie_file or tmp_cookie_file
                ydl = yt_dlp.YoutubeDL(ydl_opts)
                info = ydl.extract_info(f"bilisearch3:{query}", download=False)
                results = []
                for e in info.get("entries", []) or []:
                    bvid = e.get("id")
                    if not str(bvid or "").startswith("BV"):
                        # 展开失败的占位条目 id 是数字 aid，从 url 里找回 BV 号
                        bvid = extract_bvid(e.get("url") or "") or bvid
                    if not bvid or not str(bvid).startswith("BV"):
                        continue  # 付费课程 / 直播等非普通视频条目，无 BV 号
                    title = (e.get("title") or "").strip()
                    uploader = (e.get("uploader") or "").strip()
                    duration = float(e.get("duration") or 0)
                    cover = e.get("thumbnail")
                    # 扁平条目只有 id/url，元数据统一由 view API 补全（免登录）
                    if not uploader or not duration:
                        meta = fetch_view_data(bvid, cookie=cookie or "")
                        if meta:
                            title = title or (meta.get("title") or "").strip()
                            uploader = uploader or ((meta.get("owner") or {}).get("name") or "").strip()
                            duration = duration or float(meta.get("duration") or 0)
                            cover = cover or meta.get("pic")
                    results.append(BilibiliSearchResult(
                        bvid=bvid,
                        title=title,
                        uploader=uploader,
                        duration=duration,
                        cover=cover,
                        url=f"https://www.bilibili.com/video/{bvid}",
                    ))
                    if len(results) >= max_results:
                        break
                return results
            except Exception as exc:  # 412 等风控 → 退避重试
                last_err = exc
                logger.warning("B 站搜索失败 (attempt %d/%d): %s", attempt + 1, retries, exc)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"B 站搜索失败（已重试 {retries} 次）: {last_err}")
    finally:
        if tmp_cookie_file:
            try:
                os.unlink(tmp_cookie_file)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# LLM 笔记生成（对后端 UniversalGPT.summarize() 的可移植子集）
#
# 后端 note.py::generate() 在「下载 → 转写」之后会调用 GPT 把转写结果重写成
# 结构化 Markdown 笔记，核心四步：
#   1. RequestChunker 把长转写切成 LLM 可容纳的批次
#   2. 逐批调用 OpenAI 兼容接口，每批得到一份部分笔记（partial）
#   3. 多份 partial 用 MERGE_PROMPT 合并成一份完整笔记
#   4. 后处理：把 *Content-[mm:ss] 标记转成可跳回原片的链接 + 补来源链接
# 本脚本只移植这四步；截图 / 断点续跑等后端特有功能未移植（不下载视频）。
# ---------------------------------------------------------------------------

# -- Prompt 模板（与后端 app/gpt/prompt.py 一致） --
BASE_PROMPT = '''
你是一个专业的笔记助手，擅长将视频转录内容整理成清晰、有条理且信息丰富的笔记。

语言要求：
- 笔记必须使用 **中文** 撰写。
- 专有名词、技术术语、品牌名称和人名应适当保留 **英文**。

视频标题：
{video_title}

视频标签：
{tags}



输出说明：
- 仅返回最终的 **Markdown 内容**。
- **不要**将输出包裹在代码块中（例如：```` ```markdown ````，```` ``` ````）。
请注意，在生成 Markdown 时，避免将编号标题（如“1. **内容**”）写成有序列表的格式，以免解析错误。

- 如果要加粗并保留编号，应使用 `1\\. **内容**`（加反斜杠），防止被误解析为有序列表。
- 或者使用 `## 1. 内容` 的形式作为标题。

请确保以下格式 **不会出现误渲染**：
 `1. **xxx**`
 `1\\. **xxx**` 或 `## 1. xxx`

视频分段（格式：开始时间 - 内容）：

---
{segment_text}
---

你的任务：
根据上面的分段转录内容，生成结构化的笔记，遵循以下原则：

1. **完整信息**：记录尽可能多的相关细节，确保内容全面。
2. **去除无关内容**：省略广告、填充词、问候语和不相关的言论。
3. **保留关键细节**：保留重要事实、示例、结论和建议。(如果额外重要的任务有格式需求可以不遵守)
4. **可读布局**：必要时使用项目符号，并保持段落简短，增强可读性。(如果额外重要的任务有格式需求可以不遵守)
5. 视频中提及的数学公式必须保留，并以 LaTeX 语法形式呈现，适合 Markdown 渲染。


请始终遵循此规则。

额外重要的任务如下(每一个都必须严格完成):

'''


LINK = '''
9. **Add time markers**: THIS IS IMPORTANT For every main heading (`##`), append the starting time of that segment using the format ,start with *Content ,eg: `*Content-[mm:ss]`.


'''

AI_SUM = '''

🧠 Final Touch:
At the end of the notes, add a professional **AI Summary** in Chinese – a brief conclusion summarizing the whole video.



'''

SCREENSHOT = '''
8. **Screenshot placeholders**: If a section involves **visual demonstrations, code walkthroughs, UI interactions**, or any content where visuals aid understanding, insert a screenshot cue at the end of that section:
   - Format: `*Screenshot-[mm:ss]`
   - Only use it when truly helpful.
'''

MERGE_PROMPT = '''
你将收到多个来自同一视频的 Markdown 笔记片段，请合并成一份完整笔记：
- 只做合并与去重，不要发明新内容
- 保持原有标题层级与 Markdown 结构
- 保留所有 *Content-[mm:ss] 与 *Screenshot-[mm:ss] 标记
- 保持中文输出，专有名词保留英文
- 不要使用代码块包裹输出
'''


# -- 笔记格式 / 风格选项（与后端 app/gpt/prompt_builder.py 一致） --
NOTE_FORMATS = {
    "toc": "目录",
    "link": "原片跳转",
    "screenshot": "原片截图",
    "summary": "AI总结",
}
NOTE_STYLES = {
    "minimal": "精简",
    "detailed": "详细",
    "academic": "学术",
    "tutorial": "教程",
    "xiaohongshu": "小红书",
    "life_journal": "生活向",
    "task_oriented": "任务导向",
    "business": "商业风格",
    "meeting_minutes": "会议纪要",
}


def _format_instruction(format_type: str) -> str:
    format_map = {
        "toc": "9. **目录**: 自动生成一个基于 `##` 级标题的目录。不需要插入原片跳转",
        "link": LINK,
        "screenshot": SCREENSHOT,
        "summary": AI_SUM,
    }
    return format_map.get(format_type, "")


def _style_instruction(style: str) -> str:
    return NOTE_STYLES.get(style, "")


def generate_base_prompt(title: str, segment_text: str, tags,
                         formats: Optional[List[str]] = None,
                         style: Optional[str] = None,
                         extras: Optional[str] = None) -> str:
    """拼装最终 prompt：基础模板 + 用户选的格式 + 风格 + 额外指令。"""
    prompt = BASE_PROMPT.format(video_title=title, segment_text=segment_text, tags=tags)
    if formats:
        prompt += "\n" + "\n".join(_format_instruction(f) for f in formats)
    if style:
        prompt += "\n" + _style_instruction(style)
    if extras:
        prompt += f"\n{extras}"
    return prompt


# -- 笔记后处理（与后端 app/utils/note_helper.py 一致） --
def prepend_source_link(markdown: Optional[str], source_url: str) -> Optional[str]:
    """在笔记开头添加来源链接；若首个非空行已包含来源链接，则更新该行并避免重复。"""
    if markdown is None:
        return None
    source = (source_url or "").strip()
    if not source:
        return markdown
    header = f"> 来源链接：{source}"
    lines = markdown.splitlines()
    first_non_empty_idx = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_non_empty_idx = idx
            break
    if first_non_empty_idx is not None:
        first_line = lines[first_non_empty_idx].strip()
        if first_line.startswith("> 来源链接：") or first_line.startswith("来源链接："):
            lines[first_non_empty_idx] = header
            return "\n".join(lines)
    if markdown.strip():
        return f"{header}\n\n{markdown}"
    return header


def replace_content_markers(markdown: str, video_id: str, platform: str = "bilibili") -> str:
    """把 *Content-[mm:ss]*、Content-mm:ss、Content-[h:mm:ss] 等标记替换为跳转链接。

    分钟/小时位不强制两位：既支持 04:16 也支持 65:30 与 1:02:15（长视频）。
    """
    pattern = r"(?:\*?)Content-\[?(\d{1,3}:\d{2}(?::\d{2})?)\]?"

    def replacer(match):
        label = match.group(1)
        parts = [int(x) for x in label.split(":")]
        if len(parts) == 3:
            total_seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            total_seconds = parts[0] * 60 + parts[1]

        if platform == "bilibili":
            parsed_video_id = video_id.replace("_p", "?p=")
            url = f"https://www.bilibili.com/video/{parsed_video_id}&t={total_seconds}"
        elif platform == "youtube":
            url = f"https://www.youtube.com/watch?v={video_id}&t={total_seconds}s"
        elif platform == "douyin":
            url = f"https://www.douyin.com/video/{video_id}"
            return f"[原片 @ {label}]({url})"
        else:
            return f"({label})"

        return f"[原片 @ {label}]({url})"

    return re.sub(pattern, replacer, markdown)


# ---------------------------------------------------------------------------
# 长转写切块器（与后端 app/gpt/request_chunker.py 一致，纯 Python 零依赖）
# ---------------------------------------------------------------------------
@dataclass
class ChunkPayload:
    segments: list
    image_urls: list


# 增量估算时每段附加的安全余量（字节）。真实增量误差只有 join 的 1 个换行字节
# （JSON 转义按字符可加，单段与整批一致），余量 2 保证估算恒 ≥ 真实值且不明显虚高
_CHUNK_SEP_MARGIN = 2


class RequestChunker:
    def __init__(self, message_builder: Callable, max_bytes: int, size_estimator: Optional[Callable] = None):
        self.message_builder = message_builder
        self.max_bytes = max_bytes
        self.size_estimator = size_estimator

    def estimate(self, messages) -> int:
        if self.size_estimator:
            return self.size_estimator(messages)
        return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))

    def _messages_size(self, segments, image_urls, **kwargs) -> int:
        messages = self.message_builder(segments, image_urls, **kwargs)
        return self.estimate(messages)

    def _get_text(self, segment) -> str:
        if isinstance(segment, dict):
            return segment.get("text", "")
        return getattr(segment, "text", "")

    def _make_segment(self, segment, text: str):
        if isinstance(segment, dict):
            new_seg = dict(segment)
            new_seg["text"] = text
            return new_seg
        if hasattr(segment, "__dict__"):
            data = dict(segment.__dict__)
            data["text"] = text
            return type(segment)(**data)
        return type(segment)(segment.start, segment.end, text)

    def _split_segment_to_fit(self, segment, **kwargs):
        text = self._get_text(segment)
        if not text:
            raise ValueError("empty segment cannot be split")
        lo, hi = 1, len(text)
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = self._make_segment(segment, text[:mid])
            size = self._messages_size([candidate], [], **kwargs)
            if size <= self.max_bytes:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            raise ValueError("single segment too large to fit request")
        head = self._make_segment(segment, text[:best])
        tail = self._make_segment(segment, text[best:])
        return head, tail

    def chunk(self, segments: list, image_urls: list, **kwargs) -> List[ChunkPayload]:
        segments = list(segments or [])
        image_urls = list(image_urls or [])
        if not segments and not image_urls:
            return []

        chunks: List[ChunkPayload] = []
        seg_idx = 0

        # 增量估算：整批 size ≈ 固定开销(base) + Σ 单段增量（+每段安全余量），
        # 避免旧实现每追加一段就整包 json.dumps 的 O(n²) 开销。
        # 超大单段的精确切分仍走 _split_segment_to_fit 的二分。
        base = self._messages_size([], [], **kwargs)
        extra_cache: dict = {}

        def extra(seg) -> int:
            key = id(seg)
            if key not in extra_cache:
                extra_cache[key] = max(0, self._messages_size([seg], [], **kwargs) - base)
            return extra_cache[key]

        while seg_idx < len(segments):
            batch_segments = []
            running = base
            while seg_idx < len(segments):
                # 空批不加余量：保证与精确判定一致，单段放得下就一定入批
                add = extra(segments[seg_idx]) + (_CHUNK_SEP_MARGIN if batch_segments else 0)
                if running + add <= self.max_bytes:
                    running += add
                    batch_segments.append(segments[seg_idx])
                    seg_idx += 1
                    continue
                if not batch_segments:
                    head, tail = self._split_segment_to_fit(segments[seg_idx], **kwargs)
                    segments[seg_idx] = head
                    segments.insert(seg_idx + 1, tail)
                    continue
                break

            if not batch_segments:
                raise ValueError("unable to fit any content into chunk")

            chunks.append(ChunkPayload(segments=batch_segments, image_urls=[]))

        # ↓ 以下 image_urls 分支本项目不会走到：generate() 恒传 []，且
        #   _message_builder 忽略图片参数。保留仅为与后端 request_chunker 对齐。
        if not image_urls:
            return chunks

        if not chunks:
            chunks = [ChunkPayload(segments=[], image_urls=[])]

        if not segments:
            for image in image_urls:
                appended = False
                for chunk in chunks[-1:]:
                    candidate_images = chunk.image_urls + [image]
                    if self._messages_size(chunk.segments, candidate_images, **kwargs) <= self.max_bytes:
                        chunk.image_urls = candidate_images
                        appended = True
                        break

                if appended:
                    continue

                if self._messages_size([], [image], **kwargs) > self.max_bytes:
                    raise ValueError("single image payload exceeds max_bytes")
                chunks.append(ChunkPayload(segments=[], image_urls=[image]))
            return chunks

        chunk_count = len(chunks)
        total_images = len(image_urls)
        for idx, image in enumerate(image_urls):
            preferred_idx = min(chunk_count - 1, (idx * chunk_count) // total_images)
            placed = False

            for chunk_idx in range(preferred_idx, len(chunks)):
                chunk = chunks[chunk_idx]
                candidate_images = chunk.image_urls + [image]
                if self._messages_size(chunk.segments, candidate_images, **kwargs) <= self.max_bytes:
                    chunk.image_urls = candidate_images
                    placed = True
                    break

            if placed:
                continue

            if self._messages_size([], [image], **kwargs) > self.max_bytes:
                raise ValueError("single image payload exceeds max_bytes")
            chunks.append(ChunkPayload(segments=[], image_urls=[image]))

        return chunks

    def group_texts_by_budget(self, texts: List[str], build_messages: Callable, **kwargs) -> List[List[str]]:
        groups: List[List[str]] = []
        idx = 0
        while idx < len(texts):
            group: List[str] = []
            while idx < len(texts):
                candidate = group + [texts[idx]]
                try:
                    messages = build_messages(candidate, [], **kwargs)
                except TypeError:
                    messages = build_messages(candidate, **kwargs)
                size = self.estimate(messages)
                if size <= self.max_bytes:
                    group = candidate
                    idx += 1
                    continue
                if not group:
                    raise ValueError("single text block exceeds max_bytes")
                break
            groups.append(group)
        return groups


# ---------------------------------------------------------------------------
# LLM 笔记生成器：切块 → 逐块调 OpenAI 兼容接口 → 合并 → 后处理
# ---------------------------------------------------------------------------
@dataclass
class NoteConfig:
    """LLM 笔记生成配置（自包含，不依赖项目内模块）。"""
    api_key: str
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    tags: Optional[str] = None
    formats: List[str] = field(default_factory=list)
    style: Optional[str] = None
    extras: Optional[str] = None


class NoteGenerator:
    """把 TranscriptResult 生成结构化 Markdown 笔记。

    对齐后端 UniversalGPT.summarize() 的核心链路；截图 / 断点续跑等后端特有
    功能未移植（本脚本不下载视频，也没有前端配置页）。
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None,
                 model: str = "gpt-4o-mini", max_request_bytes: Optional[int] = None):
        if not api_key or not str(api_key).strip():
            raise ValueError("LLM API Key 未配置")
        from openai import OpenAI
        self.client = OpenAI(api_key=str(api_key).strip(), base_url=base_url)
        self.model = model
        self.max_request_bytes = int(
            # 纯文本请求的合理默认：约对应 128k token 模型的安全上限。
            # 旧默认 45MB 远超任何模型上下文，切块形同虚设，小上下文模型会直接 400。
            max_request_bytes or os.environ.get("OPENAI_MAX_REQUEST_BYTES", str(200 * 1024))
        )
        self._max_retries = max(1, int(os.environ.get("OPENAI_RETRY_ATTEMPTS", "3")))
        self._retry_backoff = float(os.environ.get("OPENAI_RETRY_BACKOFF_SECONDS", "1.5"))

    # -- 时间 / 片段格式化 --
    @staticmethod
    def _format_time(seconds: float) -> str:
        """mm:ss；≥1 小时输出 h:mm:ss。

        旧实现 str(timedelta(s))[2:] 只对 "0:MM:SS" 形状成立，≥1h 会把小时位
        连同冒号一起截掉（1:02:33 → "02:33"），导致长视频笔记时间全部错位。
        """
        total = max(0, int(seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _build_segment_text(self, segments) -> str:
        return "\n".join(
            f"{self._format_time(seg.start)} - {(seg.text or '').strip()}"
            for seg in segments
        )

    # -- 消息构造 --
    def _message_builder(self, segments, image_urls, **kwargs):
        return self._create_messages(segments, **kwargs)

    def _create_messages(self, segments, *, title, tags, formats, style, extras) -> list:
        content_text = generate_base_prompt(
            title=title,
            segment_text=self._build_segment_text(segments),
            tags=tags,
            formats=formats,
            style=style,
            extras=extras,
        )
        return [{"role": "user", "content": content_text}]

    def _build_merge_messages(self, texts, image_urls=None) -> list:
        merge_text = MERGE_PROMPT + "\n\n" + "\n\n---\n\n".join(texts)
        return [{"role": "user", "content": merge_text}]

    @staticmethod
    def _estimate_bytes(messages) -> int:
        return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))

    # -- LLM 调用（含重试 / temperature 兜底） --
    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        raw = str(exc).lower()
        tokens = (
            "error code: 524", "bad_response_status_code", "timed out", "timeout",
            "rate limit", "error code: 429", "error code: 500", "error code: 502",
            "error code: 503", "error code: 504", "apiconnectionerror",
            "connection error", "service unavailable",
        )
        if any(t in raw for t in tokens):
            return True
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return status in {408, 409, 429, 500, 502, 503, 504, 524}

    @staticmethod
    def _is_temperature_unsupported(exc: Exception) -> bool:
        raw = str(exc).lower()
        return "temperature" in raw and (
            "does not support" in raw or "unsupported_value" in raw or "only the default" in raw
        )

    def _do_create(self, messages):
        try:
            return self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7,
            )
        except Exception as exc:
            if self._is_temperature_unsupported(exc):
                logger.warning("模型 %s 不支持自定义 temperature，改用默认值重试", self.model)
                return self.client.chat.completions.create(model=self.model, messages=messages)
            raise

    def _chat(self, messages) -> str:
        last_exc = None
        for attempt in range(self._max_retries):
            try:
                resp = self._do_create(messages)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_exc = exc
                if attempt == self._max_retries - 1 or not self._is_retryable(exc):
                    raise
                time.sleep(self._retry_backoff * (2 ** attempt))
        raise last_exc or RuntimeError("chat completion failed")

    def _merge_partials(self, partials: List[str]) -> str:
        if len(partials) == 1:
            return partials[0]
        merge_chunker = RequestChunker(
            self._build_merge_messages, self.max_request_bytes, self._estimate_bytes)
        current = list(partials)
        while len(current) > 1:
            groups = merge_chunker.group_texts_by_budget(current, self._build_merge_messages)
            current = [self._chat(self._build_merge_messages(g)) for g in groups]
        return current[0]

    # -- 主入口 --
    def generate(self, transcript: TranscriptResult, *, title: str,
                 tags: Optional[str] = None, formats: Optional[List[str]] = None,
                 style: Optional[str] = None, extras: Optional[str] = None,
                 video_id: Optional[str] = None, platform: str = "bilibili",
                 source_url: Optional[str] = None) -> str:
        """把转写结果生成结构化 Markdown 笔记。"""
        segments = list(transcript.segments or [])
        if not segments:
            raise ValueError("没有可生成笔记的转写内容")

        chunker = RequestChunker(self._message_builder, self.max_request_bytes, self._estimate_bytes)
        # 单段过大时装不进请求时，逐步放大预算重试（原实现两次调用参数完全相同，
        # 是死代码；这里改成真正的递增重试，异常段才最终抛出）
        chunks = None
        for budget in (self.max_request_bytes, self.max_request_bytes * 2, self.max_request_bytes * 4):
            chunker.max_request_bytes = budget
            try:
                chunks = chunker.chunk(segments, [], title=title, tags=tags,
                                       formats=formats or [], style=style, extras=extras)
                break
            except ValueError:
                logger.warning("切块失败(预算 %d 字节)，尝试放大预算", budget)
        if chunks is None:
            raise ValueError("转写内容过大，即使放大预算也无法切块")

        partials = []
        for chunk in chunks:
            messages = self._create_messages(
                chunk.segments, title=title, tags=tags,
                formats=formats or [], style=style, extras=extras)
            partials.append(self._chat(messages))

        markdown = self._merge_partials(partials)
        if video_id:
            markdown = replace_content_markers(markdown, video_id=video_id, platform=platform)
        if source_url:
            markdown = prepend_source_link(markdown, source_url)
        return markdown.strip()


# ---------------------------------------------------------------------------
# 下载器
# ---------------------------------------------------------------------------
class BilibiliDownloader:
    def __init__(self, cookie: Optional[str] = None, cookie_file: Optional[str] = None):
        self._cookie = cookie or ""
        self._cookie_file = cookie_file
        if self._cookie_file is None:
            tmp = write_cookie_file(self._cookie)
            if tmp:
                self._cookie_file = tmp
                # 临时文件含登录态：对象回收或进程退出时自动删除，不残留 /tmp
                weakref.finalize(self, _remove_quiet, tmp)

    def download(self, video_url: str, output_dir: str = None, quality: str = "fast") -> AudioDownloadResult:
        """下载单集音频（yt-dlp，天然支持 ?p=N）。"""
        # 幂等补丁：库方式调用与 CLI 走同一条路径，都需要 dm_img 参数过风控
        apply_bilibili_dm_img_patch()
        if output_dir is None:
            output_dir = os.environ.get("BILIBILI_OUTPUT", ".")
        os.makedirs(output_dir, exist_ok=True)

        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
            "http_headers": {"Referer": "https://www.bilibili.com"},
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                 "preferredquality": QUALITY_BITRATES.get(quality, QUALITY_BITRATES["fast"])}
            ],
            "noplaylist": True,
            "quiet": False,
        }
        # yt-dlp 对 B 站分 P 合集会无视 noplaylist 展开全部集数（实测 2026.08），
        # 带 ?p=N 时必须用 playlist_items 限定到目标集；单集结果会忽略该参数
        m = re.search(r"[?&]p=(\d+)", video_url)
        if m:
            ydl_opts["playlist_items"] = m.group(1)
        if self._cookie_file:
            ydl_opts["cookiefile"] = self._cookie_file

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            entries = info.get("entries")
            if entries:
                # playlist_items 选中一集时返回单条目播放列表，取真正的条目
                info = dict(entries[0])
            elif entries is not None:
                # 播放列表但没有任何条目：?p=N 超出范围，避免返回指向不存在文件的假结果
                raise ValueError(f"目标分 P 不存在或无可用条目: {video_url}")
            video_id = info.get("id")
            audio_path = os.path.join(output_dir, f"{video_id}.mp3")
        return AudioDownloadResult(
            file_path=audio_path,
            title=info.get("title") or "",
            duration=info.get("duration", 0),
            cover_url=info.get("thumbnail"),
            platform="bilibili",
            video_id=video_id,
            raw_info=info,
        )

    # -- 分 P 编排 -----------------------------------------------------------
    def download_all_parts(
        self,
        video_url: str,
        output_dir: str = None,
        quality: str = "fast",
        transcribe: Optional[str] = None,
        cookie: Optional[str] = None,
        note_config: Optional[NoteConfig] = None,
        per_course_dir: bool = True,
        workers: int = 1,
    ) -> List[BilibiliPartResult]:
        """下载全部分集：每集音频必下，有字幕才落盘 .srt。

        :param transcribe: 无字幕分集的音频转写引擎，"bcut"（免费，需 B 站 cookie）
                           或 None（默认，不转写）。
        :param cookie: B 站登录态 cookie 串；缺省回落到构造器传入的 cookie，
                       避免字幕/音频两侧登录态不一致。
        :param note_config: LLM 笔记生成配置（None 表示不生成笔记）。
        :param per_course_dir: True（默认）时落盘到 output_dir/{bvid}/ 子目录，
                               与 funasr 批量任务、semantic_qa 索引的扫描约定一致。
        :param workers: 并行线程数（默认 1 串行）。每集流水线相互独立，可安全并行；
                        建议 2-4，过大会增加 B 站风控压力。
        """
        if output_dir is None:
            output_dir = os.environ.get("BILIBILI_OUTPUT", ".")
        os.makedirs(output_dir, exist_ok=True)

        bvid = extract_bvid(video_url)
        if not bvid:
            raise ValueError(f"无法从链接提取 B 站 BV id: {video_url}")

        # cookie 未显式传入时回落到构造器值（库调用不再静默分裂两侧登录态）
        if cookie is None:
            cookie = self._cookie

        # 提前创建并校验 LLM 配置：避免下完几百集才发现 api_key 无效
        note_generator = None
        if note_config is not None:
            note_generator = NoteGenerator(
                api_key=note_config.api_key,
                base_url=note_config.base_url,
                model=note_config.model,
            )

        if per_course_dir:
            output_dir = os.path.join(output_dir, bvid)
            os.makedirs(output_dir, exist_ok=True)

        fetcher = BilibiliSubtitleFetcher(cookie=cookie)
        pages = fetcher.list_pages(bvid)
        if not pages:
            logger.warning("%s 未获取到分 P 列表，按单集处理", bvid)
            pages = [{"cid": None, "part": "", "title": ""}]

        total = len(pages)
        workers = max(1, min(int(workers or 1), total))
        logger.info("开始下载: %s 共 %d 集 (workers=%d)", bvid, total, workers)
        results_by_p: dict = {}

        def process_part(p: int, page: dict) -> BilibiliPartResult:
            """单集完整流水线：字幕 → 音频 → (转写) → (笔记)。

            线程安全：每次调用各自创建 YoutubeDL / BCUT 会话；共享的 NoteGenerator
            只持有 OpenAI 客户端（httpx 线程安全）与不可变配置，generate() 内部状态均为局部变量。
            """
            cid = page.get("cid")
            part_title = (page.get("part") or page.get("title") or f"P{p}").strip()
            part_url = f"https://www.bilibili.com/video/{bvid}?p={p}"
            logger.info("===== [%d/%d] %s (cid=%s) =====", p, total, part_title, cid)

            transcript = None
            subtitle_path = None
            try:
                # cid 已在 pages 里，直接按 cid 拉取，省掉每集一次 view API（N+1 问题）
                if cid:
                    transcript = fetcher.fetch_subtitles_for_cid(bvid, cid)
                if transcript and transcript.segments:
                    subtitle_path = self._write_subtitle_file(bvid, p, transcript, output_dir)
                    logger.info("  [p%d] 字幕: %s (%d 段)", p, subtitle_path, len(transcript.segments))
                else:
                    logger.info("  [p%d] 该集无字幕，跳过", p)
            except Exception as exc:
                logger.warning("  [p%d] 该集字幕异常，跳过: %s", p, exc)

            part_errors: List[str] = []
            try:
                audio = self.download(part_url, output_dir=output_dir, quality=quality)
            except Exception as exc:
                # 记录失败并继续：批量场景下一集网络抖动不应废弃整批
                logger.error("  [p%d] 该集音频下载失败，跳过: %s", p, exc)
                return BilibiliPartResult(
                    p=p, bvid=bvid, cid=cid, title=part_title, audio=None,
                    error=f"音频下载失败: {exc}",
                )

            # 无字幕且指定转写引擎 → 用音频跑 ASR，把结果当字幕用
            if transcript is None and transcribe and audio.file_path:
                try:
                    if transcribe == "bcut":
                        transcript = transcribe_bcut(audio.file_path, cookie=cookie)
                    else:
                        logger.warning("  [p%d] 未知转写引擎: %s", p, transcribe)
                    if transcript and transcript.segments:
                        subtitle_path = self._write_subtitle_file(bvid, p, transcript, output_dir)
                        logger.info("  [p%d] 转写: %s (%d 段)", p, subtitle_path, len(transcript.segments))
                except Exception as exc:
                    part_errors.append(f"转写失败: {exc}")
                    logger.warning("  [p%d] 该集转写失败，跳过: %s", p, exc)

            # 有转写结果且配置了 LLM → 生成 Markdown 笔记
            note_path = None
            if note_generator is not None and transcript and transcript.segments:
                try:
                    note_md = note_generator.generate(
                        transcript,
                        title=part_title,
                        tags=note_config.tags,
                        formats=note_config.formats,
                        style=note_config.style,
                        extras=note_config.extras,
                        video_id=f"{bvid}_p{p}",
                        platform="bilibili",
                        source_url=f"https://www.bilibili.com/video/{bvid}?p={p}",
                    )
                    note_path = os.path.join(output_dir, f"{bvid}_p{p}.md")
                    with open(note_path, "w", encoding="utf-8") as f:
                        f.write(note_md)
                    logger.info("  [p%d] 笔记: %s", p, note_path)
                except Exception as exc:
                    part_errors.append(f"笔记生成失败: {exc}")
                    logger.warning("  [p%d] 该集笔记生成失败，跳过: %s", p, exc)

            return BilibiliPartResult(
                p=p, bvid=bvid, cid=cid, title=part_title,
                audio=audio, transcript=transcript, subtitle_path=subtitle_path,
                note_path=note_path,
                error="; ".join(part_errors) or None,
            )

        if workers == 1:
            for p, page in enumerate(pages, 1):
                results_by_p[p] = process_part(p, page)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process_part, p, page): p
                           for p, page in enumerate(pages, 1)}
                for fut in as_completed(futures):
                    p = futures[fut]
                    try:
                        results_by_p[p] = fut.result()
                    except Exception as exc:  # process_part 内部已兜底，此处防御意外异常
                        logger.error("  [p%d] 内部错误: %s", p, exc)
                        results_by_p[p] = BilibiliPartResult(
                            p=p, bvid=bvid, cid=pages[p - 1].get("cid"),
                            title=f"P{p}", audio=None, error=f"内部错误: {exc}")

        results = [results_by_p[p] for p in range(1, total + 1)]
        logger.info("全部完成: %s 共 %d 集", bvid, total)
        return results

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        """兼容别名：模块级 fmt_ts 的转发（旧调用方可能用到）。"""
        return fmt_ts(seconds)

    def _write_subtitle_file(self, bvid, p, transcript, output_dir) -> str:
        """按 {bvid}_p{p}.{lang}.srt 命名落盘字幕（内容拼装统一走模块级 write_srt）。"""
        lang = (transcript.language or "zh").replace("-", "_").lower()
        path = os.path.join(output_dir, f"{bvid}_p{p}.{lang}.srt")
        return write_srt(transcript.segments, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_env_file(path: Optional[str] = None) -> None:
    """极简 .env 加载：KEY=VALUE，支持注释/空行/export 前缀/成对引号。

    只补缺，不覆盖已有环境变量。使 README「复制 .env.example 为 .env 即可」
    真正生效，无需引入 python-dotenv 依赖。
    """
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError as e:
        logger.warning("读取 %s 失败: %s", path, e)


def main(argv=None):
    # .env 必须在参数默认值求值前加载（argparse 的 default= 在此刻读环境变量）
    _load_env_file()
    parser = argparse.ArgumentParser(description="B 站分 P 视频全部分集下载（音频+字幕）")
    parser.add_argument("url", nargs="?", default=None,
                        help="B 站视频链接（--search 搜索时可省略）")
    parser.add_argument("--output", "-o", default=os.environ.get("BILIBILI_OUTPUT", "./bilibili_out"),
                        help="输出目录（默认 ./bilibili_out）")
    parser.add_argument("--quality", "-q", default="fast", choices=["fast", "medium", "slow"],
                        help="音频质量: fast=64 / medium=128 / slow=320 kbps mp3（默认 fast）")
    parser.add_argument("--cookie", default=os.environ.get("BILIBILI_COOKIE"),
                        help="B 站原始 cookie 串（SESSDATA=...; ...）")
    parser.add_argument("--cookie-file", default=os.environ.get("BILIBILI_COOKIE_FILE"),
                        help="Netscape 格式 cookie 文件路径（拿 AI 字幕必需登录态）")
    parser.add_argument("--manifest", default=None, help="额外把结果写成 JSON manifest")
    parser.add_argument("--transcribe", default="none", choices=["none", "bcut"],
                        help="无字幕分集的音频转写引擎（默认 none；bcut 免费但需 B 站 cookie）")
    parser.add_argument("--note", action="store_true",
                        help="生成 LLM Markdown 笔记（转写→切块→LLM→笔记）")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"),
                        help="LLM API Key（或环境变量 LLM_API_KEY；--note 时必填）")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"),
                        help="LLM 接口 Base URL（OpenAI 兼容；不填则用 openai 默认地址）")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                        help="LLM 模型名（默认 gpt-4o-mini）")
    parser.add_argument("--style", default=None, choices=list(NOTE_STYLES.keys()),
                        help="笔记风格：" + "、".join(NOTE_STYLES.keys()))
    parser.add_argument("--format", "--formats", dest="formats", default=None,
                        help="笔记格式，逗号分隔：" + "、".join(NOTE_FORMATS.keys()))
    parser.add_argument("--tags", default=None, help="传给 LLM 的视频标签")
    parser.add_argument("--extras", default=None, help="传给 LLM 的额外指令")
    parser.add_argument("--search", default=None,
                        help="只搜索不下载：B 站关键词搜索，输出 bv/标题/UP主/时长")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="并行下载线程数（默认 1 串行；建议 2-4，过大易触发风控）")
    parser.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 组装 LLM 笔记生成配置（--note 时校验必填项）
    note_config = None
    if args.note:
        if not args.api_key:
            parser.error("--note 需要 --api-key（或环境变量 LLM_API_KEY）")
        fmt_list = [s.strip() for s in (args.formats or "").split(",") if s.strip()]
        invalid = [f for f in fmt_list if f not in NOTE_FORMATS]
        if invalid:
            parser.error(f"不支持的笔记格式: {invalid}，可选: {list(NOTE_FORMATS)}")
        note_config = NoteConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            tags=args.tags,
            formats=fmt_list,
            style=args.style,
            extras=args.extras,
        )

    # 纯搜索模式
    if args.search:
        try:
            results = search_bilibili(args.search, cookie=args.cookie, cookie_file=args.cookie_file)
        except Exception as exc:
            logger.error("搜索失败: %s", exc)
            if args.verbose:
                raise
            return 1
        if not results:
            print(f"搜索「{args.search}」无结果")
            return 1
        print(f"\n===== 搜索结果: {args.search} ({len(results)} 条) =====")
        for r in results:
            print(f"  {r.bvid} | {r.title[:50]} | {r.uploader} | {r.duration:.0f}s")
            print(f"     {r.url}")
        return 0

    if not args.url:
        parser.error("需要提供 B 站视频链接，或使用 --search 搜索")

    dl = BilibiliDownloader(cookie=args.cookie, cookie_file=args.cookie_file)
    try:
        results = dl.download_all_parts(
            args.url, output_dir=args.output, quality=args.quality,
            transcribe=None if args.transcribe == "none" else args.transcribe,
            cookie=args.cookie,
            note_config=note_config,
            workers=args.workers,
        )
    except KeyboardInterrupt:
        logger.error("用户中断")
        return 130
    except Exception as exc:
        logger.error("下载失败: %s", exc)
        if args.verbose:
            raise
        return 1

    print("\n===== 下载清单 =====")
    for r in results:
        print(f"p={r.p} {r.title!r}")
        if r.audio:
            print(f"  音频: {r.audio.file_path} ({r.audio.duration:.1f}s)")
        else:
            print(f"  音频: 失败（{r.error}）")
        print(f"  字幕: {r.subtitle_path or '（无）'}")
        print(f"  笔记: {r.note_path or '（无）'}")
    n_fail = sum(1 for r in results if r.audio is None)
    fail_hint = f"，失败 {n_fail} 集" if n_fail else ""
    out_dir = os.path.abspath(args.output)
    if results and results[0].bvid:  # 默认按课程分子目录
        out_dir = os.path.join(out_dir, results[0].bvid)
    print(f"\n共 {len(results)} 集{fail_hint}，输出目录: {out_dir}")

    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            # asdict(r) 会递归把 audio/transcript 等 dataclass 字段也转成 dict，
            # 直接对整个列表序列化即可；default=str 兜底 yt-dlp raw_info 里的非 JSON 对象。
            json.dump([asdict(r) for r in results], f, ensure_ascii=False,
                      indent=2, default=str)
        print(f"manifest 已写入: {args.manifest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
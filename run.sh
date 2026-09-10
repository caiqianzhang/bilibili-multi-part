#!/usr/bin/env bash
# 用法: ./run.sh "https://www.bilibili.com/video/BVxxx" [额外参数]
#
# 国内网络装依赖时 pip 需指定阿里云镜像（见下方 PIP_INDEX_URL）。
set -euo pipefail
cd "$(dirname "$0")"

# 国内镜像：部分公共镜像（tuna/douban）在某些环境会返回空索引，阿里云最稳。
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"

if [ ! -d .venv ]; then
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -q -r requirements.txt
else
    . .venv/bin/activate
fi

exec python bilibili_multi_part.py "$@"
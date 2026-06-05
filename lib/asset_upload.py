"""共享的本地资源上传辅助。

当前仅暴露 `upload_to_tmpfiles` —— 把本地文件传到 tmpfiles.org 拿到可被外部 API
访问的 https 直链，用于把"只接受 URL 的下游"（如 agnes-image / agnes-video）
的入参要求转成可被我们的本地素材满足的形式。

调用方自行决定上传哪些文件、限多少张、是否跳过缺失；本模块只关心"传一个文件，
换回一个直链"。
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/crud"
_TMPFILES_HOST = "tmpfiles.org/"
_TMPFILES_DL_HOST = "tmpfiles.org/dl/"
_UPLOAD_TIMEOUT_SECONDS = 30.0


async def upload_to_tmpfiles(file_path: Path) -> str:
    """上传本地文件到 tmpfiles.org，返回可下载的直链 URL。

    抛出：
        FileNotFoundError: 文件不存在（让调用方决定是否跳过）
        RuntimeError: 上传失败（非 200 / 响应结构异常）
    """
    if not file_path.exists():
        raise FileNotFoundError(f"待上传文件不存在: {file_path}")

    async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT_SECONDS) as client:
        with open(file_path, "rb") as f:
            response = await client.post(
                _TMPFILES_UPLOAD_URL,
                files={"file": (file_path.name, f, "image/png")},
            )
    if response.status_code != 200:
        raise RuntimeError(f"tmpfiles.org upload failed: HTTP {response.status_code}")

    try:
        url = response.json()["data"]["url"]
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"tmpfiles.org 响应解析失败: {e}") from e

    if "/dl/" not in url:
        url = url.replace(_TMPFILES_HOST, _TMPFILES_DL_HOST, 1)
    logger.debug("tmpfiles.org 上传完成: %s -> %s", file_path.name, url)
    return url

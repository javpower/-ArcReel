"""asset_upload.upload_to_tmpfiles 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.asset_upload import upload_to_tmpfiles


def _make_response(status_code: int, body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    if body is not None:
        resp.json.return_value = body
    else:
        resp.json.side_effect = json.JSONDecodeError("no json", "", 0)
    return resp


class TestUploadToTmpfiles:
    async def test_success_returns_dl_url(self, tmp_path: Path):
        f = tmp_path / "img.png"
        f.write_bytes(b"fake-png-bytes")

        upload_resp = _make_response(
            200,
            {"data": {"url": "https://tmpfiles.org/dl/12345/img.png"}},
        )

        with patch("lib.asset_upload.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.post = AsyncMock(return_value=upload_resp)
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = client

            url = await upload_to_tmpfiles(f)

        assert url == "https://tmpfiles.org/dl/12345/img.png"
        client.post.assert_awaited_once()
        called_url = client.post.await_args[0][0]
        assert called_url == "https://tmpfiles.org/api/v1/crud"
        # files kwarg 包含 (filename, file_obj, mime)
        files_kw = client.post.await_args[1]["files"]
        assert files_kw["file"][0] == "img.png"
        assert files_kw["file"][2] == "image/png"

    async def test_converts_host_to_dl(self, tmp_path: Path):
        """响应给的是 tmpfiles.org/... 形式（无 dl），应自动加 dl。"""
        f = tmp_path / "img.png"
        f.write_bytes(b"x")

        upload_resp = _make_response(
            200,
            {"data": {"url": "https://tmpfiles.org/12345/img.png"}},
        )

        with patch("lib.asset_upload.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.post = AsyncMock(return_value=upload_resp)
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = client

            url = await upload_to_tmpfiles(f)

        assert "tmpfiles.org/dl/" in url

    async def test_http_error_raises(self, tmp_path: Path):
        f = tmp_path / "img.png"
        f.write_bytes(b"x")
        upload_resp = _make_response(500)

        with patch("lib.asset_upload.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.post = AsyncMock(return_value=upload_resp)
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = client

            with pytest.raises(RuntimeError, match="HTTP 500"):
                await upload_to_tmpfiles(f)

    async def test_malformed_response_raises(self, tmp_path: Path):
        f = tmp_path / "img.png"
        f.write_bytes(b"x")
        upload_resp = _make_response(200, {"unexpected": "shape"})

        with patch("lib.asset_upload.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.post = AsyncMock(return_value=upload_resp)
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = client

            with pytest.raises(RuntimeError, match="响应解析失败"):
                await upload_to_tmpfiles(f)

    async def test_missing_file_raises(self, tmp_path: Path):
        ghost = tmp_path / "nope.png"
        with pytest.raises(FileNotFoundError):
            await upload_to_tmpfiles(ghost)

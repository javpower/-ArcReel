"""OpenAIVideoBackend Agnes Video v2.0 路径单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from lib.providers import PROVIDER_OPENAI
from lib.video_backends.base import VideoGenerationRequest

# ----- helpers -----

DUMMY_URL_1 = "https://tmpfiles.org/dl/aaa/img1.png"
DUMMY_URL_2 = "https://tmpfiles.org/dl/bbb/img2.png"
DUMMY_VIDEO_URL = "https://storage.googleapis.com/agnes-aigc/aigc/videos/2026/06/03/video_xyz.mp4"
DUMMY_TASK_ID = "vid_agnes_1"


def _patch_upload(monkeypatch, urls: list[str] | None = None) -> list[str]:
    """把 upload_to_tmpfiles 替成顺序返回 urls 的桩；返回实际用的列表。"""
    queue = list(urls or [DUMMY_URL_1, DUMMY_URL_2])

    async def _stub(_path):
        if not queue:
            raise AssertionError("upload_to_tmpfiles 被调用次数超出预期")
        return queue.pop(0)

    monkeypatch.setattr("lib.video_backends.openai.upload_to_tmpfiles", _stub)
    return queue  # 调试用，可读剩余期望调用


def _make_agnes_video(status="completed", video_id=DUMMY_TASK_ID, seconds="10", video_url=DUMMY_VIDEO_URL):
    v = MagicMock()
    v.id = video_id
    v.status = status
    v.seconds = seconds
    v.video_url = video_url
    v.error = None
    return v


def _stub_agnes_client(client: AsyncMock, *, final_video=None) -> None:
    """仅桩轮询用到的 videos.retrieve；Agnes 创建任务不再走 SDK 的 videos.create。"""
    client.videos.retrieve = AsyncMock(return_value=final_video or _make_agnes_video())


def _stub_agnes_create(monkeypatch, task_id: str = DUMMY_TASK_ID) -> list[dict]:
    """把 _create_agnes_task 替成直接返回 task_id 的桩；返回捕获的 payload 列表。"""
    captured: list[dict] = []

    async def _stub(_self, payload: dict) -> str:
        captured.append(payload)
        return task_id

    # 用 import 后的类引用来 patch
    from lib.video_backends.openai import OpenAIVideoBackend

    monkeypatch.setattr(OpenAIVideoBackend, "_create_agnes_task", _stub)
    return captured


# ----- 单元：_is_agnes_video -----


class TestIsAgnesVideo:
    def test_agnes_model_returns_true(self):
        with patch("lib.openai_shared.AsyncOpenAI"):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="agnes-video-v2.0")
            assert backend._is_agnes_video() is True

    def test_sora_model_returns_false(self):
        with patch("lib.openai_shared.AsyncOpenAI"):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="sora-2")
            assert backend._is_agnes_video() is False

    def test_model_name_is_case_insensitive(self):
        with patch("lib.openai_shared.AsyncOpenAI"):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="Agnes-Video-V2.0")
            assert backend._is_agnes_video() is True


# ----- 单元：_clamp_to_8n1 + _resolve_agnes_video_params -----


class TestAgnesParamHelpers:
    def test_clamp_to_8n1_below_minimum(self):
        from lib.video_backends.openai import _clamp_to_8n1

        for n in [0, 1, 4, 8]:
            assert _clamp_to_8n1(n) == 9

    def test_clamp_to_8n1_8n_plus_1_forms(self):
        from lib.video_backends.openai import _clamp_to_8n1

        for n in [9, 17, 81, 121, 161, 241, 441]:
            assert _clamp_to_8n1(n) == n

    def test_clamp_to_8n1_rounds_down(self):
        from lib.video_backends.openai import _clamp_to_8n1

        assert _clamp_to_8n1(120) == 113  # 113 = 8*14+1
        assert _clamp_to_8n1(125) == 121
        assert _clamp_to_8n1(150) == 145  # 145 = 8*18+1

    def test_clamp_to_8n1_above_max(self):
        from lib.video_backends.openai import _clamp_to_8n1

        assert _clamp_to_8n1(442) == 441
        assert _clamp_to_8n1(9999) == 441

    def test_resolve_params_default_9_16(self):
        from lib.video_backends.openai import _resolve_agnes_video_params

        p = _resolve_agnes_video_params(duration_seconds=5, aspect_ratio="9:16", resolution=None)
        assert p["width"] == 768
        assert p["height"] == 1152
        assert p["frame_rate"] == 24
        # 5 * 24 = 120, clamp → 113
        assert p["num_frames"] == 113

    def test_resolve_params_16_9_default(self):
        from lib.video_backends.openai import _resolve_agnes_video_params

        p = _resolve_agnes_video_params(duration_seconds=5, aspect_ratio="16:9", resolution=None)
        assert p["width"] == 1152
        assert p["height"] == 768

    def test_resolve_params_1080p(self):
        from lib.video_backends.openai import _resolve_agnes_video_params

        p = _resolve_agnes_video_params(duration_seconds=5, aspect_ratio="9:16", resolution="1080p")
        assert p == {"width": 1080, "height": 1920, "num_frames": 113, "frame_rate": 24}

        p2 = _resolve_agnes_video_params(duration_seconds=5, aspect_ratio="16:9", resolution="1080p")
        assert p2 == {"width": 1920, "height": 1080, "num_frames": 113, "frame_rate": 24}

    def test_resolve_params_unknown_aspect_falls_back_to_portrait(self):
        from lib.video_backends.openai import _resolve_agnes_video_params

        p = _resolve_agnes_video_params(duration_seconds=10, aspect_ratio="21:9", resolution=None)
        # 未知 ratio 走默认 (768, 1152)
        assert p["width"] == 768
        assert p["height"] == 1152
        # 10 * 24 = 240, 240 → 233 (8*29+1)
        assert p["num_frames"] == 233


# ----- 集成：_generate_agnes_video 完整流程 -----


class TestGenerateAgnesVideo:
    async def test_text_to_video_no_image(self, monkeypatch, tmp_path: Path):
        _patch_upload(monkeypatch, urls=[])  # 不会被调用
        mock_client = AsyncMock()
        _stub_agnes_client(mock_client)
        create_payloads = _stub_agnes_create(monkeypatch)

        with (
            patch("lib.openai_shared.AsyncOpenAI", return_value=mock_client),
            patch("lib.video_backends.base.asyncio.sleep", new_callable=AsyncMock),
        ):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="agnes-video-v2.0")
            output_path = tmp_path / "out.mp4"
            request = VideoGenerationRequest(
                prompt="A cat walking",
                output_path=output_path,
                duration_seconds=5,
                aspect_ratio="9:16",
            )

            async def _fake_download(self, url, dst):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(b"FAKE-VIDEO-BYTES")

            with patch.object(OpenAIVideoBackend, "_download_agnes_video_from_url", _fake_download):
                result = await backend.generate(request)

        assert result.provider == PROVIDER_OPENAI
        assert result.model == "agnes-video-v2.0"
        assert result.duration_seconds == 10
        assert result.task_id == DUMMY_TASK_ID
        assert output_path.read_bytes() == b"FAKE-VIDEO-BYTES"

        # payload 是扁平 JSON，不走 SDK 的 extra_body 包装
        payload = create_payloads[0]
        assert payload["model"] == "agnes-video-v2.0"
        assert payload["prompt"] == "A cat walking"
        assert payload["width"] == 768
        assert payload["height"] == 1152
        assert payload["frame_rate"] == 24
        assert payload["num_frames"] == 113
        assert "image" not in payload

    async def test_image_to_video_single_url(self, monkeypatch, tmp_path: Path):
        start = tmp_path / "start.png"
        start.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        _patch_upload(monkeypatch, urls=[DUMMY_URL_1])
        mock_client = AsyncMock()
        _stub_agnes_client(mock_client)
        create_payloads = _stub_agnes_create(monkeypatch)

        with (
            patch("lib.openai_shared.AsyncOpenAI", return_value=mock_client),
            patch("lib.video_backends.base.asyncio.sleep", new_callable=AsyncMock),
        ):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="agnes-video-v2.0")
            output_path = tmp_path / "out.mp4"
            request = VideoGenerationRequest(
                prompt="Animate this",
                output_path=output_path,
                start_image=start,
                duration_seconds=5,
            )

            async def _fake_download(self, url, dst):
                dst.write_bytes(b"X")

            with patch.object(OpenAIVideoBackend, "_download_agnes_video_from_url", _fake_download):
                await backend.generate(request)

        payload = create_payloads[0]
        # 单图 → payload["image"] 字符串（扁平 JSON，无 extra_body 包装）
        assert payload["image"] == DUMMY_URL_1
        assert payload["width"] == 768
        assert payload["height"] == 1152
        assert payload["frame_rate"] == 24
        assert payload["num_frames"] == 113
        assert "extra_body" not in payload

    async def test_multi_image_to_video(self, monkeypatch, tmp_path: Path):
        start = tmp_path / "start.png"
        start.write_bytes(b"x")
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"y")

        _patch_upload(monkeypatch, urls=[DUMMY_URL_1, DUMMY_URL_2])
        mock_client = AsyncMock()
        _stub_agnes_client(mock_client)
        create_payloads = _stub_agnes_create(monkeypatch)

        with (
            patch("lib.openai_shared.AsyncOpenAI", return_value=mock_client),
            patch("lib.video_backends.base.asyncio.sleep", new_callable=AsyncMock),
        ):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="agnes-video-v2.0")
            output_path = tmp_path / "out.mp4"
            request = VideoGenerationRequest(
                prompt="Blend",
                output_path=output_path,
                start_image=start,
                reference_images=[ref],
                duration_seconds=3,
            )

            async def _fake_download(self, url, dst):
                dst.write_bytes(b"X")

            with patch.object(OpenAIVideoBackend, "_download_agnes_video_from_url", _fake_download):
                await backend.generate(request)

        payload = create_payloads[0]
        # 多图 → extra_body.image 数组（API 文档要求多图放在 extra_body 内）
        assert payload["extra_body"] == {"image": [DUMMY_URL_1, DUMMY_URL_2]}
        assert "mode" not in payload
        assert "image" not in payload

    async def test_keyframes_mode(self, monkeypatch, tmp_path: Path):
        start = tmp_path / "start.png"
        start.write_bytes(b"x")
        end = tmp_path / "end.png"
        end.write_bytes(b"y")

        _patch_upload(monkeypatch, urls=[DUMMY_URL_1, DUMMY_URL_2])
        mock_client = AsyncMock()
        _stub_agnes_client(mock_client)
        create_payloads = _stub_agnes_create(monkeypatch)

        with (
            patch("lib.openai_shared.AsyncOpenAI", return_value=mock_client),
            patch("lib.video_backends.base.asyncio.sleep", new_callable=AsyncMock),
        ):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="agnes-video-v2.0")
            output_path = tmp_path / "out.mp4"
            request = VideoGenerationRequest(
                prompt="Transition",
                output_path=output_path,
                start_image=start,
                end_image=end,
                duration_seconds=5,
            )

            async def _fake_download(self, url, dst):
                dst.write_bytes(b"X")

            with patch.object(OpenAIVideoBackend, "_download_agnes_video_from_url", _fake_download):
                await backend.generate(request)

        payload = create_payloads[0]
        # 关键帧 → extra_body.image 数组 + extra_body.mode
        assert payload["extra_body"] == {
            "image": [DUMMY_URL_1, DUMMY_URL_2],
            "mode": "keyframes",
        }
        assert "image" not in payload
        assert "mode" not in payload

    async def test_missing_image_upload_url_raises(self, monkeypatch, tmp_path: Path):
        # i2i 请求但没有可用图片 → upload_to_tmpfiles 不会被调用，但请求里没有 image
        # 现有 Sora 路径会抛错（"all reference images failed to open" 等），
        # agnes 路径会构造出没有 image 的请求并最终成功（视作 T2V）。
        # 这里只断言：当没有任何 image 路径时，确实走的是 T2V（无 image 字段）。
        _patch_upload(monkeypatch, urls=[])
        mock_client = AsyncMock()
        _stub_agnes_client(mock_client)
        create_payloads = _stub_agnes_create(monkeypatch)

        with (
            patch("lib.openai_shared.AsyncOpenAI", return_value=mock_client),
            patch("lib.video_backends.base.asyncio.sleep", new_callable=AsyncMock),
        ):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="agnes-video-v2.0")
            output_path = tmp_path / "out.mp4"
            request = VideoGenerationRequest(
                prompt="T2V fallback",
                output_path=output_path,
                start_image=tmp_path / "ghost.png",  # 不存在
                duration_seconds=3,
            )

            async def _fake_download(self, url, dst):
                dst.write_bytes(b"X")

            with patch.object(OpenAIVideoBackend, "_download_agnes_video_from_url", _fake_download):
                await backend.generate(request)

        payload = create_payloads[0]
        # 没有 image 路径（仅走 start_image 缺失）→ 当作纯 T2V
        assert "image" not in payload
        assert payload["width"] == 768
        assert payload["height"] == 1152
        assert payload["frame_rate"] == 24
        assert payload["num_frames"] == 65

    async def test_sora_path_unaffected(self, monkeypatch, tmp_path: Path):
        """非 agnes 模型仍走 Sora 路径（seconds + input_reference + download_content）。"""
        mock_client = AsyncMock()
        mock_client.videos.create = AsyncMock(return_value=MagicMock(id="vid_sora", status="queued"))
        mock_client.videos.retrieve = AsyncMock(
            return_value=MagicMock(id="vid_sora", status="completed", seconds="5", error=None)
        )
        content = MagicMock()
        content.content = b"sora-video-bytes"
        mock_client.videos.download_content = AsyncMock(return_value=content)

        start = tmp_path / "start.png"
        start.write_bytes(b"x")

        with (
            patch("lib.openai_shared.AsyncOpenAI", return_value=mock_client),
            patch("lib.video_backends.base.asyncio.sleep", new_callable=AsyncMock),
        ):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="sora-2")
            output_path = tmp_path / "out.mp4"
            request = VideoGenerationRequest(
                prompt="Sora",
                output_path=output_path,
                start_image=start,
                duration_seconds=5,
            )
            result = await backend.generate(request)

        assert result.model == "sora-2"
        # 走 Sora 路径：seconds 字符串，input_reference (file tuple)
        call_kwargs = mock_client.videos.create.call_args[1]
        assert call_kwargs["seconds"] == "5"
        assert isinstance(call_kwargs["input_reference"], tuple)
        # 走 Sora 路径：下载走 videos.download_content
        mock_client.videos.download_content.assert_awaited_once()


# ----- 回归：start_image_url / reference_image_urls 短路上传 -----


class TestAgnesVideoUrlShortCircuit:
    """``start_image_url`` / ``reference_image_urls`` 命中时跳过本地上传，直接喂下游。"""

    @staticmethod
    def _patch_upload_should_not_run(monkeypatch) -> list[str]:
        """把 upload_to_tmpfiles 替成断言式桩：被调用时直接失败。"""
        calls: list[str] = []

        async def _stub(path):
            calls.append(str(path))
            raise AssertionError(f"upload_to_tmpfiles 不应被调用（url 已短路），但收到 path={path}")

        monkeypatch.setattr("lib.video_backends.openai.upload_to_tmpfiles", _stub)
        return calls

    async def test_start_image_url_skips_upload(self, monkeypatch, tmp_path: Path):
        start = tmp_path / "start.png"
        start.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        upload_calls = self._patch_upload_should_not_run(monkeypatch)
        mock_client = AsyncMock()
        _stub_agnes_client(mock_client)
        create_payloads = _stub_agnes_create(monkeypatch)

        with (
            patch("lib.openai_shared.AsyncOpenAI", return_value=mock_client),
            patch("lib.video_backends.base.asyncio.sleep", new_callable=AsyncMock),
        ):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="agnes-video-v2.0")
            output_path = tmp_path / "out.mp4"
            request = VideoGenerationRequest(
                prompt="Animate this",
                output_path=output_path,
                start_image=start,
                start_image_url=DUMMY_URL_1,  # 已存的公网直链
                duration_seconds=5,
            )

            async def _fake_download(self, url, dst):
                dst.write_bytes(b"X")

            with patch.object(OpenAIVideoBackend, "_download_agnes_video_from_url", _fake_download):
                await backend.generate(request)

        # 关键：upload_to_tmpfiles 完全没被调用
        assert upload_calls == []
        payload = create_payloads[0]
        # 单图 → image 直接在扁平 payload 中
        assert payload["image"] == DUMMY_URL_1

    async def test_reference_image_urls_skip_upload(self, monkeypatch, tmp_path: Path):
        start = tmp_path / "start.png"
        start.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"y")

        upload_calls = self._patch_upload_should_not_run(monkeypatch)
        mock_client = AsyncMock()
        _stub_agnes_client(mock_client)
        create_payloads = _stub_agnes_create(monkeypatch)

        with (
            patch("lib.openai_shared.AsyncOpenAI", return_value=mock_client),
            patch("lib.video_backends.base.asyncio.sleep", new_callable=AsyncMock),
        ):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="agnes-video-v2.0")
            output_path = tmp_path / "out.mp4"
            request = VideoGenerationRequest(
                prompt="Animate this",
                output_path=output_path,
                start_image=start,
                start_image_url=DUMMY_URL_1,
                reference_images=[ref],
                reference_image_urls=[DUMMY_URL_2],
                duration_seconds=5,
            )

            async def _fake_download(self, url, dst):
                dst.write_bytes(b"X")

            with patch.object(OpenAIVideoBackend, "_download_agnes_video_from_url", _fake_download):
                await backend.generate(request)

        assert upload_calls == []
        payload = create_payloads[0]
        # 2 张图（start + reference），走 extra_body.image 列表路径
        assert payload["extra_body"] == {"image": [DUMMY_URL_1, DUMMY_URL_2]}

    async def test_url_missing_for_one_ref_falls_back_to_upload(self, monkeypatch, tmp_path: Path):
        """reference_image_urls 里有 None（占位），对应 path 回退到 upload。"""
        start = tmp_path / "start.png"
        start.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"y")

        # 上传桩：ref 没 url，应该被传进来
        _patch_upload(monkeypatch, urls=[DUMMY_URL_2])
        mock_client = AsyncMock()
        _stub_agnes_client(mock_client)
        create_payloads = _stub_agnes_create(monkeypatch)

        with (
            patch("lib.openai_shared.AsyncOpenAI", return_value=mock_client),
            patch("lib.video_backends.base.asyncio.sleep", new_callable=AsyncMock),
        ):
            from lib.video_backends.openai import OpenAIVideoBackend

            backend = OpenAIVideoBackend(api_key="k", model="agnes-video-v2.0")
            output_path = tmp_path / "out.mp4"
            request = VideoGenerationRequest(
                prompt="Animate this",
                output_path=output_path,
                start_image=start,
                start_image_url=DUMMY_URL_1,
                reference_images=[ref],
                reference_image_urls=[None],  # ref 没有持久化的 url
                duration_seconds=5,
            )

            async def _fake_download(self, url, dst):
                dst.write_bytes(b"X")

            with patch.object(OpenAIVideoBackend, "_download_agnes_video_from_url", _fake_download):
                await backend.generate(request)

        payload = create_payloads[0]
        assert payload["extra_body"] == {"image": [DUMMY_URL_1, DUMMY_URL_2]}

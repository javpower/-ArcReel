"""OpenAIVideoBackend — OpenAI Sora 视频生成后端。"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

from lib.asset_upload import upload_to_tmpfiles
from lib.logging_utils import format_kwargs_for_log
from lib.openai_shared import OPENAI_RETRYABLE_ERRORS, create_openai_client
from lib.providers import PROVIDER_OPENAI
from lib.retry import DOWNLOAD_BACKOFF_SECONDS, DOWNLOAD_MAX_ATTEMPTS, with_retry_async
from lib.video_backends.base import (
    IMAGE_MIME_TYPES,
    VideoCapabilities,
    VideoCapability,
    VideoGenerationRequest,
    VideoGenerationResult,
    poll_with_retry,
)

_POLL_INTERVAL_SECONDS = 5.0
_MIN_POLL_TIMEOUT_SECONDS = 600.0
_POLL_TIMEOUT_PER_SECOND = 30.0

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sora-2"

_SIZE_MAP: dict[tuple[str, str], str] = {
    ("720p", "9:16"): "720x1280",
    ("720p", "16:9"): "1280x720",
    ("1080p", "9:16"): "1080x1920",
    ("1080p", "16:9"): "1920x1080",
    ("1024p", "9:16"): "1024x1792",
    ("1024p", "16:9"): "1792x1024",
}


def _resolve_size(resolution: str | None, aspect_ratio: str) -> str | None:
    """解析 size：None 不传；已知复合 key 映射；未知 → warning 后透传作为 size。"""
    if resolution is None:
        return None
    mapped = _SIZE_MAP.get((resolution, aspect_ratio))
    if mapped is not None:
        return mapped
    logger.warning(
        "OpenAI video: 未知 (resolution=%r, aspect=%r)，原样作为 size 透传",
        resolution,
        aspect_ratio,
    )
    return resolution


# --- Agnes Video v2.0 协议相关常量与解析 ---

AGNES_VIDEO_MODEL = "agnes-video-v2.0"
_AGNES_MAX_NUM_FRAMES = 441
_AGNES_DEFAULT_FRAME_RATE = 24
_AGNES_SIZE_BY_ASPECT: dict[str, tuple[int, int]] = {
    "9:16": (768, 1152),  # portrait: width, height
    "16:9": (1152, 768),  # landscape: width, height
}
_AGNES_SIZE_1080P: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
}
_AGNES_VIDEO_DOWNLOAD_TIMEOUT_SECONDS = 600.0


def _clamp_to_8n1(n: int) -> int:
    """把帧数钳到 [9, 441] 并凑成 8n+1 形式（agnes 协议硬性要求）。"""
    n = max(1, min(_AGNES_MAX_NUM_FRAMES, n))
    if n < 9:
        n = 9
    return ((n - 1) // 8) * 8 + 1


def _resolve_agnes_video_params(
    duration_seconds: int,
    aspect_ratio: str,
    resolution: str | None,
) -> dict[str, int]:
    """把 (duration, ratio, resolution) 翻译成 agnes-video 协议要求的参数子集。"""
    if resolution == "1080p" and aspect_ratio in _AGNES_SIZE_1080P:
        width, height = _AGNES_SIZE_1080P[aspect_ratio]
    else:
        width, height = _AGNES_SIZE_BY_ASPECT.get(aspect_ratio, (768, 1152))
    num_frames = _clamp_to_8n1(duration_seconds * _AGNES_DEFAULT_FRAME_RATE)
    return {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": _AGNES_DEFAULT_FRAME_RATE,
    }


class OpenAIVideoBackend:
    """OpenAI Sora 视频生成后端。"""

    def __init__(self, *, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self._client = create_openai_client(api_key=api_key, base_url=base_url)
        self._api_key = api_key
        self._base_url = base_url
        self._model = model or DEFAULT_MODEL
        self._capabilities: set[VideoCapability] = {
            VideoCapability.TEXT_TO_VIDEO,
            VideoCapability.IMAGE_TO_VIDEO,
        }

    @property
    def name(self) -> str:
        return PROVIDER_OPENAI

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[VideoCapability]:
        return self._capabilities

    @property
    def video_capabilities(self) -> VideoCapabilities:
        if self._is_agnes_video():
            # Agnes 支持 keyframes 首尾帧插值，声明 last_frame=True
            # 让 media_generator 直接传递 end_image 而非降级为 reference_image
            return VideoCapabilities(last_frame=True, reference_images=True, max_reference_images=3)
        return VideoCapabilities(reference_images=True, max_reference_images=3)

    async def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        if self._is_agnes_video():
            return await self._generate_agnes_video(request)
        kwargs: dict = {
            "prompt": request.prompt,
            "model": self._model,
            "seconds": str(request.duration_seconds),
        }
        size = _resolve_size(request.resolution, request.aspect_ratio)
        if size is not None:
            kwargs["size"] = size

        # 收集所有参考图：start_image + reference_images
        refs = []
        if request.start_image and Path(request.start_image).exists():
            refs.append(_encode_start_image(Path(request.start_image)))
        if request.reference_images:
            for ref_path in request.reference_images:
                p = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
                if p.exists():
                    refs.append(_encode_start_image(p))
        if refs:
            # 单张图时保持 tuple 格式（API 兼容），多张时用 list
            kwargs["input_reference"] = refs[0] if len(refs) == 1 else refs

        logger.info("OpenAI 视频生成开始: model=%s, seconds=%s", self._model, kwargs["seconds"])
        logger.info("调用 %s 视频 SDK kwargs=%s", self.name, format_kwargs_for_log(kwargs))

        video = await self._create_video(**kwargs)
        final = await self._poll_until_complete(video.id, request.duration_seconds)

        content = await self._download_content_with_retry(final.id)

        def _write():
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            request.output_path.write_bytes(content.content)

        await asyncio.to_thread(_write)

        logger.info("OpenAI 视频下载完成: %s", request.output_path)

        return VideoGenerationResult(
            video_path=request.output_path,
            provider=PROVIDER_OPENAI,
            model=self._model,
            duration_seconds=int(final.seconds if final.seconds is not None else kwargs["seconds"]),
            task_id=final.id,
        )

    @with_retry_async(retryable_errors=OPENAI_RETRYABLE_ERRORS)
    async def _create_video(self, **kwargs):
        """仅创建视频任务（带重试）；轮询交由 _poll_until_complete 自管。"""
        return await self._client.videos.create(**kwargs)

    async def _poll_until_complete(self, video_id: str, duration_seconds: int):
        """轮询任务直到 status=='completed'。

        不复用 SDK 的 client.videos.poll：它仅识别 in_progress/queued/completed/failed，
        对接返回非标状态（如 NOT_START）的 OpenAI 兼容网关时会提前退出，导致下载未就绪任务。
        """
        max_wait = max(_MIN_POLL_TIMEOUT_SECONDS, float(duration_seconds) * _POLL_TIMEOUT_PER_SECOND)

        return await poll_with_retry(
            poll_fn=lambda: self._client.videos.retrieve(video_id),
            is_done=lambda v: v.status == "completed",
            is_failed=lambda v: f"Sora 视频生成失败: {getattr(v, 'error', None)}" if v.status == "failed" else None,
            poll_interval=_POLL_INTERVAL_SECONDS,
            max_wait=max_wait,
            retryable_errors=OPENAI_RETRYABLE_ERRORS,
            label="OpenAI",
            on_progress=lambda v, elapsed: logger.info(
                "OpenAI 视频生成中... 状态: %s, 已等待 %d 秒", v.status, int(elapsed)
            ),
        )

    @with_retry_async(
        max_attempts=DOWNLOAD_MAX_ATTEMPTS,
        backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
        retryable_errors=OPENAI_RETRYABLE_ERRORS,
    )
    async def _download_content_with_retry(self, video_id: str):
        """单独重试内容下载，避免因下载失败重新触发视频生成。"""
        return await self._client.videos.download_content(video_id)

    # ---- Agnes Video v2.0 路径 ----

    def _is_agnes_video(self) -> bool:
        return "agnes" in self._model.lower()

    @with_retry_async(retryable_errors=OPENAI_RETRYABLE_ERRORS)
    async def _create_agnes_task(self, payload: dict) -> str:
        """直接通过 HTTP POST 创建 Agnes 视频任务，返回 task_id。

        绕过 OpenAI SDK 的 multipart/form-data 编码——SDK 的 videos.create()
        强制 multipart，会把 frame_rate 等数值参数序列化为字符串，
        导致 Agnes API 返回 400（cannot unmarshal string into float64）。
        """
        base_url = str(self._client.base_url).rstrip("/")
        url = f"{base_url}/videos"
        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                body = resp.text
                raise RuntimeError(f"Agnes 创建视频任务失败 (HTTP {resp.status_code}): {body} | payload={payload}")
            data = resp.json()

        task_id = data.get("id") or data.get("task_id")
        if not task_id:
            raise RuntimeError(f"Agnes 创建任务响应缺少 id: {data}")
        return str(task_id)

    async def _generate_agnes_video(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        """Agnes Video v2.0 异步视频生成：start_image + reference_images 都需先上传拿 URL。

        上游若提供了与本地 path 平行对齐的远端 url（start_image_url / end_image_url /
        reference_image_urls），优先用 url 喂下游，省掉本地文件→公网的上传。
        """
        ref_url_iter = iter(request.reference_image_urls or [])
        start_url = request.start_image_url
        end_url = request.end_image_url

        # 1. 收集本地图片路径 + 对应 url（按 image_paths 顺序，url 缺位用空串占位）
        image_paths: list[Path] = []
        candidate_urls: list[str] = []
        if request.start_image and request.start_image.exists():
            image_paths.append(request.start_image)
            candidate_urls.append(start_url or "")
        if request.reference_images:
            for ref in request.reference_images:
                if ref.exists():
                    image_paths.append(ref)
                    candidate_urls.append(next(ref_url_iter, "") or "")
        is_keyframes = bool(request.start_image and request.end_image)
        if is_keyframes and request.end_image and request.end_image.exists():
            image_paths.append(request.end_image)
            candidate_urls.append(end_url or "")

        # 2. 优先用 url；缺位回退到本地上传
        urls: list[str] = []
        for path, url in zip(image_paths, candidate_urls, strict=True):
            if url:
                urls.append(url)
            else:
                urls.append(await upload_to_tmpfiles(path))

        # 3. 构造 Agnes JSON 请求体
        #   - 单图 i2v：image = 字符串（顶层）
        #   - 多图 / 关键帧：extra_body.image = 数组，extra_body.mode = "keyframes"
        # 参见 Agnes-Video-V2.0 API 文档
        payload: dict = {
            "model": AGNES_VIDEO_MODEL,
            "prompt": request.prompt,
        }
        agnes_video_params = _resolve_agnes_video_params(
            request.duration_seconds,
            request.aspect_ratio,
            request.resolution,
        )
        payload.update(agnes_video_params)
        if len(urls) == 1 and not is_keyframes:
            payload["image"] = urls[0]
        elif urls:
            extra_body: dict = {"image": urls}
            if is_keyframes:
                extra_body["mode"] = "keyframes"
            payload["extra_body"] = extra_body

        logger.info(
            "Agnes 视频生成开始: model=%s, duration=%ss, urls=%d, keyframes=%s",
            self._model,
            request.duration_seconds,
            len(urls),
            is_keyframes,
        )
        logger.info("调用 Agnes 视频 API payload=%s", format_kwargs_for_log(payload))

        # 4. 创建 + 轮询（轮询器与 Sora 路径复用，status 字符串一致）
        task_id = await self._create_agnes_task(payload)
        final = await self._poll_until_complete(task_id, request.duration_seconds)

        # 5. agnes 不走 download_content，直接从响应拿 video_url 字段
        video_url = getattr(final, "video_url", None) or getattr(final, "remixed_from_video_id", None)
        if not video_url:
            raise RuntimeError(f"Agnes 视频响应缺少 video_url 字段 (model={self._model}, id={final.id})")

        await self._download_agnes_video_from_url(video_url, request.output_path)

        # agnes 的 seconds 字段是字符串（可能是 "7.7" 这种小数），先 float 再 round
        seconds_field = getattr(final, "seconds", None)
        duration = round(float(seconds_field)) if seconds_field is not None else request.duration_seconds

        logger.info("Agnes 视频下载完成: %s", request.output_path)

        return VideoGenerationResult(
            video_path=request.output_path,
            provider=PROVIDER_OPENAI,
            model=self._model,
            duration_seconds=duration,
            task_id=final.id,
        )

    async def _download_agnes_video_from_url(self, url: str, output_path: Path) -> None:
        """从 agnes video_url 流式下载到本地路径。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=_AGNES_VIDEO_DOWNLOAD_TIMEOUT_SECONDS) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)


def _encode_start_image(image_path: Path) -> tuple[str, bytes, str]:
    mime = IMAGE_MIME_TYPES.get(image_path.suffix.lower(), "image/png")
    return (image_path.name, image_path.read_bytes(), mime)

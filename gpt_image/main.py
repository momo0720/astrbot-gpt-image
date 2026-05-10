import asyncio
import base64
import io
import random
import time
from pathlib import Path

import aiohttp
from PIL import Image as PILImage

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Reply
from astrbot.api.star import Context, Star
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path


class GptImagePlugin(Star):
    """Shop API text-to-image plugin."""

    MODEL_ALIASES = {
        "gpt-image-2": None,
        "gpt-image-2-1k": "1024x1024",
        "gpt-image-2-2k": "2048x2048",
    }

    def __init__(self, context: Context, config=None) -> None:
        super().__init__(context)
        self.config = config or {}
        self._models_cache: list[str] = []
        self._models_cache_expires_at = 0.0
        self._key_state_path = (
            Path(get_astrbot_temp_path()) / "shop_image_key_state.txt"
        )

    def _cfg(self, key: str, default=None):
        return self.config.get(key, default)

    def _get_api_keys(self) -> list[str]:
        api_keys = self._cfg("api_keys", []) or []
        if isinstance(api_keys, str):
            api_keys = [line.strip() for line in api_keys.splitlines() if line.strip()]
        if not api_keys and self._cfg("api_key"):
            api_keys = [str(self._cfg("api_key")).strip()]
        return [str(key).strip() for key in api_keys if str(key).strip()]

    def _check_cfg(self) -> str | None:
        if not self._cfg("api_base_url"):
            return "❌ 插件未配置 API 地址。"
        if not self._get_api_keys():
            return "❌ 插件未配置 API Key。"
        if not self._cfg("text_model"):
            return "❌ 插件未配置文生图模型名称。"
        return None

    def _normalize_model_name(self, model: str) -> str:
        model = str(model).strip()
        return model if model in self.MODEL_ALIASES else model

    def _resolve_model_payload(self, model: str) -> tuple[str, str | None]:
        normalized = self._normalize_model_name(model)
        if normalized in self.MODEL_ALIASES:
            return "gpt-image-2", self.MODEL_ALIASES[normalized]
        return normalized, None

    def _get_default_model(self) -> str:
        return str(self._cfg("text_model", "gpt-image-2")).strip() or "gpt-image-2"

    def _get_models_cache_ttl(self) -> int:
        return int(self._cfg("models_cache_ttl", 300))

    def _get_key_strategy(self) -> str:
        strategy = str(self._cfg("key_strategy", "random")).strip().lower()
        return strategy if strategy in {"random", "round_robin"} else "random"

    def _get_ordered_api_keys(self) -> list[tuple[int, str]]:
        api_keys = self._get_api_keys()
        indexed_keys = list(enumerate(api_keys, start=1))
        if len(indexed_keys) <= 1:
            return indexed_keys

        strategy = self._get_key_strategy()
        if strategy == "round_robin":
            start_index = 0
            if self._key_state_path.exists():
                try:
                    start_index = int(self._key_state_path.read_text().strip())
                except Exception:
                    start_index = 0
            start_index = start_index % len(indexed_keys)
            ordered = indexed_keys[start_index:] + indexed_keys[:start_index]
            next_index = (start_index + 1) % len(indexed_keys)
            self._key_state_path.write_text(str(next_index), encoding="utf-8")
            return ordered

        random.shuffle(indexed_keys)
        return indexed_keys

    def _parse_prompt_model_and_quality(
        self, prompt: str, models: list[str]
    ) -> tuple[str, str, str | None]:
        prompt = prompt.strip()
        parts = prompt.split(maxsplit=1)
        size_override: str | None = None

        if parts and parts[0].lower() in {"1k", "2k"}:
            size_override = "1024x1024" if parts[0].lower() == "1k" else "2048x2048"
            prompt = parts[1].strip() if len(parts) > 1 else ""
            parts = prompt.split(maxsplit=1) if prompt else []

        if parts and parts[0].isdigit():
            index = int(parts[0]) - 1
            if 0 <= index < len(models):
                prompt = parts[1].strip() if len(parts) > 1 else ""
                return models[index], prompt, size_override

        return self._get_default_model(), prompt, size_override

    async def _read_image_component(self, comp: Image) -> bytes | None:
        try:
            image_path = await comp.convert_to_file_path()
            with open(image_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"读取输入图片失败: {e}")
            return None

    async def _get_avatar_bytes(self, user_id: str) -> bytes | None:
        if not str(user_id).isdigit():
            return None
        timeout_seconds = int(self._cfg("timeout", 300))
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        avatar_url = f"http://q4.qlogo.cn/g?b=qq&nk={user_id}&s=640"
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                avatar_bytes, error = await self._download_image(session, avatar_url)
            if error:
                logger.warning(f"读取头像失败，user_id={user_id}: {error}")
                return None
            return avatar_bytes
        except Exception as e:
            logger.warning(f"读取头像异常，user_id={user_id}: {e}")
            return None

    async def _get_images_from_event(
        self, event: AstrMessageEvent, max_count: int = 4
    ) -> list[bytes]:
        images = []

        async def _append_if_available(comp) -> bool:
            image_bytes = None
            if isinstance(comp, Image):
                image_bytes = await self._read_image_component(comp)
            elif isinstance(comp, At) and str(comp.qq) != "all":
                image_bytes = await self._get_avatar_bytes(str(comp.qq))

            if image_bytes:
                logger.info(
                    f"GPT Image 收到参考图，来源={'@头像' if isinstance(comp, At) else '图片'}，当前数量={len(images) + 1}/{max_count}"
                )
                images.append(image_bytes)
                return len(images) >= max_count
            return False

        for comp in event.message_obj.message:
            if isinstance(comp, Reply) and comp.chain:
                for quoted_comp in comp.chain:
                    if await _append_if_available(quoted_comp):
                        return images
            if await _append_if_available(comp):
                return images
        return images

    def _detect_mime_type(self, image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_bytes.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        return "image/png"

    async def _download_image(
        self, session: aiohttp.ClientSession, url: str
    ) -> tuple[bytes | None, str | None]:
        last_error = None
        retries = int(self._cfg("download_retries", 2))
        for attempt in range(retries + 1):
            try:
                async with session.get(url) as img_resp:
                    if img_resp.status != 200:
                        last_error = f"图片下载失败：HTTP {img_resp.status}"
                    else:
                        return await img_resp.read(), None
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"GPT Image 图片下载异常，第 {attempt + 1}/{retries + 1} 次: {e}"
                )
            if attempt < retries:
                await asyncio.sleep(1)
        return None, last_error or "图片下载失败"

    def _save_image_to_temp_file(self, image_bytes: bytes, model: str) -> str:
        temp_dir = get_astrbot_temp_path()
        file_path = (
            f"{temp_dir}/shop_image_{model}_{int(time.time() * 1000)}_"
            f"{random.randint(1000, 9999)}.jpg"
        )
        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(file_path, format="JPEG", quality=95)
        return file_path

    async def _fetch_models(
        self, force_refresh: bool = False
    ) -> tuple[list[str], str | None]:
        now = time.time()
        if (
            not force_refresh
            and self._models_cache
            and now < self._models_cache_expires_at
        ):
            return list(self._models_cache), None

        base_url = self._cfg(
            "api_base_url", "https://example.com"
        ).rstrip("/")
        timeout = aiohttp.ClientTimeout(total=int(self._cfg("timeout", 180)))
        api_keys = self._get_api_keys()
        if not api_keys:
            return [], "❌ 插件未配置 API Key。"

        last_error = None
        for key_index, api_key in enumerate(api_keys, start=1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        f"{base_url}/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    ) as resp:
                        data = await resp.json()
                    if resp.status != 200:
                        last_error = (
                            data.get("error", {}).get("message")
                            or data.get("message")
                            or f"HTTP {resp.status}"
                        )
                        logger.warning(
                            f"GPT Image 获取模型列表失败，key {key_index}/{len(api_keys)}: {last_error}"
                        )
                        continue
                    models = [
                        str(item.get("id")).strip()
                        for item in (data.get("data") or [])
                        if isinstance(item, dict) and str(item.get("id", "")).strip()
                    ]
                    if "gpt-image-2" in models:
                        for alias in ["gpt-image-2-1k", "gpt-image-2-2k"]:
                            if alias not in models:
                                models.append(alias)
                    self._models_cache = list(models)
                    self._models_cache_expires_at = (
                        time.time() + self._get_models_cache_ttl()
                    )
                    return models, None
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"GPT Image 获取模型列表异常，key {key_index}/{len(api_keys)}: {e}"
                )
        return [], last_error or "获取模型列表失败"

    async def _request_image_api(
        self,
        *,
        prompt: str,
        model: str,
        source_images: list[bytes] | None = None,
        size_override: str | None = None,
    ) -> tuple[bytes | None, str | None, int | None, str | None]:
        base_url = self._cfg(
            "api_base_url", "https://example.com"
        ).rstrip("/")
        timeout_seconds = int(self._cfg("timeout", 300))
        timeout = aiohttp.ClientTimeout(
            total=timeout_seconds, sock_read=timeout_seconds
        )
        retries = int(self._cfg("request_retries", 0))
        ordered_api_keys = self._get_ordered_api_keys()
        resolved_model, size_hint = self._resolve_model_payload(model)
        source_images = source_images or []
        if size_override in {"1024x1024", "2048x2048"}:
            size_hint = size_override
        elif size_hint is None and resolved_model != "gpt-image-2":
            size_hint = self._cfg("size", "1024x1024")
        last_error = None

        for key_tag, api_key in ordered_api_keys:
            for attempt in range(retries + 1):
                try:
                    logger.info(
                        f"GPT Image 请求开始，model={resolved_model}，size={size_hint}，参考图数量={len(source_images)}，key #{key_tag}/{len(ordered_api_keys)}，尝试 {attempt + 1}/{retries + 1}"
                    )
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        }
                        if source_images:
                            payload_images = []
                            for source_image in source_images:
                                mime_type = self._detect_mime_type(source_image)
                                image_data_url = (
                                    f"data:{mime_type};base64,"
                                    f"{base64.b64encode(source_image).decode()}"
                                )
                                payload_images.append({"image_url": image_data_url})
                            payload = {
                                "model": resolved_model,
                                "prompt": prompt,
                                "images": payload_images,
                            }
                            if size_hint:
                                payload["size"] = size_hint
                            request_headers = headers
                            endpoint = f"{base_url}/v1/images/edits"
                        else:
                            payload = {
                                "model": resolved_model,
                                "prompt": prompt,
                            }
                            if size_hint:
                                payload["size"] = size_hint
                            request_headers = headers
                            endpoint = f"{base_url}/v1/images/generations"

                        async with session.post(
                            endpoint,
                            headers=request_headers,
                            json=payload,
                        ) as resp:
                            try:
                                data = await resp.json()
                            except Exception:
                                text = await resp.text()
                                last_error = (
                                    f"接口返回非 JSON：HTTP {resp.status} {text[:300]}"
                                )
                                logger.warning(f"GPT Image 非 JSON 响应: {last_error}")
                                if attempt < retries:
                                    await asyncio.sleep(1)
                                    continue
                                break

                        if resp.status != 200:
                            error_obj = (
                                data.get("error") if isinstance(data, dict) else {}
                            )
                            if not isinstance(error_obj, dict):
                                error_obj = {}
                            last_error = (
                                error_obj.get("message")
                                or (
                                    data.get("message")
                                    if isinstance(data, dict)
                                    else None
                                )
                                or f"HTTP {resp.status}"
                            )
                            if source_images and resp.status == 404:
                                last_error = (
                                    "当前 Shop API 不支持图生图接口 /images/edits"
                                )
                            mode_name = "图生图" if source_images else "文生图"
                            logger.warning(
                                f"GPT Image {mode_name}失败，model={resolved_model}，size={size_hint}，key #{key_tag}/{len(ordered_api_keys)}，HTTP {resp.status}: {last_error}"
                            )
                            if attempt < retries:
                                await asyncio.sleep(1)
                                continue
                            break

                        items = data.get("data") or []
                        if not items:
                            last_error = "接口未返回图片数据"
                            if attempt < retries:
                                await asyncio.sleep(1)
                                continue
                            break

                        item = items[0]
                        if item.get("b64_json"):
                            return (
                                base64.b64decode(item["b64_json"]),
                                None,
                                key_tag,
                                size_hint,
                            )
                        if item.get("url"):
                            image_bytes_result, error = await self._download_image(
                                session, item["url"]
                            )
                            if image_bytes_result:
                                return (
                                    image_bytes_result,
                                    None,
                                    key_tag,
                                    size_hint,
                                )
                            last_error = error
                            if attempt < retries:
                                await asyncio.sleep(1)
                                continue
                            break

                        last_error = "未找到可用图片字段（b64_json/url）"
                        if attempt < retries:
                            await asyncio.sleep(1)
                            continue
                        break
                except asyncio.TimeoutError:
                    last_error = f"请求超时（>{timeout_seconds}秒）"
                    logger.warning(
                        f"GPT Image 请求超时，model={model}，key #{key_tag}/{len(ordered_api_keys)}，尝试 {attempt + 1}/{retries + 1}，timeout={timeout_seconds}s"
                    )
                    if attempt < retries:
                        await asyncio.sleep(1)
                        continue
                except Exception as e:
                    last_error = str(e) or repr(e)
                    logger.warning(
                        f"GPT Image 请求异常，model={model}，key #{key_tag}/{len(ordered_api_keys)}，尝试 {attempt + 1}/{retries + 1}: {last_error}"
                    )
                    if attempt < retries:
                        await asyncio.sleep(1)
                        continue
            logger.warning(f"GPT Image 切换下一个 key，当前错误: {last_error}")
        return None, last_error or "画图请求失败", None, size_hint

    @filter.command("gpt画图帮助")
    async def draw_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "🎨 GPT 画图插件\n"
            "命令：/gpt画图 <提示词>\n"
            "命令：/gpt画图 1k <提示词>\n"
            "命令：/gpt画图 2k <提示词>\n"
            "命令：/gpt画图 1 <提示词>\n"
            "命令：/gpt画图模型列表\n"
            f"默认模型：{self._get_default_model()}\n"
            f"模型列表缓存：{self._get_models_cache_ttl()} 秒\n"
            f"Key 策略：{self._get_key_strategy()}\n"
            "特性：支持多个 API Key、失败自动重试并切换 Key"
        )

    @filter.command("gpt画图模型列表")
    async def draw_model_list(self, event: AstrMessageEvent):
        err = self._check_cfg()
        if err:
            yield event.plain_result(err)
            return

        yield event.plain_result("🔎 正在获取模型列表，请稍候…")
        models, error = await self._fetch_models()
        if error:
            yield event.plain_result(f"❌ 获取模型列表失败：{error}")
            return
        if not models:
            yield event.plain_result("❌ 模型列表为空")
            return

        lines = ["🧾 可用画图模型列表："]
        for index, model in enumerate(models, start=1):
            suffix = "（默认）" if model == self._get_default_model() else ""
            lines.append(f"{index}. {model}{suffix}")
        yield event.plain_result("\n".join(lines))

    @filter.command("gpt画图")
    async def draw_image(self, event: AstrMessageEvent):
        err = self._check_cfg()
        if err:
            yield event.plain_result(err)
            return

        prompt = event.message_str.strip()
        if prompt.startswith("/gpt画图"):
            prompt = prompt[len("/gpt画图") :].strip()
        elif prompt.startswith("gpt画图"):
            prompt = prompt[len("gpt画图") :].strip()

        models, model_error = await self._fetch_models()
        if model_error:
            yield event.plain_result(f"❌ 获取模型列表失败：{model_error}")
            return

        model, prompt, size_override = self._parse_prompt_model_and_quality(
            prompt, models
        )
        if not prompt:
            yield event.plain_result(
                "用法：/gpt画图 <提示词>、/gpt画图 1k <提示词>、/gpt画图 2k <提示词> 或 /gpt画图 1 <提示词>"
            )
            return

        image_inputs = await self._get_images_from_event(event, max_count=4)
        mode = "图生图" if image_inputs else "文生图"

        try:
            started_at = time.perf_counter()
            yield event.plain_result(f"🎨 正在进行[{mode}]，请稍候…")
            image_bytes, error, key_index, size_hint = await self._request_image_api(
                prompt=prompt,
                model=model,
                source_images=image_inputs,
                size_override=size_override,
            )
            elapsed_seconds = time.perf_counter() - started_at
            if error:
                yield event.plain_result(f"❌ 画图失败：{error}")
                return
            key_tag = f"#key{key_index}" if key_index else "#key?"
            image_path = self._save_image_to_temp_file(image_bytes, model)
            size_line = f"\n尺寸参数：{size_hint}" if size_hint else ""
            yield (
                event.make_result()
                .message(
                    f"🖼️ 模式：{mode}\n模型：{model}{size_line}\n使用：{key_tag}\n耗时：{elapsed_seconds:.2f}秒\n提示词：{prompt}"
                )
                .file_image(image_path)
            )
        except Exception as e:
            logger.error(f"GPT Image 画图异常: {e}")
            yield event.plain_result(f"❌ 画图失败：{e}")

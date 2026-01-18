import asyncio
import base64
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path

import aiohttp
from PIL import Image as PILImage

import astrbot.core.message.components as Comp
from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Image
from astrbot.core.platform.astr_message_event import AstrMessageEvent


class ImageWorkflow:
    def __init__(self):
        self.session = aiohttp.ClientSession()

    async def _download_image(self, url: str) -> bytes | None:
        try:
            async with self.session.get(url, timeout=30) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"图片下载失败: {e}")
            return None

    async def _get_avatar(self, user_id: str) -> bytes | None:
        if not user_id.isdigit():
            user_id = "".join(random.choices("0123456789", k=9))
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            async with self.session.get(avatar_url, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"下载头像失败: {e}")
            return None

    def _extract_first_frame_sync(self, raw: bytes) -> bytes:
        try:
            img_io = io.BytesIO(raw)
            img = PILImage.open(img_io)
            if img.format != "GIF":
                return raw
            logger.info("检测到GIF, 将抽取 GIF 的第一帧来生图")
            first_frame = img.convert("RGBA")
            out_io = io.BytesIO()
            first_frame.save(out_io, format="PNG")
            return out_io.getvalue()
        except Exception as e:
            logger.warning(f"图片预处理失败，尝试直接使用原数据: {e}")
            return raw

    async def _load_bytes(self, src: str) -> bytes | None:
        raw: bytes | None = None
        loop = asyncio.get_running_loop()

        if Path(src).is_file():
            raw = await loop.run_in_executor(None, Path(src).read_bytes)
        elif src.startswith("http"):
            raw = await self._download_image(src)
        elif src.startswith("base64://"):
            raw = await loop.run_in_executor(None, base64.b64decode, src[9:])

        if not raw:
            return None
        return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

    async def get_first_image(self, event: AstrMessageEvent) -> bytes | None:
        if not event.message_obj:
            return None

        for s in event.message_obj.message:
            if isinstance(s, Comp.Reply) and s.chain:
                for seg in s.chain:
                    if isinstance(seg, Comp.Image):
                        if seg.url and (img := await self._load_bytes(seg.url)):
                            return img
                        if seg.file and (img := await self._load_bytes(seg.file)):
                            return img
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Image):
                if seg.url and (img := await self._load_bytes(seg.url)):
                    return img
                if seg.file and (img := await self._load_bytes(seg.file)):
                    return img
            elif isinstance(seg, Comp.At):
                if avatar := await self._get_avatar(str(seg.qq)):
                    return avatar

        sender_id = event.get_sender_id()
        if sender_id:
            return await self._get_avatar(sender_id)
        return None

    async def terminate(self):
        if self.session and not self.session.closed:
            await self.session.close()


@register(
    "astrbot_plugin_figurine_workshop",
    "长安某",
    "使用 Gemini 2.5/3.0 或 OpenAI 兼容 API 进行图片风格化",
    "1.5.4",
)
class LMArenaPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.save_image = config.get("save_image", False)
        self.plugin_data_dir = StarTools.get_data_dir(
            "astrbot_plugin_figurine_workshop"
        )
        self.usage_file = self.plugin_data_dir / "daily_usage.json"

        self.api_type = self.conf.get("api_type", "google").lower()
        self.api_keys = self.conf.get("gemini_api_keys", [])
        self.current_key_index = 0
        self.api_base_url = self.conf.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )
        self.gemini_model = self.conf.get("gemini_model", "gemini-2.5-flash-image")
        self.figurine_style = self.conf.get("figurine_style", "deluxe_box")

        self.enable_private_chat = self.conf.get("enable_private_chat", True)
        self.group_mode = self.conf.get("group_mode", "all")
        self.group_whitelist = set(str(x) for x in self.conf.get("group_whitelist", []))
        self.group_blacklist = set(str(x) for x in self.conf.get("group_blacklist", []))
        self.daily_limit = self.conf.get("daily_limit", 0)
        self.show_quota_reminder = self.conf.get("show_quota_reminder", True)

        if not self.api_keys:
            logger.error("LMArenaPlugin: 未配置任何 API 密钥")

    async def initialize(self):
        self.iwf = ImageWorkflow()

    def _check_permission(self, event: AstrMessageEvent) -> bool:
        group_id = event.message_obj.group_id
        if not group_id:
            return self.enable_private_chat

        group_id = str(group_id)
        if self.group_mode == "whitelist":
            return group_id in self.group_whitelist
        elif self.group_mode == "blacklist":
            return group_id not in self.group_blacklist
        return True

    def _load_usage_data(self) -> dict:
        if not self.usage_file.exists():
            return {"date": "", "counts": {}}
        try:
            return json.loads(self.usage_file.read_text(encoding='utf-8'))
        except Exception:
            return {"date": "", "counts": {}}

    def _save_usage_data(self, data: dict):
        try:
            if not self.plugin_data_dir.exists():
                self.plugin_data_dir.mkdir(parents=True)
            self.usage_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            logger.error(f"保存配额数据失败: {e}")

    def _check_and_update_quota(self, user_id: str) -> tuple[bool, int]:
        if self.daily_limit <= 0:
            return True, 9999

        today_str = datetime.now().strftime("%Y-%m-%d")
        data = self._load_usage_data()

        if data.get("date") != today_str:
            data = {"date": today_str, "counts": {}}
        
        counts = data.get("counts", {})
        user_count = counts.get(user_id, 0)

        if user_count >= self.daily_limit:
            return False, 0
        
        counts[user_id] = user_count + 1
        data["counts"] = counts
        self._save_usage_data(data)
        
        return True, self.daily_limit - (user_count + 1)

    @filter.regex(r"(?i)^(手办化|cos化|!galgame|双v)", priority=3)
    async def on_generate_request(self, event: AstrMessageEvent):
        if not self._check_permission(event):
            return

        sender_id = event.get_sender_id()
        is_allowed, remaining_quota = self._check_and_update_quota(sender_id)
        
        if not is_allowed:
            yield event.plain_result(f"今日使用次数已达上限 ({self.daily_limit}次)，请明天再来吧~")
            return

        msg_str = event.message_obj.message_str.strip()
        trigger_match = re.match(r"(?i)^(手办化|!galgame|cos化|双v)", msg_str)
        if not trigger_match:
            return

        raw_command = trigger_match.group(1)
        command_lower = raw_command.lower()

        img_bytes = await self.iwf.get_first_image(event)
        if not img_bytes:
            yield event.plain_result("缺少图片参数（可以发送图片或@用户）")
            return

        user_input_text = re.sub(
            r"(?i)^(手办化|!galgame|cos化|双v)\s*", "", msg_str, count=1
        ).strip()

        prompt_key = ""
        prompts_config = self.conf.get("prompts", {})

        if command_lower == "手办化":
            prompt_key = self.figurine_style
        elif command_lower == "!galgame":
            prompt_key = "galgame"
        elif command_lower == "cos化":
            prompt_key = "cosplay"
        elif command_lower == "双v":
            prompt_key = "double_v"
        else:
            return

        base_prompt = prompts_config.get(prompt_key, "")
        if not base_prompt:
            yield event.plain_result(f"配置错误：找不到 key '{prompt_key}'")
            return

        final_prompt = (
            f"{base_prompt}\n\nAdditional user requirements: {user_input_text}"
            if user_input_text
            else base_prompt
        )

        quota_msg = f"\n(今日剩余次数: {remaining_quota})" if self.daily_limit > 0 and self.show_quota_reminder else ""
        yield event.plain_result(f"正在请求，请稍后...{quota_msg}")

        res = await self._generate_image_core(img_bytes, final_prompt)

        if isinstance(res, bytes):
            yield event.chain_result([Image.fromBytes(res)])
            if self.save_image:
                self._save_image_to_disk(res, command_lower.replace("!", ""))
        elif isinstance(res, str):
            yield event.plain_result(f"生成失败: {res}")
        else:
            yield event.plain_result("生成失败，发生未知错误。")

    def _save_image_to_disk(self, img_bytes: bytes, prefix: str):
        try:
            if not self.plugin_data_dir.exists():
                self.plugin_data_dir.mkdir(parents=True)
            save_path = self.plugin_data_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            def write_file():
                with save_path.open("wb") as f:
                    f.write(img_bytes)
            asyncio.get_running_loop().run_in_executor(None, write_file)
        except Exception as e:
            logger.error(f"保存图片失败: {e}")

    async def _generate_image_core(self, image_bytes: bytes, prompt: str) -> bytes | str | None:
        async def operation(api_key):
            if self.api_type == "openai":
                return await self._send_openai_request(self.gemini_model, prompt, image_bytes, api_key)
            else:
                return await self._send_google_request(self.gemini_model, prompt, image_bytes, api_key)

        image_data = await self._with_retry(operation)
        return image_data if image_data else "所有API密钥均尝试失败或模型未返回图片"

    def _get_current_key(self):
        return self.api_keys[self.current_key_index] if self.api_keys else None

    def _switch_key(self):
        if self.api_keys:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

    async def _send_google_request(self, model_name, prompt, image_bytes, api_key):
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/png", "data": image_base64}},
                    ],
                }
            ],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        endpoint = f"{self.api_base_url.strip().rstrip('/')}/v1beta/models/{model_name}:generateContent?key={api_key}"
        async with self.iwf.session.post(url=endpoint, json=payload, headers={"Content-Type": "application/json"}) as response:
            if response.status != 200:
                response.raise_for_status()
            data = await response.json()

        if "candidates" in data and data["candidates"]:
            for part in data["candidates"][0].get("content", {}).get("parts", []):
                if "inlineData" in part and "data" in part["inlineData"]:
                    return base64.b64decode(part["inlineData"]["data"])
        raise Exception("Google API 未返回图片数据")

    async def _send_openai_request(self, model_name, prompt, image_bytes, api_key):
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]
        payload = {"model": model_name, "messages": messages, "max_tokens": 4096}
        base_url = self.api_base_url.strip()
        endpoint = base_url if "/v1/chat/completions" in base_url else f"{base_url.rstrip('/')}/v1/chat/completions"

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        async with self.iwf.session.post(url=endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                err_text = await response.text()
                raise Exception(f"HTTP {response.status}: {err_text}")
            data = await response.json()

        try:
            message = data.get("choices", [{}])[0].get("message", {})
            
            # 处理 'images' 列表字段格式
            images = message.get("images")
            if isinstance(images, list) and len(images) > 0:
                img_obj = images[0]
                img_url = img_obj.get("image_url", {}).get("url", "")
                if img_url.startswith("data:"):
                    return base64.b64decode(img_url.split(",", 1)[1])
                elif img_url.startswith("http"):
                    return await self.iwf._download_image(img_url)

            # 回退到处理 content 字段中的 Base64/URL
            content = message.get("content")
            if not content:
                logger.error(f"API 响应中未找到图片字段。完整响应: {data}")
                return "API 返回结果异常，未找到图片。请将此错误截图提交作者项目git或者qq"

            img_matches = re.findall(r"!\[.*?\]\((.*?)\)", content)
            if img_matches:
                img_url = img_matches[0].strip()
                if img_url.startswith("data:"):
                    return base64.b64decode(img_url.split(",", 1)[1])
                return await self.iwf._download_image(img_url)

            if "base64," in content:
                b64_part = content.split("base64,")[1].strip().split(" ")[0].split(")")[0]
                return base64.b64decode(b64_part)

            clean_content = re.sub(r'[\s\n\r]', '', content)
            if len(clean_content) > 100:
                try:
                    return base64.b64decode(clean_content)
                except Exception:
                    pass

            return f"无法识别响应中的图片数据: {content[:100]}"
        except Exception as e:
            raise Exception(f"解析响应失败: {str(e)}")

    async def _with_retry(self, operation, *args, **kwargs):
        max_attempts = len(self.api_keys)
        for attempt in range(max_attempts):
            try:
                return await operation(self._get_current_key(), *args, **kwargs)
            except Exception as e:
                logger.error(f"密钥索引 {self.current_key_index} 尝试失败: {e}")
                if attempt < max_attempts - 1:
                    self._switch_key()
                else:
                    return None
        return None

    async def terminate(self):
        if self.iwf:
            await self.iwf.terminate()

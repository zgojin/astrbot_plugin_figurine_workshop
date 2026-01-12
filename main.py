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
        """
        使用PIL库处理图片数据。如果是GIF，则提取第一帧并转为PNG。
        """
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

        # 尝试获取发送者头像
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
    "使用 Gemini 2.5/3.0 或 OpenAI 兼容 API 进行图片风格化（手办/Galgame/Cos化/双V）",
    "1.5.1",
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

        # 兼容性配置
        self.api_type = self.conf.get("api_type", "google").lower()  # google 或 openai
        self.api_keys = self.conf.get("gemini_api_keys", [])
        self.current_key_index = 0
        self.api_base_url = self.conf.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )
        # 默认模型
        self.gemini_model = self.conf.get("gemini_model", "gemini-2.5-flash-image")
        # 手办化的默认风格
        self.figurine_style = self.conf.get("figurine_style", "deluxe_box")

        # === 权限与配额配置 ===
        self.enable_private_chat = self.conf.get("enable_private_chat", True)
        self.group_mode = self.conf.get("group_mode", "all")
        self.group_whitelist = set(str(x) for x in self.conf.get("group_whitelist", []))
        self.group_blacklist = set(str(x) for x in self.conf.get("group_blacklist", []))
        self.daily_limit = self.conf.get("daily_limit", 0)
        self.show_quota_reminder = self.conf.get("show_quota_reminder", True)

        if not self.api_keys:
            logger.error("LMArenaPlugin: 未配置任何 API 密钥")

        logger.info(
            f"LMArenaPlugin 加载完成: API类型={self.api_type}, 模型={self.gemini_model}, 每日限制={self.daily_limit}"
        )

    async def initialize(self):
        self.iwf = ImageWorkflow()

    def _check_permission(self, event: AstrMessageEvent) -> bool:
        """检查私聊开关和群组黑白名单"""
        group_id = event.message_obj.group_id
        
        # 1. 检查私聊
        if not group_id:
            if not self.enable_private_chat:
                return False
            return True # 私聊开启且是私聊消息，直接通过

        # 2. 检查群组
        group_id = str(group_id)
        if self.group_mode == "whitelist":
            if group_id not in self.group_whitelist:
                return False
        elif self.group_mode == "blacklist":
            if group_id in self.group_blacklist:
                return False
        
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
        """
        检查并更新用户配额
        返回: (是否允许, 剩余次数)
        """
        if self.daily_limit <= 0:
            return True, 9999

        today_str = datetime.now().strftime("%Y-%m-%d")
        data = self._load_usage_data()

        # 如果日期变更，重置数据
        if data.get("date") != today_str:
            data = {"date": today_str, "counts": {}}
        
        counts = data.get("counts", {})
        user_count = counts.get(user_id, 0)

        if user_count >= self.daily_limit:
            return False, 0
        
        # 通过，计数+1并保存
        counts[user_id] = user_count + 1
        data["counts"] = counts
        self._save_usage_data(data)
        
        return True, self.daily_limit - (user_count + 1)

    @filter.regex(r"(?i)^(手办化|cos化|!galgame|双v)", priority=3)
    async def on_generate_request(self, event: AstrMessageEvent):
        """
        统一处理所有图片生成请求
        """
        # 权限检查
        if not self._check_permission(event):
            return

        # 额度检查
        sender_id = event.get_sender_id()
        is_allowed, remaining_quota = self._check_and_update_quota(sender_id)
        
        if not is_allowed:
            yield event.plain_result(f"今日使用次数已达上限 ({self.daily_limit}次)，请明天再来吧~")
            return

        # 触发指令
        msg_str = event.message_obj.message_str.strip()

        trigger_match = re.match(r"(?i)^(手办化|!galgame|cos化|双v)", msg_str)
        if not trigger_match:
            return

        raw_command = trigger_match.group(1)
        command_lower = raw_command.lower()

        # 获取图片
        img_bytes = await self.iwf.get_first_image(event)
        if not img_bytes:
            yield event.plain_result("缺少图片参数（可以发送图片或@用户）")
            return

        user_input_text = re.sub(
            r"(?i)^(手办化|!galgame|cos化|双v)\s*", "", msg_str, count=1
        ).strip()

        prompt_key = ""
        base_prompt = ""
        final_prompt = ""
        prompts_config = self.conf.get("prompts", {})

        style_display_name = "未知风格"
        if command_lower == "手办化":
            prompt_key = self.figurine_style  # 使用配置中的默认手办风格
            style_display_name = f"手办化-{prompt_key}"

        elif command_lower == "!galgame":
            prompt_key = "galgame"
            style_display_name = "Galgame"

        elif command_lower == "cos化":
            prompt_key = "cosplay"
            style_display_name = "Cos化"

        elif command_lower == "双v":
            prompt_key = "double_v"
            style_display_name = "双V模式"

        else:
            return

        base_prompt = prompts_config.get(prompt_key, "")

        if not base_prompt:
            yield event.plain_result(
                f"配置错误：找不到 key 为 '{prompt_key}' 的提示词。"
            )
            return

        # 统一将用户输入作为附加要求添加到提示词后
        final_prompt = (
            f"{base_prompt}\n\nAdditional user requirements: {user_input_text}"
            if user_input_text
            else base_prompt
        )

        # 构造提示消息
        quota_msg = ""
        if self.daily_limit > 0 and self.show_quota_reminder:
            quota_msg = f"\n(今日剩余次数: {remaining_quota})"

        yield event.plain_result(f"正在请求，请稍后...{quota_msg}")

        logger.info(f"生成 Prompt ({style_display_name}): {final_prompt[:50]}...")

        res = await self._generate_image_core(img_bytes, final_prompt)

        safe_prefix = command_lower.replace("!", "")

        if isinstance(res, bytes):
            yield event.chain_result([Image.fromBytes(res)])
            if self.save_image:
                self._save_image_to_disk(res, safe_prefix)
        elif isinstance(res, str):
            yield event.plain_result(f"生成失败: {res}")
        else:
            yield event.plain_result("生成失败，发生未知错误。")

    def _save_image_to_disk(self, img_bytes: bytes, prefix: str):
        try:
            if not self.plugin_data_dir.exists():
                self.plugin_data_dir.mkdir(parents=True)

            save_path = (
                self.plugin_data_dir
                / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            )

            def write_file():
                with save_path.open("wb") as f:
                    f.write(img_bytes)

            asyncio.get_running_loop().run_in_executor(None, write_file)
        except Exception as e:
            logger.error(f"保存图片失败: {e}")

    async def _generate_image_core(
        self, image_bytes: bytes, prompt: str
    ) -> bytes | str | None:
        """
        核心生成逻辑，根据配置分发到不同的 API 处理函数
        """

        async def operation(api_key):
            if self.api_type == "openai":
                return await self._send_openai_request(
                    self.gemini_model, prompt, image_bytes, api_key
                )
            else:
                return await self._send_google_request(
                    self.gemini_model, prompt, image_bytes, api_key
                )

        image_data = await self._with_retry(operation)
        if not image_data:
            return "所有API密钥均尝试失败或模型未返回图片"
        return image_data

    def _get_current_key(self):
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def _switch_key(self):
        if not self.api_keys:
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"切换到下一个 API 密钥（索引：{self.current_key_index}）")

    async def _send_google_request(self, model_name, prompt, image_bytes, api_key):
        """
        Google Native API 请求逻辑
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": image_base64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }

        base_url = self.api_base_url.strip().removesuffix("/")
        endpoint = f"{base_url}/v1beta/models/{model_name}:generateContent"
        headers = {"Content-Type": "application/json"}

        # 使用 params 参数传递 api_key，aiohttp 会自动进行 URL 编码
        async with self.iwf.session.post(
            url=endpoint, json=payload, headers=headers, params={"key": api_key}
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.error(
                    f"Google API请求失败: HTTP {response.status}, 响应: {response_text}"
                )
                response.raise_for_status()
            data = await response.json()

        if (
            "candidates" in data
            and data["candidates"]
            and "content" in data["candidates"][0]
            and "parts" in data["candidates"][0]["content"]
        ):
            # Google API 直接返回 Base64
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    return base64.b64decode(part["inlineData"]["data"])

            logger.warning(
                f"Google 响应中未包含 inlineData。Raw: {data['candidates'][0]}"
            )
            raise Exception("Google API 未返回图片数据")

        raise Exception("Google API 响应结构异常")

    async def _send_openai_request(self, model_name, prompt, image_bytes, api_key):
        """
        OpenAI 兼容 API 请求逻辑 (Chat Completions with Vision)
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 构造 OpenAI Vision 格式的消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]

        payload = {"model": model_name, "messages": messages, "max_tokens": 4096}

        base_url = self.api_base_url.strip().removesuffix("/")
        # 处理 endpoint：有些中转直接是 base_url，有些需要加 v1/chat/completions
        if "/chat/completions" not in base_url:
            endpoint = f"{base_url}/v1/chat/completions"
        else:
            endpoint = base_url

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        async with self.iwf.session.post(
            url=endpoint, json=payload, headers=headers
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.error(
                    f"OpenAI API请求失败: HTTP {response.status}, 响应: {response_text}"
                )
                response.raise_for_status()
            data = await response.json()

        # 解析 OpenAI 响应
        try:
            content = data["choices"][0]["message"]["content"]
            logger.debug(f"OpenAI 响应内容: {content[:100]}...")

            img_matches = re.findall(r"!\[.*?\]\((.*?)\)", content)
            if img_matches:
                image_url = img_matches[0]
                
                if image_url.strip().startswith("data:"):
                    logger.info("从响应中提取到 Base64 图片数据，正在解码...")
                    try:
                        # data:image/png;base64,xxxxxx
                        if "," in image_url:
                            _, encoded_data = image_url.split(",", 1)
                            return base64.b64decode(encoded_data)
                        else:
                            logger.error("Data 格式无法识别，请提交完整日志到issue")
                            return None
                    except Exception as e:
                        logger.error(f"Base64 解码失败: {e}")
                        return None

                logger.info(f"从响应中提取到 Markdown 图片链接: {image_url}")
                return await self.iwf._download_image(image_url)

            # 这里做一个比较宽泛的匹配，然后尝试下载
            url_matches = re.findall(r"(https?://[^\s\)]+)", content)
            for url in url_matches:
                # 过滤掉一些极不可能的 URL
                if "api" in url or "openai" in url:
                    continue
                logger.info(f"尝试从响应中提取 URL 并下载: {url}")
                img_data = await self.iwf._download_image(url)
                if img_data:
                    return img_data
            # 如果直接是 Base64 编码的图片数据
            if self._is_base64(content):
                return base64.b64decode(content)

            logger.warning(f"OpenAI 响应中未找到图片 URL。内容: {content}")
            return f"API 返回了文本但未检测到图片链接: {content[:50]}..."

        except (KeyError, IndexError) as e:
            logger.error(f"解析 OpenAI 响应异常: {e}")
            raise Exception("OpenAI API 响应结构异常")

    def _is_base64(self, s: str) -> bool:
        try:
            return base64.b64encode(base64.b64decode(s)).decode() == s.replace("\n", "")
        except Exception:
            return False

    async def _with_retry(self, operation, *args, **kwargs):
        max_attempts = len(self.api_keys)
        if max_attempts == 0:
            return None

        for attempt in range(max_attempts):
            current_key = self._get_current_key()
            logger.info(
                f"尝试操作（API类型：{self.api_type}，密钥索引：{self.current_key_index}，次数：{attempt + 1}/{max_attempts}）"
            )
            try:
                return await operation(current_key, *args, **kwargs)
            except Exception as e:
                logger.error(f"第{attempt + 1}次尝试失败：{str(e)}")
                if attempt < max_attempts - 1:
                    self._switch_key()
                else:
                    logger.error("所有API密钥均尝试失败")
        return None

    async def terminate(self):
        if self.iwf:
            await self.iwf.terminate()
            logger.info("[figurine_workshop] session已关闭")

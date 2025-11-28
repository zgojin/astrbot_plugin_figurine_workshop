import asyncio
import base64
import io
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
            async with self.session.get(url) as resp:
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
        img_io = io.BytesIO(raw)
        img = PILImage.open(img_io)
        if img.format != "GIF":
            return raw
        logger.info("检测到GIF, 将抽取 GIF 的第一帧来生图")
        first_frame = img.convert("RGBA")
        out_io = io.BytesIO()
        first_frame.save(out_io, format="PNG")
        return out_io.getvalue()

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
        return await self._get_avatar(event.get_sender_id())

    async def terminate(self):
        if self.session and not self.session.closed:
            await self.session.close()


@register(
    "astrbot_plugin_figurine_workshop",
    "长安某",
    "使用 Gemini 2.5/3.0 进行图片风格化（手办/Galgame/Cos化/双V）",
    "1.3.4",
)
class LMArenaPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.save_image = config.get("save_image", False)
        self.plugin_data_dir = StarTools.get_data_dir(
            "astrbot_plugin_figurine_workshop"
        )
        self.api_keys = self.conf.get("gemini_api_keys", [])
        self.current_key_index = 0
        self.api_base_url = self.conf.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )
        # 默认模型
        self.gemini_model = self.conf.get("gemini_model", "gemini-2.5-flash-image")
        # 手办化的默认风格
        self.figurine_style = self.conf.get("figurine_style", "deluxe_box")
        
        if not self.api_keys:
            logger.error("LMArenaPlugin: 未配置任何 Gemini API 密钥")

    async def initialize(self):
        self.iwf = ImageWorkflow()

    # 修改点 1: 在正则中将 galgame 改为 !galgame
    @filter.regex(r"(?i)^(手办化|cos化|!galgame|双v)", priority=3)
    async def on_generate_request(self, event: AstrMessageEvent):
        """
        统一处理所有图片生成请求
        """
        # 1. 确定触发指令
        msg_str = event.message_obj.message_str.strip()
        
        # 修改点 2: 再次匹配提取分组时，也将 galgame 改为 !galgame
        trigger_match = re.match(r"(?i)^(手办化|!galgame|cos化|双v)", msg_str)
        if not trigger_match:
            return
        
        raw_command = trigger_match.group(1)
        # 统一转为小写进行逻辑判断
        command_lower = raw_command.lower()
        
        # 2. 获取图片
        img_bytes = await self.iwf.get_first_image(event)
        if not img_bytes:
            yield event.plain_result("缺少图片参数（可以发送图片或@用户）")
            return

        user_input_text = re.sub(
            r"(?i)^(手办化|!galgame|cos化|双v)\s*", "", msg_str, count=1
        ).strip()

        prompt_key = ""
        final_prompt = ""
        prompts_config = self.conf.get("prompts", {})

        if command_lower == "手办化":
            prompt_key = self.figurine_style # 使用配置中的默认手办风格
            base_prompt = prompts_config.get(prompt_key, "")
            style_display_name = f"手办化-{prompt_key}"

        # 判断条件匹配 !galgame
        elif command_lower == "!galgame":
            prompt_key = "galgame"
            base_prompt = prompts_config.get(prompt_key, "")
            style_display_name = "Galgame"

        elif command_lower == "cos化":
            prompt_key = "cosplay"
            base_prompt = prompts_config.get(prompt_key, "")
            style_display_name = "Cos化"
        
        elif command_lower == "双v":
            prompt_key = "double_v"
            base_prompt = prompts_config.get(prompt_key, "")
            style_display_name = "双V模式"

        else:
             return

        if not base_prompt:
             yield event.plain_result(f"配置错误：找不到 key 为 '{prompt_key}' 的提示词。")
             return

        # 统一将用户输入作为附加要求添加到提示词后
        final_prompt = (
            f"{base_prompt}\n\nAdditional user requirements: {user_input_text}"
            if user_input_text
            else base_prompt
        )

        yield event.plain_result("正在请求，请稍后...")
        
        logger.info(f"Gemini 生成 Prompt ({style_display_name}): {final_prompt[:100]}...") 

        res = await self._generate_with_gemini(img_bytes, final_prompt)

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
        save_path = (
            self.plugin_data_dir
            / f"gemini_{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        )
        def write_file():
            with save_path.open("wb") as f:
                f.write(img_bytes)
        asyncio.get_running_loop().run_in_executor(None, write_file)

    async def _generate_with_gemini(
        self, image_bytes: bytes, prompt: str
    ) -> bytes | str | None:
        
        async def edit_operation(api_key):
            model_name = self.gemini_model
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
            return await self._send_image_request(model_name, payload, api_key)

        image_data = await self._with_retry(edit_operation)
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
        logger.info(f"切换到下一个 Gemini API 密钥（索引：{self.current_key_index}）")

    async def _send_image_request(self, model_name, payload, api_key):
        base_url = self.api_base_url.strip().removesuffix("/")
        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        headers = {"Content-Type": "application/json"}

        async with self.iwf.session.post(
            url=endpoint, json=payload, headers=headers
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.error(
                    f"API请求失败: HTTP {response.status}, 响应: {response_text}"
                )
                response.raise_for_status()
            data = await response.json()

        if (
            "candidates" in data
            and data["candidates"]
            and "content" in data["candidates"][0]
            and "parts" in data["candidates"][0]["content"]
        ):
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    return base64.b64decode(part["inlineData"]["data"])
            
            logger.warning(f"Gemini 响应中未包含图片数据。Raw Parts: {data['candidates'][0]['content']['parts']}")

        raise Exception("操作成功，但未在响应中获取到图片数据")

    async def _with_retry(self, operation, *args, **kwargs):
        max_attempts = len(self.api_keys)
        if max_attempts == 0:
            return None

        for attempt in range(max_attempts):
            current_key = self._get_current_key()
            logger.info(
                f"尝试操作（密钥索引：{self.current_key_index}，次数：{attempt + 1}/{max_attempts}）"
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
            logger.info("[ImageWorkflow] session已关闭")

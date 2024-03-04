# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import logging

from langchain_core.messages import SystemMessage
from zhipuai import ZhipuAI
from ...utils.str_utils import remove_spaces_and_tabs
from ...utils.chat_message_utils import format_chat_text

logger = logging.getLogger(__name__)


class GLMGeneration:

    model = 'glm-4'

    client: ZhipuAI

    def __int__(self):
        from dotenv import load_dotenv
        load_dotenv()

        zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
        if zhipuai_api_key is None:
            logger.error("** env: ZHIPUAI_API_KEY is not set")

        self.client = ZhipuAI(api_key=zhipuai_api_key)

    def chat(self,
             prompt: str,
             role_name: str,
             you_name: str,
             query: str,
             short_history: list[dict[str, str]],
             long_history: str) -> str:

        # prompt = prompt + query
        logger.debug(f">> prompt: {prompt}\n>> query: {query}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )

        return response.choices[0].message.content

    async def chatStream(self,
                         prompt: str,
                         role_name: str,
                         you_name: str,
                         query: str,
                         history: list[dict[str, str]],
                         realtime_callback=None,
                         conversation_end_callback=None):
        logger.debug(f"prompt:{prompt}")
        messages = []
        messages.append(SystemMessage(content=prompt))
        # for item in history:
        #     message = HumanMessage(content=item["human"])
        #     messages.append(message)
        #     message = AIMessage(content=item["ai"])
        #     messages.append(message)
        # messages.append(HumanMessage(content=you_name + "说" + query))

        sys_prompt: str = prompt or "你是一个美食家，你会做很多治愈的美食"

        response = self.client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query},
            ],
            stream=True,
        )

        answer = ''
        for chunk in response:
            content: str = chunk.choices[0].delta.content
            content = remove_spaces_and_tabs(content)
            if content == "":
                continue
            answer += content
            if realtime_callback:
                realtime_callback(role_name, you_name, content, False)  # 调用实时消息推送的回调函数

        answer = format_chat_text(role_name, you_name, answer)
        if conversation_end_callback:
            conversation_end_callback(role_name, answer, you_name, query)  # 调用对话结束消息的回调函数

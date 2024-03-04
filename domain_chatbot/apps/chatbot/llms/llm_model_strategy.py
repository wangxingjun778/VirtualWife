from __future__ import annotations
from abc import ABC, abstractmethod
import threading
import asyncio
from .zhipu.glm_chat_robot import GLMGeneration
from .openai.openai_chat_robot import OpenAIGeneration
from .text_generation.text_generation_chat_robot import TextGeneration


class LlmModelStrategy(ABC):
    
    @abstractmethod
    def chat(self, prompt: str, role_name: str, you_name: str, query: str, short_history: list[dict[str, str]], long_history: str) -> str:
        pass

    @abstractmethod
    async def chatStream(self,
                         prompt: str,
                         role_name: str,
                         you_name: str,
                         query: str,
                         history: list[dict[str, str]],
                         realtime_callback=None,
                         conversation_end_callback=None):
        pass


# 定义策略类实现
class ZhipuAILlmModelStrategy(LlmModelStrategy):

    glm_generation: GLMGeneration

    def __init__(self) -> None:
        super().__init__()
        self.glm_generation = GLMGeneration()

    def chat(self, prompt: str, role_name: str, you_name: str, query: str, short_history: list[dict[str, str]], long_history: str) -> str:
        return self.glm_generation.chat(prompt=prompt,
                                        role_name=role_name,
                                        you_name=you_name,
                                        query=query,
                                        short_history=short_history,
                                        long_history=long_history)

    async def chatStream(self,
                         prompt: str,
                         role_name: str,
                         you_name: str,
                         query: str,
                         history: list[dict[str, str]],
                         realtime_callback=None,
                         conversation_end_callback=None):

        return await self.glm_generation.chatStream(prompt=prompt,
                                                    role_name=role_name,
                                                    you_name=you_name,
                                                    query=query,
                                                    history=history,
                                                    realtime_callback=realtime_callback,
                                                    conversation_end_callback=conversation_end_callback)


class OpenAILlmModelStrategy(LlmModelStrategy):

    openai_generation: OpenAIGeneration

    def __init__(self) -> None:
        super().__init__()
        self.openai_generation = OpenAIGeneration()

    def chat(self, prompt: str, role_name: str, you_name: str, query: str, short_history: list[dict[str, str]], long_history: str) -> str:
        return self.openai_generation.chat(prompt=prompt, role_name=role_name, you_name=you_name, query=query, short_history=short_history, long_history=long_history)

    async def chatStream(self,
                         prompt: str,
                         role_name: str,
                         you_name: str,
                         query: str,
                         history: list[dict[str, str]],
                         realtime_callback=None,
                         conversation_end_callback=None
                         ):
        return await self.openai_generation.chatStream(prompt=prompt,
                                                       role_name=role_name,
                                                       you_name=you_name,
                                                       query=query,
                                                       history=history,
                                                       realtime_callback=realtime_callback,
                                                       conversation_end_callback=conversation_end_callback)


class TextGenerationLlmModelStrategy(LlmModelStrategy):

    generation: TextGeneration

    def __init__(self) -> None:
        super().__init__()
        self.generation = TextGeneration()

    def chat(self, prompt: str, role_name: str, you_name: str, query: str, short_history: list[dict[str, str]], long_history: str) -> str:
        return self.generation.chat(prompt=prompt, role_name=role_name, you_name=you_name, query=query, short_history=short_history, long_history=long_history)

    async def chatStream(self,
                         prompt: str,
                         role_name: str,
                         you_name: str,
                         query: str,
                         history: list[dict[str, str]],
                         realtime_callback=None,
                         conversation_end_callback=None
                         ):
        return await self.generation.chatStream(prompt=prompt,
                                                role_name=role_name,
                                                you_name=you_name,
                                                query=query,
                                                history=history,
                                                realtime_callback=realtime_callback,
                                                conversation_end_callback=conversation_end_callback)


class LlmModelDriver:

    def __init__(self):
        self.glm_model_strategy = ZhipuAILlmModelStrategy()
        self.openai_model_strategy = OpenAILlmModelStrategy()
        self.text_generation_strategy = TextGenerationLlmModelStrategy()
        self.chat_stream_lock = threading.Lock()

    def chat(self, prompt: str, type: str, role_name: str, you_name: str, query: str, short_history: list[dict[str, str]], long_history: str) -> str:
        strategy = self.get_strategy(type)
        result = strategy.chat(prompt=prompt, role_name=role_name,
                               you_name=you_name, query=query, short_history=short_history, long_history=long_history)
        return result

    def chatStream(self,
                   prompt: str,
                   type: str,
                   role_name: str,
                   you_name: str,
                   query: str,
                   history: list[dict[str, str]],
                   realtime_callback=None,
                   conversation_end_callback=None):
        strategy = self.get_strategy(type)
        asyncio.run(strategy.chatStream(prompt=prompt,
                                        role_name=role_name,
                                        you_name=you_name,
                                        query=query,
                                        history=history,
                                        realtime_callback=realtime_callback,
                                        conversation_end_callback=conversation_end_callback))

    def get_strategy(self, type: str) -> LlmModelStrategy:
        if type == "zhipuai":
            return self.glm_model_strategy
        elif type == "openai":
            return self.openai_model_strategy
        elif type == "text_generation":
            return self.text_generation_strategy
        else:
            raise ValueError("Unknown type")

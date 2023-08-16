
from ..customrole.custom_role_generation import CustomRoleGeneration
from ..config import singleton_sys_config
import logging
import re


class ChatService():

    custom_role_generation: CustomRoleGeneration

    def __init__(self) -> None:

        # 加载自定义角色生成模块
        self.custom_role_generation = CustomRoleGeneration()

    def chat(self, role_name: str, you_name: str, query: str) -> str:

        # 生成角色prompt
        prompt = self.custom_role_generation.get_prompt(role_name)

        # 检索相关记忆
        short_history, long_history = singleton_sys_config.memory_storage_driver.search(
            query_text=query, owner=you_name)

        # 对话聊天
        prompt = prompt.format(input=query,you_name=you_name,short_history=short_history,long_history=long_history)
        answer_text = singleton_sys_config.llm_model_driver.chat(prompt=prompt, type=singleton_sys_config.conversation_llm_model_driver_type, role_name=role_name,
                                                      you_name=you_name, query=query, short_history=short_history, long_history=long_history)
        
        # 格式化输出结果
        answer_text = self.format_chat_text(
            role_name=role_name, you_name=you_name, text=answer_text).strip()

        logging.info(
            f'[BIZ] # ChatService.chat # role_name：{role_name} you_name：{you_name} query：{query} short_history：{short_history}  long_history:{long_history}# \n => answer_text：{answer_text}')

        if answer_text != "":
            # 保存记忆
            singleton_sys_config.memory_storage_driver.save(
                role_name=role_name, you_name=you_name, query_text=query, answer_text=answer_text)

        # 合成语音
        return answer_text

    def format_chat_text(self, role_name: str, you_name: str, text: str):
        # 去除特殊字符 * 、`role_name：`、`you_name:`
        # text = text.replace(f'*', "")
        pattern = r'\*.*?\*'
        text = re.sub(pattern, '', text)
        text = text.replace(f'{role_name}：', "")
        text = text.replace(f'{you_name}：', "")
        text = text.replace(f'{role_name}:', "")
        text = text.replace(f'{you_name}:', "")
        return text

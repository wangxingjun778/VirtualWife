import json
import logging
import os
from ..llms.llm_model_strategy import LlmModelDriver
from ..models import CustomRoleModel, SysConfigModel
from ..character.sys.aili_zh import aili_zh

config_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(config_dir, 'sys_config.json')
sys_code = "adminSettings"

logger = logging.getLogger(__name__)


def lazy_memory_storage(sys_config_json: any, sys_cofnig: any):
    from ..memory.memory_storage import MemoryStorageDriver
    # 加载记忆模块配置
    memory_storage_config = {
        "host": sys_config_json["memoryStorageConfig"]["milvusMemory"]["host"],
        "port": sys_config_json["memoryStorageConfig"]["milvusMemory"]["port"],
        "user": sys_config_json["memoryStorageConfig"]["milvusMemory"]["user"],
        "password": sys_config_json["memoryStorageConfig"]["milvusMemory"]["password"],
        "db_name": sys_config_json["memoryStorageConfig"]["milvusMemory"]["dbName"],
    }
    logger.debug(f"=> memory_storage_config:{memory_storage_config}")
    # 加载记忆模块驱动
    return MemoryStorageDriver(memory_storage_config=memory_storage_config, sys_config=sys_cofnig)


class SysConfig():

    llm_model_driver: LlmModelDriver
    conversation_llm_model_driver_type: str
    enable_summary: bool
    enable_longMemory: bool
    summary_llm_model_driver_type: str
    enable_reflection: bool
    reflection_llm_model_driver_type: str
    memory_storage_driver: any
    character: int
    your_name: str
    room_id: str
    local_memory_num: int = 5

    def __init__(self) -> None:
        self.load()

    def get(self):
        sys_config_obj = None
        sys_config_json = "{}"
        with open(config_path, 'r') as f:
            sys_config_json = json.load(f)
            logger.debug(f"\n>>>sys_config_json 1111: {json.dumps(sys_config_json, ensure_ascii=False)}\n")
        try:
            sys_config_obj = SysConfigModel.objects.filter(
                code=sys_code).first()
            logger.debug(f'>>>sys_config_obj: {sys_config_obj}')
            if sys_config_obj == None:
                logger.debug("=> save sys config to db")
                sys_config_model = SysConfigModel(
                    code=sys_code,
                    config=json.dumps(sys_config_json)
                )
                sys_config_model.save()
                logger.debug(f"\n>>>sys_config_json 2222: {json.dumps(sys_config_json, ensure_ascii=False)}\n")
            else:
                sys_config_json = json.loads(sys_config_obj.config)
                logger.debug(f"\n>>>sys_config_json 3333: {json.dumps(sys_config_json, ensure_ascii=False)}\n")
        except Exception as e:
            logger.debug("=> load sys config error: %s" % str(e))
        return sys_config_json

    def save(self, sys_config_json: any):
        sys_config_obj = SysConfigModel.objects.get(code=sys_code)
        sys_config_obj.config = json.dumps(sys_config_json)
        sys_config_obj.save()

    def load(self):

        logger.debug(
            "======================== Load SysConfig ========================")

        sys_config_json = self.get()

        # 初始化默认角色
        try:
            result = CustomRoleModel.objects.all()
            if len(result) == 0:
                logger.debug("=> load default character")
                custom_role = CustomRoleModel(
                    role_name=aili_zh.role_name,
                    persona=aili_zh.persona,
                    personality=aili_zh.personality,
                    scenario=aili_zh.scenario,
                    examples_of_dialogue=aili_zh.examples_of_dialogue,
                    custom_role_template_type=aili_zh.custom_role_template_type
                )
                custom_role.save()
        except Exception as e:
            logger.debug("=> load default character ERROR: %s" % str(e))

        # 加载角色配置
        character = sys_config_json["characterConfig"]["character"]
        yourName = sys_config_json["characterConfig"]["yourName"]
        logger.debug("=> character Config")
        logger.debug(f"character:{character}")
        logger.debug(f"yourName:{yourName}")
        self.character = character
        self.yourName = yourName

        # 加载大语言模型配置
        os.environ['ZHIPUAI_API_KEY'] = sys_config_json["languageModelConfig"]["zhipuai"]["ZHIPUAI_API_KEY"]
        os.environ['OPENAI_API_KEY'] = sys_config_json["languageModelConfig"]["openai"]["OPENAI_API_KEY"]
        os.environ['OPENAI_BASE_URL'] = sys_config_json["languageModelConfig"]["openai"]["OPENAI_BASE_URL"]
        os.environ['TEXT_GENERATION_API_URL'] = sys_config_json["languageModelConfig"]["textGeneration"]["TEXT_GENERATION_API_URL"]
        os.environ['TEXT_GENERATION_WEB_SOCKET_URL'] = sys_config_json["languageModelConfig"]["textGeneration"].get(
            "TEXT_GENERATION_WEB_SOCKET_URL", "ws://127.0.0.1:5005/api/v1/stream")

        # 是否开启proxy
        enableProxy = sys_config_json["enableProxy"]
        logger.debug("=> Proxy Config ")
        logger.debug(f"enableProxy:{enableProxy}")
        if enableProxy:
            os.environ['HTTP_PROXY'] = sys_config_json["httpProxy"]
            os.environ['HTTPS_PROXY'] = sys_config_json["httpsProxy"]
            os.environ['SOCKS5_PROXY'] = sys_config_json["socks5Proxy"]
            logger.debug(f"=> HTTP_PROXY:" + os.environ['HTTP_PROXY'])
            logger.debug(f"=> HTTPS_PROXY:"+os.environ['HTTPS_PROXY'])
            logger.debug(f"=> SOCKS5_PROXY:"+os.environ['SOCKS5_PROXY'])
        else:
            os.environ['HTTP_PROXY'] = ""
            os.environ['HTTPS_PROXY'] = ""
            os.environ['SOCKS5_PROXY'] = ""

        # 加载对话模块配置
        logger.debug("=> Chat Config")
        self.llm_model_driver = LlmModelDriver()
        self.conversation_llm_model_driver_type = sys_config_json[
            "conversationConfig"]["languageModel"]
        logger.debug(f"conversation_llm_model_driver_type:" +
                     self.conversation_llm_model_driver_type)

        # 是否开启记忆摘要
        logger.debug("=> Memory Config")
        self.enable_summary = sys_config_json["memoryStorageConfig"]["enableSummary"]
        self.enable_longMemory = sys_config_json["memoryStorageConfig"]["enableLongMemory"]
        logger.debug("=> enable_longMemory："+str(self.enable_longMemory))
        logger.debug("=> enable_summary："+str(self.enable_summary))
        if (self.enable_summary):
            self.summary_llm_model_driver_type = sys_config_json[
                "memoryStorageConfig"]["languageModelForSummary"]
            logger.debug("=> summary_llm_model_driver_type：" +
                         self.summary_llm_model_driver_type)

        self.enable_reflection = sys_config_json["memoryStorageConfig"]["enableReflection"]
        logger.debug("=> enableReflection："+str(self.enable_reflection))
        if (self.enable_reflection):
            self.reflection_llm_model_driver_type = sys_config_json[
                "memoryStorageConfig"]["languageModelForReflection"]
            logger.debug("=> reflection_llm_model_driver_type" +
                         self.summary_llm_model_driver_type)

        # 懒加载记忆模块
        try:
            self.memory_storage_driver = lazy_memory_storage(
                sys_config_json=sys_config_json, sys_cofnig=self)
        except Exception as e:
            logger.error("init memory_storage error: %s" % str(e))

        logger.info("=> Load SysConfig Success")

        # 加载直播配置
        # if self.bili_live_client != None:
        #     self.bili_live_client.stop()
        # room_id = str(sys_config_json["liveStreamingConfig"]["B_STATION_ID"])
        # print("=> liveStreaming Config")
        # self.room_id = room_id
        # self.bili_live_client = BiliLiveClient(room_id=room_id)
        # # 创建后台线程
        # background_thread = threading.Thread(
        #     target=asyncio.run(self.bili_live_client.start()))
        # # 将后台线程设置为守护线程，以便在主线程结束时自动退出
        # background_thread.daemon = True
        # # 启动后台线程
        # background_thread.start()

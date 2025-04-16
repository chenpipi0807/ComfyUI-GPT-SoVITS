WEB_DIRECTORY = "./web"

from .tts_node import TextDictNode, GSVTTSNode,TSCY_Node,PreViewSRT,LoadSRT
from .ft_node import AudioSlicerNode,ASRNode,DatasetNode,\
      ExperienceNode,GSFinetuneNone, ConfigSoVITSNode,ConfigGPTNode
from .node_display_names import NODE_DISPLAY_NAMES, CATEGORY_NAMES

NODE_CLASS_MAPPINGS = {
    "TSCY_Node":TSCY_Node,
    "ASRNode": ASRNode,
    "DatasetNode":DatasetNode,
    "ExperienceNode": ExperienceNode,
    "AudioSlicerNode": AudioSlicerNode,
    "GSFinetuneNone": GSFinetuneNone,
    "ConfigSoVITSNode":ConfigSoVITSNode,
    "ConfigGPTNode": ConfigGPTNode,
    "TextDictNode": TextDictNode,
    "GSVTTSNode": GSVTTSNode,
    "PreViewSRT":PreViewSRT,
    "LoadSRT":LoadSRT,
}

# 节点显示名称映射（英文 -> 中文）
NODE_DISPLAY_NAME_MAP = NODE_DISPLAY_NAMES

# 分类显示名称映射（英文 -> 中文）
CATEGORY_NAME_MAP = CATEGORY_NAMES
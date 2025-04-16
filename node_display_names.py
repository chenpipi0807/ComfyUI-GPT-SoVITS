"""
定义 GSTTS-ComfyUI 节点在 UI 中显示的中文名称
"""

NODE_DISPLAY_NAMES = {
    # tts_node.py 中的节点
    "TextDictNode": "文本字典节点",
    "GSVTTSNode": "GPT-SoVITS语音合成",
    "TSCY_Node": "文本转语音预览",
    "PreViewSRT": "SRT字幕预览",
    "LoadSRT": "加载SRT字幕",
    
    # ft_node.py 中的节点
    "AudioSlicerNode": "音频切片器",
    "ASRNode": "语音识别(ASR)",
    "DatasetNode": "数据集准备",
    "ExperienceNode": "实验配置",
    "GSFinetuneNone": "GPT-SoVITS微调",
    "ConfigSoVITSNode": "SoVITS配置",
    "ConfigGPTNode": "GPT配置",
}

# 分类名称翻译
CATEGORY_NAMES = {
    "AIFSH_GPT-SoVITS": "AI浮世绘_语音合成"
}

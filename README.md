# ComfyUI-GPT-SoVITS
一个基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的 ComfyUI 插件，用于高质量语音合成。示例工作流程可在[这里](./workflows)找到。

本项目是 [GSTTS-ComfyUI](https://github.com/AIFSH/GSTTS-ComfyUI) 的改进版，添加了许多优化和修复。

# Disclaimer  / 免责声明
We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.
我们不对代码库的任何非法使用承担任何责任. 请参阅您当地关于 DMCA (数字千年法案) 和其他相关法律法规.

## Windows一键包
[下载链接，及时转存](https://pan.quark.cn/s/aaadcbf3181f)

## How to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

然后！
```
git clone https://github.com/chenpipi0807/ComfyUI-GPT-SoVITS.git
cd ComfyUI-GPT-SoVITS
pip install -r requirements.txt
```
`weights` will be downloaded from huggingface automatically! if you in china,make sure your internet attach the huggingface
or if you still struggle with huggingface, you may try follow [hf-mirror](https://hf-mirror.com/) to config your env.

## Tutorial

- [Run Online](https://www.xiangongyun.com/image/detail/13706bf7-f3e6-4e29-bb97-c79405f5def4)
- [GPT-SoVITS V2的ComfyUI插件来啦！ | ComfyUI数字人之音色克隆篇-哔哩哔哩］(https://b23.tv/mhskKcZ)

## 改进内容

- 修复了 SoVITS 训练脚本中的导入错误和模块引用问题
- 优化了在 Windows 环境下的单 GPU 训练流程
- 增加了对 PyTorch 2.6+ 版本的兼容性支持
- 修复了模型加载时的 `HParams` 类访问问题
- 改进了错误提示和日志输出

## 鸣谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - 原始 GPT-SoVITS 项目
- [AIFSH/GSTTS-ComfyUI](https://github.com/AIFSH/GSTTS-ComfyUI) - 原始 ComfyUI 插件实现

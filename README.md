# Interview Copilot

Windows 本地优先的实时英文面试助手。它通过 WASAPI 回环捕获会议声音，使用
Qwen LiveTranslate 显示英文原文和中文翻译，再结合本地简历、JD、知识文档和历史记忆，
流式生成中英文回答建议。

## 快速开始

要求 Python 3.12+、Windows 10/11 和阿里云 Model Studio API Key。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m interview_copilot
```

首次启动后，在“设置”中填写：

- DashScope API Key
- Workspace ID
- 区域（北京或新加坡）

API Key 由 Windows Credential Manager 保存，不写入项目文件或 SQLite。

## 使用流程

1. 在知识库页面导入简历、岗位 JD 和项目材料。
2. 选择正在播放会议声音的 WASAPI loopback 设备。
3. 点击“开始”，英文原文与中文翻译会实时显示。
4. 每个问题结束后，应用检索本地资料并生成双语回答。

## 数据边界

文档原件、会话和向量索引保存在本机。音频、待向量化的文本以及命中的少量上下文会发送
到配置的 Qwen 云端 API。请遵守面试方的规则，并优先将本工具用于练习、语言辅助或经允许
的无障碍场景。

## 测试与打包

```powershell
pytest
ruff check .
.\scripts\build.ps1
```

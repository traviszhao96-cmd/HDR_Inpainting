# Qwen 视觉模型本地配置

复制示例文件：

```sh
cp config/qwen.env.example config/qwen.env
```

编辑 `config/qwen.env`，只填写自己的 Qwen/DashScope API Key。`run-local.sh` 启动后端时会自动读取它；Key 不会进入网页、结果目录或请求给 FLUX。当前配置只完成安全存储和后端读取，下一步再接入“分析局部切片并生成英文提示词”的接口。

可替换的配置项：

- `QWEN_VISION_API_KEY`：视觉模型 API Key
- `QWEN_VISION_BASE_URL`：OpenAI 兼容接口地址
- `QWEN_VISION_MODEL`：默认 `qwen3-vl-plus`
- `QWEN_VISION_MAX_PIXELS`：上传给识别模型前的最大像素数
- `QWEN_VISION_TIMEOUT`：请求超时秒数

验证是否读到配置：

```sh
curl http://127.0.0.1:7860/health
```

只检查 `qwen_vision_configured: true`，不会返回 Key。

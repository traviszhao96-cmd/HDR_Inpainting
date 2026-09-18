# 页面流程与五张样张测试

当前页面：圈选确认 → FLUX.1 Fill-dev Q4_K_S + ObjectRemovalFluxFill v2
完成 SDR 消除 → 单通道 gainmap 平滑插值 → UHDR 合成。
不启用 gainmap 生成或 ControlNet；普通 SDR 输入仅输出 SDR。

脚本 `backend/test_fill_pipeline.py` 调用页面同一个 `/erase` 接口，使用
已保存蒙版及原 mask_expand=8，当前无人背景英文提示词。
Fill 为 20 步，工作图长边 768，每次随机种子；本轮不是固定种子 A/B。 
下表是整条接口耗时，包括上传、生成、插值和编码。

|样张|耗时|输出|目视检查|
|---|---:|---|---|
|路灯 clean_input_uhdr|51.2s|Ultra HDR|失败：路灯主体变成建筑状内容；细杆仍有残留|
|寺庙路人 104440|60.4s|Ultra HDR|人物消除，但阴影残留、地面纹理接缝明显|
|雕塑前人物 104523|125.9s|Ultra HDR|目标人物消除，旁边白衣人边缘被误改|
|远处路人 104528|61.3s|SDR|效果较好，目标消除，缩略对比未见明显新增人物或接缝|
|夕阳建筑 104538|72.4s|Ultra HDR|高楼已去除，补成天空与低处轮廓；仍需放大检查边缘与补出的背景|

不能把接口成功视为质量通过。截图主要检查 SDR 消除；HDR 文件编码成功
不代表已在 HDR 显示器上验证亮度观感。

## 对比图

- [路灯](../test_runs/20260915-155255-fill-smooth-pipeline/clean_input_uhdr-comparison.png)
- [寺庙路人](../test_runs/20260915-155255-fill-smooth-pipeline/20260911-104440-comparison.png)
- [雕塑前人物](../test_runs/20260915-155255-fill-smooth-pipeline/20260911-104523-comparison.png)
- [远处路人](../test_runs/20260915-160040-fill-smooth-pipeline/20260911-104528-comparison.png)
- [夕阳建筑](../test_runs/20260915-160040-fill-smooth-pipeline/20260911-104538-comparison.png)

对应两个 test_runs 目录的 results.json 保存了来源、参数、job_id、SDR、
gainmap 及 HDR 文件链接。第四张首次读取 ComfyUI 结果超时，确认队列空闲后
重新测试成功。后端已增加在截止时间内重试读取同一个任务结果的逻辑，不会
因一次读取超时重新提交 GPU 任务。

## 验证与限制

Python 编译检查与 git diff --check 通过。浏览器实际页面已检查四步流程、
Fill 默认选项和快速 gainmap 说明。页面现有 oxlint 检查仍报 img 元素、
role 标签以及 fetch JSON 的 unknown 类型问题，本轮没有通过全量 lint。

下一步重点：路灯场景提示与遮罩、近邻人物保护、阴影覆盖。完整流程的
50–126s 也说明离线低分辨率 gainmap 的计算时间不能代表网页端到端耗时。

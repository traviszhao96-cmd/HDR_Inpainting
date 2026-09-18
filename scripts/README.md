# scripts

## libultrahdr-local-fixes.patch

`libultrahdr` 以 git 子模块指向上游 `google/libultrahdr`（无法 push）。本机对子模块
源码做了两处修复，导出为补丁文件，供任何新克隆（如 Windows 台式机）复现：

1. `examples/ultrahdr_app.cpp` —— EXIF 输入内存块长度未赋值（`mExifBlock.data_sz`/`capacity`）。
2. `lib/src/dsp/arm/gainmapmath_neon.cpp` —— P3/BT.2020 的 RGB→YUV 系数分支颠倒，
   并将 P3 JPEG 的 YCbCr 系数与标量实现的 BT.601 对齐。

### 应用方式（clone 后、构建前）

```bash
git submodule update --init
cd libultrahdr
git apply ../scripts/libultrahdr-local-fixes.patch
```

### 重新导出（本机改动后）

```bash
git -C libultrahdr diff > scripts/libultrahdr-local-fixes.patch
```

> 子模块内的工作区改动不会被父仓库提交捕获，只记录子模块的提交号，
> 因此必须靠这个补丁文件同步。
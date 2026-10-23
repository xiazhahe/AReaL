# 安装指南（AMD ROCm）

**注意**：AMD GPU 支持位于 `main` 分支。目前还没有发布 ROCm 镜像，需要按下文用
[`Dockerfile.rocm`](https://github.com/areal-project/AReaL/blob/main/Dockerfile.rocm)
自行构建。

## 前置要求

### 硬件要求

以下硬件配置已经过测试：

- **GPU**：每节点 8 张 AMD Instinct MI308X（gfx942），或每节点 8 张 AMD Instinct MI355X（gfx950）
- **节点**：仅测试了单节点，ROCm 上的多节点训练尚未测试

MI300X 和 MI325X 使用相同的 gfx942 架构，MI350X 使用相同的 gfx950 架构，但都尚未测试。MI355X
目前只测试了快速入门，见[已测试的配置](#rocm-tested-configurations)。

### 软件要求

| 组件          |                        版本                        |
| ------------- | :------------------------------------------------: |
| 操作系统      |       Ubuntu 24.04 或满足以下要求的任何系统        |
| AMD GPU 驱动  |              amdgpu 6.16.13（已测试）              |
| 主机上的 ROCm | 7.2.4 或 7.0.1（已测试；镜像自带 ROCm 用户态组件） |
| Docker        |                  29.1.3（已测试）                  |
| 基础镜像      |                `rocm/primus:v26.5`                 |

MI355X 的测试使用的是 Ubuntu 22.04，主机上的 ROCm 为 7.0.1。

**注意**：本教程不涵盖 AMD GPU 驱动和共享存储的安装。驱动安装请参考
[ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)。

## 运行环境

`Dockerfile.rocm` 基于 `rocm/primus:v26.5` 构建，并保留基础镜像中 ROCm 版的 PyTorch。镜像包含以下组件：

| 组件               |                   版本                    |
| ------------------ | :---------------------------------------: |
| PyTorch            |     2.12.0（ROCm 7.15，来自基础镜像）     |
| Transformer Engine |       2.15（ROCm 版，来自基础镜像）       |
| SGLang             | 0.5.10.post1，从源码构建（`sglang` 版本） |
| vLLM               |     0.19.1，从源码构建（`vllm` 版本）     |
| Megatron-Core      |               core_v0.19.0                |
| Transformers       |                   5.3.0                   |
| Ray                |                  2.59.0                   |

完整的 Python 依赖版本列表见
[`requirements-rocm.txt`](https://github.com/areal-project/AReaL/blob/main/requirements-rocm.txt)。

### 构建镜像

在 AReaL 仓库根目录下执行构建。镜像有两个版本，区别只在推理后端：

```bash
git clone https://github.com/areal-project/AReaL
cd AReaL

# MI350X/MI355X 请设为 gfx950
GPU_ARCH=gfx942

# SGLang 推理后端（默认）
docker build -f Dockerfile.rocm --build-arg GPU_ARCH=${GPU_ARCH} \
    -t areal-runtime:dev-rocm-sglang-${GPU_ARCH} .

# vLLM 推理后端
docker build -f Dockerfile.rocm --build-arg VARIANT=vllm --build-arg GPU_ARCH=${GPU_ARCH} \
    -t areal-runtime:dev-rocm-vllm-${GPU_ARCH} .
```

构建过程会从源码编译 SGLang 或 vLLM，需要一些时间。构建最后会检查 ROCm 版 PyTorch 没有被替换，并且推理后端可以正常导入。

**注意**：每个镜像只针对一种 GPU 架构，默认是 `gfx942`。`GPU_ARCH` 决定镜像里所有 kernel 的目标架构，两个版本都一样：包括 SGLang 的
kernel、vLLM 的 kernel，以及 aiter 在运行时编译的 kernel。所以为一种架构构建的镜像不能在另一种架构上运行，上面的镜像标签因此带上了架构名。

如果您的 Docker 没有安装 `buildx` 插件，请用 `DOCKER_BUILDKIT=0` 执行构建。这个 Dockerfile 没有用到 BuildKit
专有的功能。

构建过程中 aiter 会打印
`ROCm/HIP JIT runtime not available: No module named 'jax'. CK and HIP ops are disabled.`。这是因为构建时看不到
GPU，不影响镜像本身；容器在 GPU 上运行时 aiter 会正常加载。

### 创建容器

```bash
docker run -it --name areal-rocm \
   --device=/dev/kfd --device=/dev/dri --group-add video \
   --ipc=host --network host \
   --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
   -v /path/to/mount:/path/to/mount \
   areal-runtime:dev-rocm-sglang-gfx942 \
   /bin/bash
```

`/dev/kfd` 和 `/dev/dri` 用于把 GPU 暴露给容器。`--ipc=host` 让容器直接使用主机的共享内存，因此不需要 `--shm-size`。请保留
`--cap-add=SYS_PTRACE`：进程间共享显存要用到 `pidfd_getfd` 系统调用，缺少这项权限会报
`Operation not permitted`。使用 vLLM 版本时，把镜像标签换成 vLLM 的标签。

使用 `--network host` 时，AReaL 的服务会监听节点
IP，而不是回环地址。启动任务前请先看[必需的设置](#rocm-required-settings)。

**对于多节点训练**：请确保在每个节点上挂载共享存储路径，并挂载到容器中。此路径将用于保存检查点和日志。

### 使用自己的 AReaL 代码

镜像在 `/AReaL` 下安装了构建时所用的 AReaL 源码。如果要使用其他代码目录，在其中重新安装即可覆盖镜像自带的版本：

```bash
cd /path/to/mount/AReaL
pip install --no-deps --no-build-isolation -e .
```

请保留 `--no-deps`。`pyproject.toml` 中固定的是 CUDA 版 PyTorch，解析依赖会替换掉镜像中的 ROCm 版。

### 自定义环境安装

我们推荐使用镜像。如果要把 AReaL 安装到自己的 ROCm 环境中：

- 不要运行 `uv sync --extra cuda`。PyPI 上的 PyTorch、SGLang 和 vLLM 都是 CUDA 版，会覆盖 ROCm 安装。
- 按照 `Dockerfile.rocm` 逐步安装。它用 `--no-deps` 安装每个包，并基于已安装的 ROCm 版 PyTorch 编译 SGLang 和
  vLLM。
- 设置镜像通过 `ENV` 设置的环境变量：
  - 把 `GPU_ARCHS`、`PYTORCH_ROCM_ARCH` 和 `AMDGPU_TARGET` 设为您的 GPU 架构，例如 `gfx942`。aiter
    无法解析 `GPU_ARCHS=native`，会把 kernel 编译成错误的架构。
  - 按 Dockerfile 的顺序设置 `LD_LIBRARY_PATH`，让 ROCm SDK 的库排在最前面。如果加载了两份
    LLVM，`torch_memory_saver` 和 Transformer Engine 会报
    `Option spirv-expand-step registered more than once` 并退出。
  - 使用 Megatron 时设置 `NVTE_FLASH_ATTN=1` 和 `NVTE_FUSED_ATTN=1`。其中任何一个为 `0` 时，Megatron 的
    `attention_backend=auto` 会在启动时断言失败。
- 如果自己安装 SGLang，请在安装之后、开始训练之前执行一次下面的命令。它会给 SGLang 0.5.10.post1 打补丁，使在线权重更新后 MoE 权重仍保持
  aiter 要求的布局；遇到其他 SGLang 版本时会拒绝执行：

```bash
python -m areal.engine.sglang_aiter_moe_layout
```

(rocm-required-settings)=

## 必需的设置

有两项设置在 CUDA 上是可选的，在 ROCm 上是必需的。缺少它们，[下一步](#rocm-next-steps)中的命令会在启动时失败。

**SGLang 的 attention 后端。** AReaL 给 SGLang 的默认 `attention_backend` 是 `fa3`，而 ROCm 版的
`sgl-kernel` 不包含 FA3。SGLang 服务会在启动时报 `Can not import FA3 in sgl_kernel` 并退出。请改用 aiter：

```bash
+sglang.attention_backend=aiter
```

请保留开头的 `+`：示例 YAML 文件里没有 `attention_backend` 这个键，加号表示新增这个键。

**管理 API 密钥。** AReaL 的 HTTP 服务如果监听在非回环地址上，会拒绝使用默认的管理 API 密钥启动，并报
`Refusing to start server on non-loopback host ... default admin API key`。使用
`--network host` 时服务监听在节点地址上，但这个检查对所有非回环地址都生效。请设置一个唯一的密钥：示例配置使用
`rollout.agent.admin_api_key`，v2 控制器使用 `rollout.admin_api_key` 和
`actor.admin_api_key`。如果节点处在可信网络中，也可以用下面的方式确认接受风险：

```bash
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
```

## 环境变量

| 变量                              | 作用                                                                                   |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| `SGLANG_USE_AITER=1`              | 让 SGLang 使用 AMD 的 aiter kernel。下表中所有 SGLang 测试都设置了该变量。             |
| `VLLM_ROCM_USE_AITER=1`           | 让 vLLM 使用 aiter kernel。vLLM 默认不开启，下表中的 vLLM 测试保持了默认值。           |
| `VLLM_SERVER_DEV_MODE=1`          | 镜像已设置。vLLM 0.19.1 只在开发模式下提供共卡模式所需的 `/sleep` 和 `/wake_up` 接口。 |
| `AREAL_ALLOW_DEFAULT_ADMIN_KEY=1` | 允许 AReaL 的服务在非回环地址上使用默认的管理 API 密钥启动。只应在可信网络中使用。     |

开发模式还会在 vLLM 服务上暴露未鉴权的调试接口。请不要把 vLLM 端口暴露到不可信的网络。

AReaL 在 ROCm 上同样通过 `CUDA_VISIBLE_DEVICES` 选择 GPU。如果您的环境导出了 `HIP_VISIBLE_DEVICES` 或
`ROCR_VISIBLE_DEVICES`，请在启动前取消设置，确保只有一个变量控制设备可见性。

(rocm-tested-configurations)=

## 已测试的配置

以下测试使用 8 张 MI308X、`gsm8k_rl.py` 示例入口、DAPO-Math-17k 数据集、TP=1 和 xccl
权重更新，并应用了[必需的设置](#rocm-required-settings)，在 `main` 分支的 `18f5271f` 加上后来以 #1834 和 #1820
合入的 ROCm 改动上运行。

| 推理   | 训练     | 部署方式 | 模型            | 测试              | 结果                     |
| ------ | -------- | -------- | --------------- | ----------------- | ------------------------ |
| SGLang | FSDP2    | 分离部署 | Qwen2.5-0.5B    | smoke，3 步       | 通过                     |
| SGLang | FSDP2    | 分离部署 | Qwen2.5-Math-7B | 20 步             | 通过                     |
| SGLang | FSDP2    | 分离部署 | Qwen3-30B-A3B   | 10 步，稳定性测试 | 无 OOM、卡死或 RCCL 超时 |
| vLLM   | Megatron | 分离部署 | Qwen2.5-0.5B    | smoke，3 步       | 通过                     |
| vLLM   | Megatron | 分离部署 | Qwen2.5-Math-7B | 20 步             | 通过                     |

smoke 测试以学习率 0 训练，检查权重更新后推理端的权重与训练端逐位一致，并检查检查点的保存和恢复正常。20 步的测试检查训练端和推理端对相同 token
计算出相同的对数概率，但并不能证明收敛。

### MI355X

以下测试使用 8 张 MI355X，镜像用 `--build-arg GPU_ARCH=gfx950`
构建，运行的是[下一步](#rocm-next-steps)中的快速入门命令并加上
`+total_train_steps=5`：`gsm8k_grpo.yaml`、Qwen2.5-1.5B-Instruct、GSM8K 数据集、分离部署和 xccl
权重更新，在 `main` 分支的 `fa8bd789` 上运行。

| 推理   | 训练  | 部署方式 | 模型                  | 测试           | 结果         |
| ------ | ----- | -------- | --------------------- | -------------- | ------------ |
| SGLang | FSDP2 | 分离部署 | Qwen2.5-1.5B-Instruct | 快速入门，5 步 | 5 步全部完成 |
| vLLM   | FSDP2 | 分离部署 | Qwen2.5-1.5B-Instruct | 快速入门，5 步 | 5 步全部完成 |

每一步中，推理端和训练端对采样 token 计算出的平均负对数似然之差，SGLang 最大为 0.016，vLLM 最大为 0.085。这是每步的平均值，不是 20 步测试中逐
token 的检查。这些只是快速入门测试，上面的 smoke、20 步和稳定性测试尚未在 MI355X 上重复。

### 已知限制

- **不使用 AWEX 的共卡模式**（训练端和推理端通过 offload 共用 GPU）：在 ROCm 上的 `main` 分支中，SGLang 和 vLLM
  都会失败。问题出在与平台无关的代码中，修复在一个待合入的 PR 里。另外，vLLM 0.19.1 没有包含 vLLM PR
  #44972（唤醒时清零重新映射的显存），vLLM 的 sleep/wake 只在 amdgpu 6.16.13 上测试过。
- **FSDP 训练 MoE 模型并通过 xccl 更新权重**：在 Transformers 5 下，FSDP
  发送的是合并后的专家张量，其名称会被推理引擎跳过，因此推理端一直使用初始的专家权重。这不是 ROCm 特有的问题。修复在一个待合入的 PR 里；在此之前，MoE 模型请使用
  Megatron，或使用 `weight_update_mode: disk`。上面的 30B 稳定性测试只检查崩溃和显存问题，因此不受影响。
- **开启 expandable segments 时跨进程共享显存**：在 `rocm/primus:v26.5` 中，用
  `expandable_segments:True` 分配的张量无法被另一个进程打开，导入端会报 `CUDA error: invalid argument`。这是
  ROCm 运行时（ROCr）的回归，在 MI308X 和 MI355X 上都已复现。AReaL 默认不开启 expandable
  segments，因此上面的测试不受影响。部分示例配置会通过 `env_vars` 为某个角色开启它，例如
  `examples/math/qwen3_5_27b_grpo_awex_colocate.yaml` 中的
  `PYTORCH_ALLOC_CONF: "expandable_segments:True"`。在 ROCm
  上，`PYTORCH_ALLOC_CONF`、`PYTORCH_CUDA_ALLOC_CONF` 和 `PYTORCH_HIP_ALLOC_CONF`
  都会生效。如果某个进程要通过 CUDA/HIP IPC 把 GPU 张量共享给其他进程，请不要通过其中任何一个变量在该进程上开启它。
- 量化 MoE 模型、TP > 1 和多节点训练尚未在 ROCm 上测试。
- AReaL 没有 AMD 的 CI，因此这些结果不会在每次代码变更时自动检查。

## 问题排查

- **失败后残留进程。** rollout 服务启动失败时，AReaL 会打印 `Deleting ... workers for role rollout`，但它启动的
  `areal.infra.rpc.rpc_server` 进程可能仍然存活，等待其输出的 shell 或管道也不会返回。请用
  `ps aux | grep areal.infra.rpc.rpc_server` 检查，并在重试前结束这些进程。
- **aiter 报某个 `~/.aiter/build/` 下的 kernel `cannot open shared object file`。** 该 kernel
  之前的编译失败了，例如 `GPU_ARCHS` 设置错误，或者 home 目录在 gfx942 和 gfx950 机器之间共享，而 aiter
  会一直加载这个坏掉的目录。请删除 `~/.aiter/build/` 下该 kernel 的目录后重新运行。

(rocm-next-steps)=

## 下一步

请查看[快速入门部分](quickstart.md)来启动您的第一个 AReaL 任务。在 ROCm
上，请加上[必需的设置](#rocm-required-settings)。例如，在 SGLang 容器中：

```bash
export SGLANG_USE_AITER=1
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1  # 仅限可信网络；或者设置 admin_api_key
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    +sglang.attention_backend=aiter
```

在 vLLM 容器中，改为切换推理后端：

```bash
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1  # 仅限可信网络；或者设置 admin_api_key
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    rollout.backend=vllm:d4p1t1
```

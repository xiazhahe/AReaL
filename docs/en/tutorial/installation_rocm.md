# Installation (AMD ROCm)

**Note**: AMD GPU support lives on `main`. There is no published ROCm image yet, so you
build one from
[`Dockerfile.rocm`](https://github.com/areal-project/AReaL/blob/main/Dockerfile.rocm) as
described below.

## Prerequisites

### Hardware Requirements

The following hardware configurations have been tested:

- **GPU**: 8x AMD Instinct MI308X (gfx942) or 8x AMD Instinct MI355X (gfx950) per node
- **Nodes**: single node only; multi-node training has not been tested on ROCm

MI300X and MI325X use the same gfx942 architecture, and MI350X the same gfx950
architecture, but they have not been tested. MI355X has been tested with the quickstart
only; see [Tested Configurations](#rocm-tested-configurations).

### Software Requirements

| Component        |                             Version                              |
| ---------------- | :--------------------------------------------------------------: |
| Operating System |    Ubuntu 24.04 or any system meeting the requirements below     |
| AMD GPU driver   |                     amdgpu 6.16.13 (tested)                      |
| ROCm on the host | 7.2.4 or 7.0.1 (tested; the image ships its own ROCm user space) |
| Docker           |                         29.1.3 (tested)                          |
| Base image       |                       `rocm/primus:v26.5`                        |

The MI355X runs used Ubuntu 22.04 with ROCm 7.0.1 on the host.

**Note**: This tutorial does not cover the installation of the AMD GPU driver or of
shared storage. See the
[ROCm installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
for the driver.

## Runtime Environment

`Dockerfile.rocm` builds on `rocm/primus:v26.5` and keeps the base image's ROCm build of
PyTorch. It ships the following stack:

| Component          |                      Version                       |
| ------------------ | :------------------------------------------------: |
| PyTorch            |      2.12.0 (ROCm 7.15, from the base image)       |
| Transformer Engine |       2.15 (ROCm build, from the base image)       |
| SGLang             | 0.5.10.post1, built from source (`sglang` variant) |
| vLLM               |     0.19.1, built from source (`vllm` variant)     |
| Megatron-Core      |                    core_v0.19.0                    |
| Transformers       |                       5.3.0                        |
| Ray                |                       2.59.0                       |

The full list of pinned Python packages is in
[`requirements-rocm.txt`](https://github.com/areal-project/AReaL/blob/main/requirements-rocm.txt).

### Build the Image

Run the build from the root of the AReaL repository. The image has two variants that
differ only in the inference backend:

```bash
git clone https://github.com/areal-project/AReaL
cd AReaL

# Set to gfx950 for MI350X/MI355X
GPU_ARCH=gfx942

# SGLang inference backend (default)
docker build -f Dockerfile.rocm --build-arg GPU_ARCH=${GPU_ARCH} \
    -t areal-runtime:dev-rocm-sglang-${GPU_ARCH} .

# vLLM inference backend
docker build -f Dockerfile.rocm --build-arg VARIANT=vllm --build-arg GPU_ARCH=${GPU_ARCH} \
    -t areal-runtime:dev-rocm-vllm-${GPU_ARCH} .
```

The build compiles SGLang or vLLM from source, which takes a while. It ends with a check
that the ROCm build of PyTorch is still in place and that the inference backend imports.

**Note**: Each image targets one GPU architecture, `gfx942` by default. `GPU_ARCH` sets
the target for every kernel the image compiles, in both variants: the SGLang kernels,
the vLLM kernels and the kernels aiter compiles at runtime. An image built for one
architecture therefore does not run on the other, which is why the tags above carry the
architecture.

If your Docker installation has no `buildx` plugin, run the build with
`DOCKER_BUILDKIT=0`. The Dockerfile does not use BuildKit-only features.

During the build, aiter prints
`ROCm/HIP JIT runtime not available: No module named 'jax'. CK and HIP ops are disabled.`
This happens because no GPU is visible during the build. It does not affect the image,
and aiter loads normally when the container runs on a GPU.

### Create Container

```bash
docker run -it --name areal-rocm \
   --device=/dev/kfd --device=/dev/dri --group-add video \
   --ipc=host --network host \
   --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
   -v /path/to/mount:/path/to/mount \
   areal-runtime:dev-rocm-sglang-gfx942 \
   /bin/bash
```

`/dev/kfd` and `/dev/dri` expose the GPUs to the container. `--ipc=host` gives the
container the host's shared memory, so `--shm-size` is not needed. Keep
`--cap-add=SYS_PTRACE`: sharing GPU memory between processes uses the `pidfd_getfd`
system call, which fails with `Operation not permitted` without it. Replace the image
tag with the vLLM one for the vLLM variant.

With `--network host`, AReaL's servers listen on the node's IP address rather than on
loopback. See [Required Settings](#rocm-required-settings) before launching a job.

**For multi-node training**: Ensure a shared storage path is mounted on every node (and
mounted to the container). This path will be used to save checkpoints and logs.

### Use Your Own AReaL Checkout

The image installs AReaL from the source tree it was built from, under `/AReaL`. To work
on another checkout, install it over the built-in copy:

```bash
cd /path/to/mount/AReaL
pip install --no-deps --no-build-isolation -e .
```

Keep `--no-deps`. `pyproject.toml` pins a CUDA build of PyTorch, and resolving it would
replace the ROCm build in the image.

### Custom Environment Installation

We recommend the image. If you install AReaL into your own ROCm environment instead:

- Do not run `uv sync --extra cuda`. The PyTorch, SGLang and vLLM wheels on PyPI are
  CUDA-only and replace a ROCm installation.
- Follow `Dockerfile.rocm` step by step. It installs every package with `--no-deps` and
  compiles SGLang and vLLM against the installed ROCm PyTorch.
- Set the environment variables the image sets with `ENV`:
  - `GPU_ARCHS`, `PYTORCH_ROCM_ARCH` and `AMDGPU_TARGET` to your GPU architecture, for
    example `gfx942`. aiter cannot resolve `GPU_ARCHS=native` and then compiles its
    kernels for the wrong architecture.
  - `LD_LIBRARY_PATH` with the ROCm SDK libraries first, as in the Dockerfile. With two
    copies of LLVM loaded, `torch_memory_saver` and Transformer Engine abort with
    `Option spirv-expand-step registered more than once`.
  - `NVTE_FLASH_ATTN=1` and `NVTE_FUSED_ATTN=1` for Megatron. With either set to `0`,
    Megatron's `attention_backend=auto` fails an assertion at startup.
- If you install SGLang yourself, run the following once after installing it and before
  starting training. It patches SGLang 0.5.10.post1 so that online weight updates keep
  the MoE weight layout aiter expects, and it refuses any other SGLang version:

```bash
python -m areal.engine.sglang_aiter_moe_layout
```

(rocm-required-settings)=

## Required Settings

Two settings that are optional on CUDA are required on ROCm. Without them, the commands
in [Next Steps](#rocm-next-steps) fail at startup.

**SGLang attention backend.** AReaL's default SGLang `attention_backend` is `fa3`, and
the ROCm build of `sgl-kernel` does not include FA3. The SGLang server then exits at
startup with `Can not import FA3 in sgl_kernel`. Select aiter instead:

```bash
+sglang.attention_backend=aiter
```

Keep the leading `+`: the example YAML files have no `attention_backend` key, so the
override adds one.

**Admin API key.** AReaL refuses to start its HTTP services with the default admin API
key when they listen on a non-loopback address, and stops with
`Refusing to start server on non-loopback host ... default admin API key`. With
`--network host` they listen on the node's address, but the check applies to any
non-loopback address. Set a unique key instead: the example configs use
`rollout.agent.admin_api_key`, and the v2 controllers read `rollout.admin_api_key` and
`actor.admin_api_key`. On a trusted network you can acknowledge the risk instead:

```bash
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
```

## Environment Variables

| Variable                          | Purpose                                                                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_USE_AITER=1`              | Use AMD's aiter kernels in SGLang. All SGLang runs in the tables below set it.                                          |
| `VLLM_ROCM_USE_AITER=1`           | Use aiter kernels in vLLM. vLLM leaves it off by default, and the vLLM runs below kept that default.                    |
| `VLLM_SERVER_DEV_MODE=1`          | Set by the image. vLLM 0.19.1 serves `/sleep` and `/wake_up`, which colocation uses, only in dev mode.                  |
| `AREAL_ALLOW_DEFAULT_ADMIN_KEY=1` | Lets AReaL's services start with the default admin API key on a non-loopback address. Use it only on a trusted network. |

Dev mode also exposes unauthenticated debug endpoints on the vLLM server. Do not publish
the vLLM port to untrusted networks.

AReaL selects GPUs through `CUDA_VISIBLE_DEVICES` on ROCm as well. If your environment
exports `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`, unset them before launching so
that only one variable controls device visibility.

(rocm-tested-configurations)=

## Tested Configurations

These runs used 8x MI308X, the `gsm8k_rl.py` example entry point, the DAPO-Math-17k
dataset, TP=1 and xccl weight updates, with the
[required settings](#rocm-required-settings) applied. They were run on `main` at
`18f5271f` with the ROCm changes that later merged as #1834 and #1820.

| Inference | Training | Placement  | Model           | Run                 | Result                       |
| --------- | -------- | ---------- | --------------- | ------------------- | ---------------------------- |
| SGLang    | FSDP2    | separation | Qwen2.5-0.5B    | smoke, 3 steps      | passed                       |
| SGLang    | FSDP2    | separation | Qwen2.5-Math-7B | 20 steps            | passed                       |
| SGLang    | FSDP2    | separation | Qwen3-30B-A3B   | 10 steps, stability | no OOM, hang or RCCL timeout |
| vLLM      | Megatron | separation | Qwen2.5-0.5B    | smoke, 3 steps      | passed                       |
| vLLM      | Megatron | separation | Qwen2.5-Math-7B | 20 steps            | passed                       |

A smoke run trains at learning rate 0 and checks that the rollout's weights after the
update are bit-identical to the trainer's and that a checkpoint save and resume work.
The 20-step runs check that the trainer and the rollout compute the same log
probabilities for the same tokens. They do not prove convergence.

### MI355X

These runs used 8x MI355X, images built with `--build-arg GPU_ARCH=gfx950`, and the
quickstart in [Next Steps](#rocm-next-steps) with `+total_train_steps=5`:
`gsm8k_grpo.yaml`, Qwen2.5-1.5B-Instruct, the GSM8K dataset, separation and xccl weight
updates. They were run on `main` at `fa8bd789`.

| Inference | Training | Placement  | Model                 | Run                 | Result                 |
| --------- | -------- | ---------- | --------------------- | ------------------- | ---------------------- |
| SGLang    | FSDP2    | separation | Qwen2.5-1.5B-Instruct | quickstart, 5 steps | 5 of 5 steps completed |
| vLLM      | FSDP2    | separation | Qwen2.5-1.5B-Instruct | quickstart, 5 steps | 5 of 5 steps completed |

In every step, the mean negative log-likelihood of the sampled tokens differed between
the rollout and the trainer by at most 0.016 with SGLang and 0.085 with vLLM. This is a
per-step average, not the per-token check of the 20-step runs. These are quickstart runs
only: the smoke, 20-step and stability runs above have not been repeated on MI355X.

### Known Limitations

- **Colocation without AWEX** (trainer and rollout sharing GPUs through offload) fails
  on `main` on ROCm, with both SGLang and vLLM. The causes are in platform-independent
  code, and the fixes are in a pending pull request. vLLM 0.19.1 also lacks vLLM PR
  #44972, which zeroes remapped memory on wake-up; vLLM sleep and wake have only been
  tested with amdgpu 6.16.13.
- **FSDP with MoE models over xccl**: with Transformers 5, FSDP sends the fused expert
  tensors under names the rollout engines skip, so the rollout keeps its initial expert
  weights. This is not specific to ROCm. A fix is in a pending pull request; until then,
  use Megatron for MoE models or `weight_update_mode: disk`. The 30B stability run above
  only checks for crashes and memory problems, so it is not affected by this.
- **GPU memory sharing between processes with expandable segments**: in
  `rocm/primus:v26.5`, a tensor allocated with `expandable_segments:True` cannot be
  opened by another process. The importing process fails with
  `CUDA error: invalid argument`. This is a ROCm runtime (ROCr) regression and was
  reproduced on both MI308X and MI355X. AReaL does not enable expandable segments by
  default, so the runs above are not affected. Some example configs enable it per role
  through `env_vars`, for example `PYTORCH_ALLOC_CONF: "expandable_segments:True"` in
  `examples/math/qwen3_5_27b_grpo_awex_colocate.yaml`. On ROCm, `PYTORCH_ALLOC_CONF`,
  `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_HIP_ALLOC_CONF` all take effect, so do not
  enable it through any of them on a process that exports GPU tensors to another process
  through CUDA/HIP IPC.
- Quantized MoE models, TP > 1 and multi-node training have not been tested on ROCm.
- There is no AMD CI for AReaL, so these results are not checked on every change.

## Troubleshooting

- **A failed run leaves processes behind.** When a rollout server fails to start, AReaL
  logs `Deleting ... workers for role rollout`, but the `areal.infra.rpc.rpc_server`
  processes it started can stay alive, and a shell or pipe that waits on their output
  does not return. Check for them with `ps aux | grep areal.infra.rpc.rpc_server` and
  stop them before retrying.
- **aiter reports `cannot open shared object file` for a kernel under
  `~/.aiter/build/`.** A previous build of that kernel failed, for example with the
  wrong `GPU_ARCHS` or in a home directory shared between gfx942 and gfx950 machines,
  and aiter keeps loading the broken directory. Delete the kernel's directory under
  `~/.aiter/build/` and run again.

(rocm-next-steps)=

## Next Steps

Check the [quickstart section](quickstart.md) to launch your first AReaL job. On ROCm,
add the [required settings](#rocm-required-settings). For example, inside the SGLang
container:

```bash
export SGLANG_USE_AITER=1
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1  # trusted network only; or set an admin_api_key
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    +sglang.attention_backend=aiter
```

Inside the vLLM container, switch the rollout backend instead:

```bash
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1  # trusted network only; or set an admin_api_key
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    rollout.backend=vllm:d4p1t1
```

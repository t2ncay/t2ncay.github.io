---
layout: post
title: "Reducing Neural Network VRAM by 94%: Fixing PyTorch's Memory and Determinism Problems"
date: 2026-11-13 10:00:00+0400
inline: false
related_posts: true
tags: compiler cuda edge-ai pytorch
categories: systems
---

Over the last few months, I’ve been attacking a specific bottleneck in Edge AI: the fact that modern ML frameworks are too heavy to train on constrained hardware, and they give slightly different results every time you run them.

I recently benchmarked two iterations of the training graph in my compiler, **Vyne**, using an ESPCN Super-Resolution model on an NVIDIA RTX 5060.

Here is what I found.

---

### 📉 1. The Memory Problem: The 94% Internal Drop

In an earlier version of Vyne (V1), the compiler naively called `cudaMalloc` inside the training loop for every iteration, causing massive VRAM churn and fragmentation. By upgrading the compiler to use a scoped `@device` region (V2B), I preallocated the entire forward and backward graph exactly once.

Here is what the explicit allocation looks like in Vyne. Everything from the weights to the AdamW optimizer moments is allocated exactly once at the top of the region:

```vyne
@device<cuda> region train {
    # Weights
    d_w1 :: Int64 := device_alloc(w1f.size());
    d_b1 :: Int64 := device_alloc(model.c1);
    
    # Preallocated gradients (No dynamic allocation in the loop!)
    d_gw1 :: Int64 := device_alloc(w1f.size());
    d_gb1 :: Int64 := device_alloc(model.c1);
    
    # Preallocated AdamW optimizer moments (m and v)
    d_m_w1 :: Int64 := device_alloc(w1f.size());
    d_v_w1 :: Int64 := device_alloc(w1f.size());
    
    # ... training loop runs here, reusing these buffers ...
};
```

* **Vyne V1 (Dynamic Allocation):** ~11.2 MB peak VRAM
* **Vyne V2B (Preallocated Graph):** **0.65 MB peak VRAM**

That is a **94% reduction** within my own compiler's architecture.

**But how does this compare to PyTorch?**
To be clear, PyTorch doesn't use raw `cudaMalloc` per iteration either; it uses a sophisticated caching allocator. However, PyTorch still pays a massive fixed "framework tax." If you run this exact workload in PyTorch, the peak VRAM usage is typically **300 MB to 700 MB**.

Why? Before PyTorch even loads your model weights, it initializes the CUDA context and dynamically loads vendor libraries like cuDNN and cuBLAS into VRAM. For massive LLMs on datacenter GPUs, this fixed overhead is negligible. But for Edge AI deploying a 0.65 MB model, paying a 500 MB framework tax is impossible.

Vyne avoids this tax entirely by bypassing the heavy runtime and using the low-level CUDA Driver API to JIT-compile only the exact PTX kernels needed.

### 🔬 2. The Determinism Problem: Bit-Exact Reproducibility

If you run a PyTorch CUDA training script twice, the weights will be slightly different. This is because default PyTorch backward passes use `atomicAdd` operations, which introduce non-deterministic floating-point rounding order. (PyTorch does offer a deterministic mode via `torch.use_deterministic_algorithms(True)`, but it forces the framework to abandon its fastest vendor kernels, resulting in a severe performance penalty).

To fix this without sacrificing speed, Vyne uses a **"gather" formulation** for the backward convolution kernels. Instead of many threads throwing their math into one shared bucket (scatter/atomicAdd), one thread owns one output element and loops sequentially over its inputs.

When Vyne compiles the CUDA PTX, it injects it directly into the C binary. At runtime, Vyne loads the driver dynamically and JIT-compiles it:

```c
// Inside the Vyne runtime (vdevice.h)
// We don't link against CUDA. We dynamically load the driver.
VyneCudaLib lib = LoadLibraryA("nvcuda.dll");
cuModuleLoadDataEx(&module, VYNE_CUDA_PTX, 0, NULL, NULL);
cuModuleGetFunction(&kernel, module, "conv2d_backward_input_f32");

// Launch the deterministic gather kernel
cuLaunchKernel(kernel, grid, block, ...);
```

I ran my exact same Vyne training script twice. The `train_loss.csv` and `val_metrics.csv` matched **byte-for-byte**. No atomics. No drift. You get the exact same weights every single time.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/cuda_benchmark.png" title="Vyne CUDA Training and Inference Output" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Terminal output showing the explicit CUDA training loop (left) and inference metrics (right). Note the 0.66 MB device peak during training, identical CPU/GPU PSNR (33.1963 dB), and a max numerical difference of just 2.38e-07.
</div>

*(Note on the 62ms GPU inference time: This includes the one-time JIT compilation of the PTX string at runtime. For a tiny 64x64 image with 1x1 kernels, the math is so small that JIT and PCIe launch overhead dominate. The goal of this benchmark was to prove memory safety and determinism, not raw throughput. For large models, this overhead is amortized to zero.)*

### 🏗️ Under the Hood

*(To be fully transparent, I architected this system and used AI to accelerate the boilerplate generation. The real engineering challenge was designing the explicit memory model, debugging the PTX kernels, and verifying the numerical parity down to 2e-7).*

The Vyne compiler bridges the gap between high-level numerical computing and hardware execution. By generating a self-contained C file with embedded PTX, the compiled binary doesn't depend on heavy frameworks or vendor libraries. It dynamically links to the CUDA driver at runtime and executes only the exact math needed for the graph.

### 🔗 Read the Code

I am currently planning to write a paper on deterministic, region-based static allocation for GPU training. If you are working on ML Systems, Compilers, or Edge AI, I’d love to connect.

You can view the explicit determinism guarantees and benchmarks in the repository here:
[Vyne ESPCN Expected Metrics](https://github.com/t2ncay/vyne/blob/master/examples/projects/espcn/EXPECTED.md)

`#CompilerDesign` `#CUDA` `#MachineLearning` `#EdgeAI` `#HPC`

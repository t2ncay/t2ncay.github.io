---
layout: page
title: Vyne Compiler & Edge AI
description: A region-based systems language for deterministic GPU training and bare-metal deployment.
img: assets/img/vyne_cover.jpg
importance: 1
category: work
---

Vyne is a statically-analyzed systems programming language and multi-target compiler designed to bridge the gap between high-level numerical computing and low-level hardware execution.

Instead of relying on a runtime garbage collector, Vyne uses **region-based memory management** to enforce deterministic execution and strict compile-time memory bounds. A single Vyne AST compiles down to x86 C, CUDA PTX for GPU acceleration, and bare-metal C/C++ for AVR and ESP32 microcontrollers.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/vyne_terminal.png" title="CUDA Training Output" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/vyne_esp32.png" title="Wi-Fi Radar on ESP32" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: Vyne's deterministic CUDA training pipeline. Right: A 42-line Vyne script turning an ESP32 into a live Wi-Fi radar.
</div>

### The Core Innovations

**Deterministic GPU Training**
Standard frameworks like PyTorch rely on `atomicAdd` operations in their backward passes, which introduce floating-point non-determinism. Vyne bypasses this by generating custom CUDA PTX kernels using a "gather" formulation (one thread per output element). This guarantees **bit-exact reproducibility** across runs—a critical requirement for safety-critical AI.

**Massive VRAM Reduction**
By using explicit `@device` regions, Vyne preallocates the entire training graph. In our ESPCN super-resolution benchmarks, Vyne reduced peak VRAM usage from **11.2 MB to 0.65 MB (a 94% reduction)** compared to dynamic allocation methods.

**Unified Hardware Deployment**
The exact same language handles high-level neural network training and low-level hardware interrupts. The ESP32 and AVR backends include native bindings for Wi-Fi, SPI displays, I2C, GPIO, and real-time processing.

### Code Example: Bare-Metal Hardware

Because Vyne compiles directly to the target, hardware interaction is concise and memory-safe. Here is the core loop of the ESP32 Wi-Fi radar:

```vyne
use native esp32;

esp32.display_init(1);
networks :: Array = esp32.wifi_scan();
count :: Int64 = networks.size();

esp32.display_text(10, 10, "Found " + string(count) + " networks", 0x07E0, 2);

# Render the list to the screen
y :: Int64 = 40;
through i :: 0..count-1 -> loop {
    if y < esp32.display_height() - 20 {
        esp32.display_text(10, y, networks[i], 0xFFFF, 1);
        y = y + 15;
    };
};
```

### Under the Hood

The Vyne compiler is written in C++ and features:

- **Source-to-Source Autodiff:** Generates static C functions for the backward pass, eliminating the need for a runtime tape.
- **Native ABI:** Unboxes primitives and typed arrays at call sites, achieving performance close to hand-written C.
- **Dynamic Hardware Presets:** Uses `hardware.toml` to automatically inject compiler flags for specific SPI displays and microcontrollers without the user needing to touch C++ headers.

*The project is open-source and available on [GitHub](https://github.com/t2ncay/vyne).*

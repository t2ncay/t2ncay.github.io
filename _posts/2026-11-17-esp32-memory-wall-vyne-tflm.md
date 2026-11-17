---
layout: post
title: "The ESP32 Memory Wall: What Bounded Vyne Actually Saves—and What TFLM Wins"
date: 2026-11-17 05:00:00+0400
inline: false
related_posts: true
tags: compiler embedded esp32 tinyml systems
categories: systems
---

When deploying neural networks onto microcontrollers like the classic ESP32, the bottleneck rarely starts with raw FLOPs—it starts with static SRAM. You are working inside a few hundred kilobytes shared across your network buffers, runtime metadata, task stacks, and system heap.

To test how my compiler, **Vyne**, holds up against the industry standard, I ran a falsifiable, apples-to-apples hardware benchmark on a classic ESP32 (240 MHz, PSRAM disabled, Wi-Fi/BT off) against **TensorFlow Lite for Microcontrollers (TFLM)**.

Instead of cherry-picking a workload where Vyne wins everything, I wanted to measure the exact trade-offs: where static region-based scratch saves real memory, what happens when a compiler accidentally falls back to boxed soft-float arithmetic, and where TFLM still wins on raw execution speed.

---

### 🧱 1. The Wall and a Falsifiable Experiment

To avoid classification accuracy or dataset quirks muddying the systems metrics, both runtimes execute the exact same **Float32 `64 → 32 → 10` MLP** (ReLU hidden activation, linear output logits, 2,410 parameters / 9,640 bytes of flash-resident weights).

* **Identical Weights & Validation:** Trained deterministically for 5 epochs on seed `20261117`, exported to `.vwt` for Vyne and converted via an equivalent Keras graph into a TFLite FlatBuffer. Both runtimes must pass numerical verification across all 8 test vectors (`atol=2e-5, rtol=2e-4`) before a single timing sample is accepted.
* **Strict Timing Protocol:** After 10 warmup passes, each cold-reset run executes 20 batches of 100 inferences timed via the 64-bit `esp_timer_get_time()` hardware counter—including input copy, math, and 10 volatile logit stores to prevent dead-code elimination.
* **Fail-Closed Harness:** If `AllocateTensors()` fails, outputs drift, or a capture drops batches, the runner rejects the run and logs the failure rather than publishing partial numbers.

---

### ⚙️ 2. Two Fixed Buffers, Different Orchestration

Neither runtime uses `malloc` inside the inference loop—both avoid conventional per-inference heap fragmentation—but they manage static memory differently:

* **TFLM (Manual Arena + Runtime Planner):** You supply a statically allocated byte array (`tensor_arena`), and `AllocateTensors()` plans activation reuse and stores persistent interpreter metadata inside it. For this MLP, TFLM's planner requires **904 bytes** of arena space. Give it `1,024` bytes (`sized` profile) and it runs cleanly; give it `896` bytes and `AllocateTensors()` fails deterministically on the device.
* **Vyne (`@bounded` Lexical Scratch):** Instead of an interpreter planner, Vyne allocates activation buffers (`64 + 32 + 10` floats = **424 bytes**) directly on the stack inside a lexically scoped `@bounded` region. The compiler enforces a conservative static budget (`@bounded<848>`, currently accounting at 8 bytes/element) and rejects over-budget scratch at compile time (`VNE-101`).

**The Explicit Integration Cost:**
To keep this honest, Vyne is not yet a standalone `.vwt` microcontroller framework. Because Vyne lacks embedded VFS loading and native C FFI bindings today, the benchmark runner emits ESP32-targeted C code and swaps in checked flash-weight accessors and telemetry stubs (verifying via SHA-256 that the generated `vyne_setup()` inference body remains byte-for-byte untouched).

---

### 📊 3. The Numbers (Including the Inconvenient Results)

Before fixing a critical codegen bottleneck in Vyne, my first hardware runs hit a wall—not in memory, but in latency.

In the initial baseline (`caa738e`), Vyne's multiply-accumulate expression:

```vyne
acc = float32(acc + float32(input[k] * bench_weight1(j * 64 + k)));
```

lost its native `Float32` type information across `float32(...)` conversion boundaries, falling back to generic boxed dispatchers and `Float32 ↔ Float64` software emulation helpers (`__extendsfdf2` / `__truncdfsf2`). That dragged Vyne down to **3.10 ms/inference** and kept its unused 64 KiB runtime arena linked into `.bss`.

After updating `BuiltInCallNode` and `codegen/builtins.cpp` to materialize native `Float32` temporaries directly into the compiler's fast path, the Xtensa disassembly (`vyne-setup.asm`) confirmed pure hardware `mul.s` and `add.s` instructions—and the dead arena dropped out of the linked ELF image completely (**0 linked arena bytes**).

Here are the cold-reset hardware results across 20,000 timed inferences (10 accepted runs, order-reversed between passes):

| Profile | Runtime | Configured / Linked Arena | Arena Used / Scratch | Static DRAM (`.data`/`.bss`) | Flash Image | Median Latency (min–max) |
| --- | --- | --- | --- | --- | --- | --- |
| **Default** | TFLM | 16,384 B / 16,384 B | 904 B / 0 B | 39,060 B | 322,272 B | **0.077040 ms** (0.07699–0.07706) |
| **Default** | Vyne | 65,536 B / **0 B** | 0 B / **424 B** | **22,324 B** | **283,424 B** | 0.132350 ms (0.13234–0.13239) |
| **Sized** | TFLM | 1,024 B / 1,024 B | 904 B / 0 B | 23,700 B | 322,272 B | **0.077040 ms** (0.07699–0.07709) |
| **Sized** | Vyne | 1,024 B / **0 B** | 0 B / **424 B** | **22,324 B** | **283,424 B** | 0.132350 ms (0.13234–0.13239) |

*(Note: Total device peak SRAM is not a single counter on ESP-IDF; static `.data`/`.bss`, task-stack high-water marks—7,036 B free of 8,192 B on Vyne vs. 6,688–6,720 B on TFLM—and internal heap telemetry are tracked separately without double-counting.)*

**What the data actually says:**

1. **23.4× Internal Speedup:** Eliminating boxed/double conversions cut Vyne's latency from `3.101680 ms` down to **`0.132350 ms`** (a **95.7% latency reduction**).
2. **TFLM Still Wins on Latency (1.72× Faster):** Even with native hardware `mul.s`/`add.s` instructions, TFLM finishes in **`0.077040 ms`**—about **55 microseconds faster** per inference. TFLM's fused `FullyConnected` kernels and vendor-tuned inner loops still beat Vyne's straightforward emitted C loops and flash-accessor calls.
3. **Vyne Wins on Linked Footprint:** Without interpreter metadata or a persistent tensor arena, Vyne saves **1,376 B of static DRAM (~5.8%)** even against a tightly sized 1 KiB TFLM build (and **16,736 B / ~42.9%** against default arena reservations), while shrinking the Flash binary by **38,848 B (~12.1%)**.
4. **Honest Engineering Effort:** Counting only the core inference loop (`52` LOC in Vyne vs. `62` LOC in TFLM) would be misleading—once you include the data bridge stubs (`11` LOC), MCU Float32 compatibility header (`9` LOC), and staging machinery (`401` LOC), Vyne currently requires more surrounding plumbing than dropping in TFLM.

---

### 🚀 4. What I Would Ship—and What I Need to Fix Next

If I were shipping a Float32 or Int8 microcontroller product **today**, I would still ship **TFLM**. It has mature operator coverage, battle-tested CMSIS-NN/ESP-NN SIMD kernels, and a 1.72× latency lead on this workload—and once you properly size its arena using `arena_used_bytes()`, its static RAM overhead is modest.

Where **Vyne's** model proves compelling is ultra-tight Flash/DRAM budgets where carrying a ~322 KB interpreter runtime isn't viable, and where compile-time region budgets (`@bounded`) prevent silent memory creep before flashing the board.

To close the remaining **55 µs gap** and turn this prototype into a standalone embedded ML target, three things are next on my compiler roadmap:

* **Loop Unrolling & Direct Flash Pointer Lowering:** Eliminating per-element accessor overhead and unrolling inner dot-product accumulators to hide Xtensa FPU pipeline latency.
* **Tighter Static Scratch Accounting:** Refining `@bounded` static analysis from conservative 8-byte slots down to exact `Float32` width (`424 B` instead of `848 B`).
* **Int8 Quantization & Hardware Energy Profiling:** Measuring quantized kernels against ESP-NN alongside real power/current draw and long-running multi-region allocation stress tests.

### 🔗 Reproduce the Benchmark

All capture logs, ELF section tables, disassembly audits (`vyne-setup.asm`), and automated failure tests are reproducible directly from the repository:
[Vyne ESP32 Memory Benchmark (`esp32_memory_bench`)](https://github.com/t2ncay/vyne)

`#CompilerDesign` `#ESP32` `#TinyML` `#EmbeddedSystems` `#EdgeAI`

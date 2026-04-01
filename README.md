# AccelSim: Neural Accelerator Runtime Simulator

A modular, extensible simulator for neural network execution on hardware accelerators. AccelSim enables performance analysis and bottleneck detection for accelerator-based inference and training workloads.

## Project Overview

**What is AccelSim?**

AccelSim is a lightweight runtime simulator designed to model the behavior of specialized neural network accelerators (TPUs, GPUs, IPUs, etc.). It provides:

- A minimal **Tensor Intermediate Representation (IR)** for expressing neural network operations
- **Instruction lowering** from high-level operations to accelerator instructions
- **Cycle-accurate simulation** with configurable memory and compute models
- **Performance metrics** including throughput, memory bandwidth, and buffer utilization
- **Bottleneck detection** with actionable optimization suggestions
- **Visualization tools** for understanding execution behavior

**Why Accelerator Simulation Matters**

Understanding accelerator performance is critical for:
- **Algorithm optimization**: Identifying compute vs. memory bottlenecks
- **Hardware-algorithm co-design**: Matching workload characteristics to accelerator capabilities
- **Performance prediction**: Estimating execution time before deployment
- **Design space exploration**: Evaluating different architectural choices

AccelSim provides a simple yet realistic model for these analyses.

## Architecture

AccelSim implements a complete end-to-end pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Program                          │
│            (PyTorch model, TensorFlow graph, etc.)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Tensor IR Layer                         │
│         (High-level operation representation)              │
│    TensorOp(LINEAR, RELU, ADD, MATMUL, ...)               │
└──────────────────────┬──────────────────────────────────────┘
                       │ ir/tensor_ir.py
┌──────────────────────▼──────────────────────────────────────┐
│                 Code Generation                            │
│         (IR → Accelerator Instructions)                    │
│    LOAD, MATMUL, ADD, RELU, STORE                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ backend/codegen.py
┌──────────────────────▼──────────────────────────────────────┐
│              Accelerator Runtime                           │
│  ┌──────────────────┬──────────────────────────────────┐  │
│  │   Scheduler      │      Memory System              │  │
│  │ (Latency model)  │ (Buffer capacity, traffic)     │  │
│  └──────────────────┴──────────────────────────────────┘  │
│         runtime/scheduler.py + runtime/memory_system.py   │
└──────────────────────┬──────────────────────────────────────┘
                       │ runtime/simulator.py
┌──────────────────────▼──────────────────────────────────────┐
│              Performance Analysis                          │
│  ┌──────────────────┬──────────────────────────────────┐  │
│  │ Metrics          │ Bottleneck Detection             │  │
│  │ (cycles, memory) │ (compute/memory bound, buffer)  │  │
│  └──────────────────┴──────────────────────────────────┘  │
│  analysis/performance_analyzer.py + bottleneck_detector.py │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Visualization                             │
│     (Timeline plots, memory usage graphs)                  │
│   visualization/timeline_plot.py + memory_plot.py          │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

- **Tensor IR**: Minimal, composable representation of neural network operations
- **Instruction Lowering**: Automatic compilation from high-level ops to low-level instructions
- **Accelerator Simulator**: Sequential instruction execution with latency modeling
- **Memory System**: Realistic on-chip buffer management with load/store tracking
- **Performance Metrics**: Compute utilization, memory bandwidth, throughput, latency
- **Bottleneck Detection**: Automatic identification of compute-bound, memory-bound, or buffer-limited workloads
- **Visualization**: Gantt-style timeline and memory usage plots

## Instruction Set

The simulator supports five core instructions:

| Instruction | Purpose | Latency | Usage |
|-------------|---------|---------|-------|
| **LOAD** | Load tensor data into on-chip buffers | 1 cycle | Data movement |
| **MATMUL** | Matrix multiplication | 10 cycles | Compute |
| **ADD** | Element-wise addition | 2 cycles | Compute |
| **RELU** | Rectified linear activation | 1 cycle | Compute |
| **STORE** | Write results back to main memory | 1 cycle | Data movement |

## Example: 2-Layer MLP

The default example demonstrates a simple multi-layer perceptron:

```
Input (1024) → Linear (512) → ReLU → Linear (10) → Output
```

**Execution Summary:**
- 9 instructions generated (5 LOAD, 2 MATMUL, 1 RELU, 1 STORE)
- 27 total cycles (21 compute, 6 memory)
- 2.1 MB memory traffic
- 50.6% buffer utilization

**Performance Analysis:**
```
Total Cycles:        27
Compute Cycles:      21 (77.8%)
Memory Cycles:       6  (22.2%)
Idle Cycles:         0  (0.0%)

Avg Latency:         3.00 cycles/instruction
Throughput:          0.33 instr/cycle

Memory Traffic:      2,123,856 bytes
Memory Bandwidth:    353,976 bytes/cycle
Buffer Utilization:  50.6%
```

**Bottleneck Detected:**
```
⚠ [MEDIUM] Compute Bound
  Compute operations dominate (77.8% of cycles).
  Recommendation: Consider loop tiling, parallelization, or compute unit scaling.
```

## Visualization: Example Outputs

### Execution Timeline
The timeline plot shows instruction execution order and duration:
- X-axis: Cycle count
- Y-axis: Instructions
- Each bar represents one instruction execution
- Color indicates operation type (LOAD, MATMUL, RELU, etc.)

This helps visualize:
- **Critical path**: Which operations take the longest
- **Instruction scheduling**: Execution ordering and dependencies
- **Parallelism opportunities**: Where operations could run concurrently

### Memory Usage
The memory plot tracks buffer utilization over time:
- **Top graph**: Buffer occupancy during execution with capacity limit
- **Bottom graph**: Per-instruction memory traffic (loads and stores)

This reveals:
- **Memory hotspots**: Instructions with high memory traffic
- **Buffer efficiency**: Peak vs. average buffer usage
- **Load-store balance**: Read vs. write patterns

## Repository Structure

```
AccelSim/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── main.py                      # Entry point
│
├── ir/                          # Intermediate Representation
│   ├── instructions.py          # Low-level instruction definitions
│   └── tensor_ir.py             # High-level tensor operation IR
│
├── backend/                     # Code Generation
│   └── codegen.py               # IR lowering to instructions
│
├── runtime/                     # Execution Engine
│   ├── simulator.py             # Main simulator orchestrator
│   ├── scheduler.py             # Instruction scheduler
│   ├── memory_system.py         # Buffer management
│   └── stats.py                 # Statistics collection
│
├── analysis/                    # Performance Analysis
│   ├── performance_analyzer.py  # Metrics computation
│   └── bottleneck_detector.py   # Bottleneck identification
│
├── visualization/               # Plotting & Visualization
│   ├── timeline_plot.py         # Execution timeline Gantt chart
│   └── memory_plot.py           # Memory usage visualization
│
├── examples/                    # Example Programs
│   └── mlp_program.py           # 2-layer MLP example
│
├── utils/                       # Utilities
│   └── pretty_print.py          # Formatted output helpers
│
├── tests/                       # Test Suite
│   └── test_simulator.py        # Comprehensive unit tests
│
└── results/                     # Generated Outputs
    ├── timeline.png             # Execution timeline plot
    └── memory_usage.png         # Memory usage plot
```

### Key Modules

**ir/instructions.py**: Defines the instruction set with latency and memory traffic models
- `OpCode`: Enum for instruction types
- `Instruction`: Dataclass representing a single instruction

**ir/tensor_ir.py**: High-level tensor operation representation
- `TensorOp`: Single operation (LINEAR, RELU, etc.)
- `TensorIR`: Container for a computation graph

**backend/codegen.py**: Lowers TensorIR to instructions
- `Codegen`: Compiler with `compile()` method

**runtime/simulator.py**: Orchestrates execution
- `Simulator`: Main execution engine with memory and scheduler

**runtime/scheduler.py**: Models execution timing
- `Scheduler`: Sequential instruction execution with latencies
- `ExecutionEntry`: Execution record with cycle counts

**runtime/memory_system.py**: Models on-chip buffer
- `MemorySystem`: Buffer capacity, load/store tracking
- `MemoryStats`: Memory statistics

**analysis/performance_analyzer.py**: Computes performance metrics
- `PerformanceAnalyzer`: Metrics like utilization, bandwidth, latency

**analysis/bottleneck_detector.py**: Identifies performance issues
- `BottleneckDetector`: Detects compute-bound, memory-bound, buffer-limited patterns

**visualization/timeline_plot.py**: Gantt-style execution timeline
**visualization/memory_plot.py**: Memory usage over time

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AccelSim

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Default Example

```bash
python main.py
```

This will:
1. Build a 2-layer MLP in TensorIR
2. Lower it to 9 accelerator instructions
3. Simulate execution (27 cycles)
4. Compute performance metrics
5. Detect bottlenecks
6. Generate visualization plots

**Output files:**
- Console output: Performance summary and bottleneck analysis
- `results/timeline.png`: Execution timeline (Gantt chart)
- `results/memory_usage.png`: Memory usage visualization

### Running Tests

```bash
pytest tests/test_simulator.py -v
```

Currently 15 tests covering:
- Instruction creation and properties
- Codegen (simple ops and complex MLP)
- Scheduler timing
- Memory system (load, store, peak tracking)
- End-to-end simulation
- Performance analysis metrics
- Bottleneck detection heuristics

All tests should pass with output like:
```
======================== 15 passed in 0.02s ========================
```

## Creating Custom Workloads

You can extend AccelSim to simulate custom neural networks:

```python
from ir.tensor_ir import TensorIR, TensorOp, TensorOpType
from backend.codegen import Codegen
from runtime.simulator import Simulator

# Build TensorIR
ir = TensorIR()
ir.set_input_shapes({"input": (1, 512), "weight": (512, 256)})

op = TensorOp(
    op_type=TensorOpType.MATMUL,
    input_shapes={"input": (1, 512), "weight": (512, 256)},
    output_shape=(1, 256),
    name="output"
)
ir.add_op(op)
ir.set_output_shapes({"output": (1, 256)})

# Lower to instructions
codegen = Codegen()
instructions = codegen.compile(ir)

# Simulate
simulator = Simulator(buffer_capacity=4 * 1024 * 1024)
stats = simulator.simulate(instructions)

# Analyze
print(f"Total cycles: {stats.total_cycles}")
print(f"Memory traffic: {stats.total_memory_traffic} bytes")
```

## Design Decisions

**Why a Sequential Scheduler?**
AccelSim uses sequential execution to keep the baseline simple. Future work could add pipelining and parallel execution.

**Why Constant Latencies?**
Fixed latencies per opcode provide predictable models. Real accelerators may use variable latencies based on data sizes and resource conflicts.

**Why Simple Memory Model?**
A single on-chip buffer with load/store tracking is sufficient for initial analysis. Future work can add cache hierarchies and prefetching.

## Future Work

Potential extensions and improvements:

- **Parallel Scheduling**: Model multi-core or systolic array execution
- **Memory Hierarchy**: Add L1/L2/L3 caches with realistic latency models
- **Pipelining**: Instruction-level parallelism with hazard detection
- **Advanced Workloads**: CNN, RNN, Transformer workloads
- **Data Reuse Analysis**: Roofline model and memory access patterns
- **Energy Modeling**: Power consumption in addition to cycles
- **Hardware Backend**: Integrate with real hardware validators
- **Distributed Training**: Multi-device execution and communication

## Contributing

Contributions are welcome! Areas of interest:
- Additional instruction types (convolution, pooling, normalization)
- New workload types (CNNs, RNNs, attention layers)
- Enhanced performance models
- Additional analysis tools
- Documentation improvements

## Reference

**Key Concepts:**
- **Accelerator**: Specialized hardware for neural network computation
- **Tensor IR**: Intermediate representation separating frontends from backends
- **Instruction Lowering**: Compilation from high-level operations to low-level instructions
- **Cycle-Accurate Simulation**: Simulating execution cycle-by-cycle
- **Bottleneck Analysis**: Identifying which resource limits performance

**Related Work:**
- NVIDIA's CUTLASS (GPU tensor operations)
- Google's XLA Compiler (hardware-agnostic compilation)
- ARM Compute Library
- Glow (LinkedIn's compiler)

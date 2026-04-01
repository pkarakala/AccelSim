"""
Example program: Simple 2-layer MLP

Architecture:
    Input (1024) → Linear (512) → RELU → Linear (10) → Output

This demonstrates:
1. Building a TensorIR program
2. Lowering to accelerator instructions
3. Simulating execution
4. Analyzing performance
5. Detecting bottlenecks
6. Generating visualizations
"""

from ir.tensor_ir import TensorIR, TensorOp, TensorOpType
from backend.codegen import Codegen
from runtime.simulator import Simulator
from analysis.performance_analyzer import PerformanceAnalyzer
from analysis.bottleneck_detector import BottleneckDetector
from visualization.timeline_plot import plot_timeline
from visualization.memory_plot import plot_memory_usage
from utils.pretty_print import (
    print_tensor_ir_summary,
    print_instruction_stream,
    print_execution_timeline,
)


def build_mlp_ir() -> TensorIR:
    """
    Build a 2-layer MLP as TensorIR.
    
    Network structure:
        x (1024) → Linear (512) → ReLU → Linear (10) → y
    
    Tensors:
        input: shape (1, 1024)
        w1: weight matrix (1024, 512)
        b1: bias (512,)
        h1: hidden activations (1, 512)
        h1_relu: ReLU output (1, 512)
        w2: weight matrix (512, 10)
        b2: bias (10,)
        output: final output (1, 10)
    """
    ir = TensorIR()
    
    # Define input shapes
    ir.set_input_shapes({
        "input": (1, 1024),
        "w1": (1024, 512),
        "b1": (512,),
        "w2": (512, 10),
        "b2": (10,),
    })
    
    # Operation 1: First linear layer (y = x @ w1 + b1)
    # Input shape: (1, 1024), Weight: (1024, 512) -> Output: (1, 512)
    op1 = TensorOp(
        op_type=TensorOpType.LINEAR,
        input_shapes={
            "input": (1, 1024),
            "w1": (1024, 512),
            "b1": (512,),
        },
        output_shape=(1, 512),
        name="h1",
        attributes={"has_bias": True},
    )
    ir.add_op(op1)
    
    # Operation 2: ReLU activation
    op2 = TensorOp(
        op_type=TensorOpType.RELU,
        input_shapes={"h1": (1, 512)},
        output_shape=(1, 512),
        name="h1_relu",
        attributes={},
    )
    ir.add_op(op2)
    
    # Operation 3: Second linear layer (y = h1_relu @ w2 + b2)
    # Input shape: (1, 512), Weight: (512, 10) -> Output: (1, 10)
    op3 = TensorOp(
        op_type=TensorOpType.LINEAR,
        input_shapes={
            "h1_relu": (1, 512),
            "w2": (512, 10),
            "b2": (10,),
        },
        output_shape=(1, 10),
        name="output",
        attributes={"has_bias": True},
    )
    ir.add_op(op3)
    
    # Define output shapes
    ir.set_output_shapes({
        "output": (1, 10),
    })
    
    return ir


def run_example():
    """Run the MLP example end-to-end."""
    print("\n" + "=" * 80)
    print("AccelSim: Neural Accelerator Runtime Simulator")
    print("=" * 80)
    
    # Build IR
    print("\n[1] Building Tensor IR for 2-layer MLP...")
    ir = build_mlp_ir()
    print("✓ IR constructed")
    print_tensor_ir_summary(ir.summary())
    
    # Codegen
    print("\n[2] Lowering IR to accelerator instructions...")
    codegen = Codegen()
    instructions = codegen.compile(ir)
    print(f"✓ Generated {len(instructions)} instructions")
    print_instruction_stream(instructions)
    
    # Simulate
    print("\n[3] Simulating execution...")
    simulator = Simulator(buffer_capacity=4 * 1024 * 1024)  # 4 MB buffer
    stats = simulator.simulate(instructions)
    print("✓ Simulation complete")
    print_execution_timeline(simulator.get_execution_timeline())
    
    # Results
    print("\n[4] Results:")
    print(stats)
    
    # Performance Analysis
    print("\n[5] Analyzing performance...")
    analyzer = PerformanceAnalyzer(stats, simulator.get_execution_timeline())
    print(analyzer.summary())
    
    # Bottleneck Detection
    print("\n[6] Detecting bottlenecks...")
    metrics = analyzer.analyze()
    detector = BottleneckDetector(metrics)
    print(detector.detect())
    
    # Visualization
    print("\n[7] Generating visualizations...")
    try:
        plot_timeline(simulator.get_execution_timeline(), output_path="results/timeline.png")
        plot_memory_usage(simulator.get_execution_timeline(), 
                         buffer_capacity=4 * 1024 * 1024,
                         output_path="results/memory_usage.png")
        print("✓ Visualizations saved to results/")
    except Exception as e:
        print(f"⚠ Visualization skipped: {e}")


if __name__ == "__main__":
    run_example()

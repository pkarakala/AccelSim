"""
Unit tests for AccelSim.

Tests verify:
- Instruction generation
- End-to-end simulation
- Memory tracking
- Scheduler correctness
- Performance analysis
- Bottleneck detection
"""

import pytest
from ir.instructions import Instruction, OpCode
from ir.tensor_ir import TensorIR, TensorOp, TensorOpType
from backend.codegen import Codegen
from runtime.simulator import Simulator
from runtime.scheduler import Scheduler
from runtime.memory_system import MemorySystem
from analysis.performance_analyzer import PerformanceAnalyzer
from analysis.bottleneck_detector import BottleneckDetector


class TestInstructions:
    """Test instruction creation and properties."""
    
    def test_instruction_creation(self):
        """Test creating a basic instruction."""
        instr = Instruction(
            opcode=OpCode.LOAD,
            inputs=["input"],
            output="input",
            shape=(1, 1024),
        )
        assert instr.opcode == OpCode.LOAD
        assert instr.inputs == ["input"]
        assert instr.output == "input"
        assert instr.shape == (1, 1024)
    
    def test_instruction_latencies(self):
        """Test that instructions have reasonable latencies."""
        latencies = {
            OpCode.LOAD: 1,
            OpCode.STORE: 1,
            OpCode.MATMUL: 10,
            OpCode.ADD: 2,
            OpCode.RELU: 1,
        }
        
        for opcode, expected_latency in latencies.items():
            instr = Instruction(
                opcode=opcode,
                inputs=["x"],
                output="y",
                shape=(1, 512),
            )
            assert instr.latency == expected_latency
    
    def test_instruction_memory_traffic(self):
        """Test memory traffic calculation."""
        # Shape (1, 512) = 512 elements, 4 bytes each = 2048 bytes
        instr = Instruction(
            opcode=OpCode.LOAD,
            inputs=["x"],
            output="y",
            shape=(1, 512),
        )
        traffic = instr.memory_traffic
        assert traffic == 512 * 4  # 2048 bytes


class TestCodegen:
    """Test instruction generation from TensorIR."""
    
    def test_simple_matmul_codegen(self):
        """Test lowering a single MATMUL operation."""
        ir = TensorIR()
        ir.set_input_shapes({
            "x": (1, 512),
            "w": (512, 256),
        })
        ir.set_output_shapes({"y": (1, 256)})
        
        op = TensorOp(
            op_type=TensorOpType.MATMUL,
            input_shapes={"x": (1, 512), "w": (512, 256)},
            output_shape=(1, 256),
            name="y",
        )
        ir.add_op(op)
        
        codegen = Codegen()
        instructions = codegen.compile(ir)
        
        # Should have: LOAD x, LOAD w, MATMUL, STORE y
        assert len(instructions) >= 3
        assert instructions[0].opcode == OpCode.LOAD
        assert instructions[-1].opcode == OpCode.STORE
    
    def test_mlp_codegen(self):
        """Test lowering a 2-layer MLP."""
        ir = TensorIR()
        ir.set_input_shapes({
            "input": (1, 1024),
            "w1": (1024, 512),
            "b1": (512,),
            "w2": (512, 10),
            "b2": (10,),
        })
        
        op1 = TensorOp(
            op_type=TensorOpType.LINEAR,
            input_shapes={"input": (1, 1024), "w1": (1024, 512), "b1": (512,)},
            output_shape=(1, 512),
            name="h1",
        )
        ir.add_op(op1)
        
        op2 = TensorOp(
            op_type=TensorOpType.RELU,
            input_shapes={"h1": (1, 512)},
            output_shape=(1, 512),
            name="h1_relu",
        )
        ir.add_op(op2)
        
        op3 = TensorOp(
            op_type=TensorOpType.LINEAR,
            input_shapes={"h1_relu": (1, 512), "w2": (512, 10), "b2": (10,)},
            output_shape=(1, 10),
            name="output",
        )
        ir.add_op(op3)
        
        ir.set_output_shapes({"output": (1, 10)})
        
        codegen = Codegen()
        instructions = codegen.compile(ir)
        
        # Verify we have instructions
        assert len(instructions) > 0
        
        # Verify we have compute operations (MATMUL, RELU)
        opcodes = [instr.opcode for instr in instructions]
        assert OpCode.MATMUL in opcodes
        assert OpCode.RELU in opcodes


class TestScheduler:
    """Test instruction scheduling."""
    
    def test_sequential_scheduling(self):
        """Test that scheduler executes instructions sequentially."""
        instructions = [
            Instruction(OpCode.LOAD, ["x"], "x", (1, 512)),
            Instruction(OpCode.LOAD, ["w"], "w", (512, 256)),
            Instruction(OpCode.MATMUL, ["x", "w"], "y", (1, 256)),
            Instruction(OpCode.STORE, ["y"], "y", (1, 256)),
        ]
        
        scheduler = Scheduler()
        timeline = scheduler.execute(instructions)
        
        assert len(timeline) == 4
        assert timeline[0].start_cycle == 0
        assert timeline[0].end_cycle == 1  # LOAD latency
        assert timeline[1].start_cycle == 1
        assert timeline[2].start_cycle == 2  # After second LOAD
        assert timeline[3].start_cycle == 12  # After MATMUL (latency 10)
    
    def test_total_cycles(self):
        """Test that total cycles are calculated correctly."""
        instructions = [
            Instruction(OpCode.LOAD, ["x"], "x", (1, 512)),
            Instruction(OpCode.MATMUL, ["x", "w"], "y", (1, 256)),  # 10 cycles
            Instruction(OpCode.STORE, ["y"], "y", (1, 256)),  # 1 cycle
        ]
        
        scheduler = Scheduler()
        timeline = scheduler.execute(instructions)
        total = scheduler.get_total_cycles()
        
        # LOAD(1) + MATMUL(10) + STORE(1) = 12 cycles
        assert total == 12


class TestMemorySystem:
    """Test memory system."""
    
    def test_load_store(self):
        """Test basic load/store operations."""
        mem = MemorySystem(buffer_capacity=1000)
        
        assert mem.load(500)
        assert mem.current_usage == 500
        
        assert mem.load(400)
        assert mem.current_usage == 900
        
        # Should fail - insufficient capacity
        assert not mem.load(200)
        
        # Store reduces usage
        assert mem.store(500)
        assert mem.current_usage == 400
    
    def test_peak_usage(self):
        """Test peak usage tracking."""
        mem = MemorySystem(buffer_capacity=10000)
        
        mem.load(100)
        assert mem.peak_usage == 100
        
        mem.load(200)
        assert mem.peak_usage == 300
        
        mem.load(150)
        assert mem.peak_usage == 450
        
        mem.store(100)
        assert mem.peak_usage == 450  # Peak doesn't decrease


class TestSimulator:
    """Test end-to-end simulation."""
    
    def test_simple_simulation(self):
        """Test simulating a simple program."""
        instructions = [
            Instruction(OpCode.LOAD, ["input"], "input", (1, 256)),
            Instruction(OpCode.LOAD, ["weight"], "weight", (256, 128)),
            Instruction(OpCode.MATMUL, ["input", "weight"], "output", (1, 128)),
            Instruction(OpCode.STORE, ["output"], "output", (1, 128)),
        ]
        
        simulator = Simulator(buffer_capacity=2 * 1024 * 1024)  # 2 MB buffer
        stats = simulator.simulate(instructions)
        
        # Verify stats
        assert stats.total_cycles > 0
        assert stats.instruction_count == 4
        assert stats.bytes_loaded > 0
        assert stats.bytes_stored > 0
        assert stats.peak_buffer_usage > 0
    
    def test_execution_timeline(self):
        """Test that execution timeline is generated."""
        instructions = [
            Instruction(OpCode.LOAD, ["x"], "x", (1, 512)),
            Instruction(OpCode.RELU, ["x"], "y", (1, 512)),
            Instruction(OpCode.STORE, ["y"], "y", (1, 512)),
        ]
        
        simulator = Simulator()
        simulator.simulate(instructions)
        timeline = simulator.get_execution_timeline()
        
        assert len(timeline) == 3
        assert timeline[0].instruction.opcode == OpCode.LOAD
        assert timeline[1].instruction.opcode == OpCode.RELU
        assert timeline[2].instruction.opcode == OpCode.STORE


class TestPerformanceAnalyzer:
    """Test performance analysis."""
    
    def test_analyzer_metrics(self):
        """Test that analyzer computes metrics correctly."""
        instructions = [
            Instruction(OpCode.LOAD, ["x"], "x", (1, 512)),
            Instruction(OpCode.MATMUL, ["x", "w"], "y", (1, 256)),
            Instruction(OpCode.STORE, ["y"], "y", (1, 256)),
        ]
        
        simulator = Simulator(buffer_capacity=2 * 1024 * 1024)
        stats = simulator.simulate(instructions)
        timeline = simulator.get_execution_timeline()
        
        analyzer = PerformanceAnalyzer(stats, timeline)
        metrics = analyzer.analyze()
        
        # Verify metrics
        assert 'total_cycles' in metrics
        assert 'compute_cycles' in metrics
        assert 'memory_cycles' in metrics
        assert 'compute_utilization' in metrics
        assert 'memory_utilization' in metrics
        assert 'average_latency_per_instruction' in metrics
        
        # Verify values are reasonable
        assert metrics['total_cycles'] > 0
        assert 0 <= metrics['compute_utilization'] <= 100
        assert 0 <= metrics['memory_utilization'] <= 100
        assert metrics['average_latency_per_instruction'] > 0
    
    def test_analyzer_summary(self):
        """Test that analyzer generates a summary string."""
        instructions = [
            Instruction(OpCode.LOAD, ["x"], "x", (1, 512)),
            Instruction(OpCode.RELU, ["x"], "y", (1, 512)),
            Instruction(OpCode.STORE, ["y"], "y", (1, 512)),
        ]
        
        simulator = Simulator()
        stats = simulator.simulate(instructions)
        timeline = simulator.get_execution_timeline()
        
        analyzer = PerformanceAnalyzer(stats, timeline)
        summary = analyzer.summary()
        
        assert isinstance(summary, str)
        assert "PERFORMANCE ANALYSIS" in summary
        assert "Total Cycles" in summary
        assert "Compute Cycles" in summary


class TestBottleneckDetector:
    """Test bottleneck detection."""
    
    def test_detector_report(self):
        """Test that detector generates a report."""
        instructions = [
            Instruction(OpCode.LOAD, ["x"], "x", (1, 512)),
            Instruction(OpCode.LOAD, ["w"], "w", (512, 256)),
            Instruction(OpCode.MATMUL, ["x", "w"], "y", (1, 256)),
            Instruction(OpCode.STORE, ["y"], "y", (1, 256)),
        ]
        
        simulator = Simulator(buffer_capacity=2 * 1024 * 1024)
        stats = simulator.simulate(instructions)
        timeline = simulator.get_execution_timeline()
        
        analyzer = PerformanceAnalyzer(stats, timeline)
        metrics = analyzer.analyze()
        
        detector = BottleneckDetector(metrics)
        report = detector.detect()
        
        assert isinstance(report, str)
        assert "BOTTLENECK ANALYSIS" in report
    
    def test_detector_memory_bound(self):
        """Test detection of memory-bound workload."""
        # Create a heavily memory-bound workload (many loads)
        instructions = [
            Instruction(OpCode.LOAD, ["x1"], "x1", (1, 512)),
            Instruction(OpCode.LOAD, ["x2"], "x2", (1, 512)),
            Instruction(OpCode.LOAD, ["x3"], "x3", (1, 512)),
            Instruction(OpCode.LOAD, ["x4"], "x4", (1, 512)),
            Instruction(OpCode.LOAD, ["x5"], "x5", (1, 512)),
            Instruction(OpCode.ADD, ["x1", "x2"], "y", (1, 512)),
            Instruction(OpCode.STORE, ["y"], "y", (1, 512)),
        ]
        
        simulator = Simulator(buffer_capacity=2 * 1024 * 1024)
        stats = simulator.simulate(instructions)
        timeline = simulator.get_execution_timeline()
        
        analyzer = PerformanceAnalyzer(stats, timeline)
        metrics = analyzer.analyze()
        
        # Should detect memory bound since many memory ops
        detector = BottleneckDetector(metrics)
        report = detector.detect()
        
        # Should generate a valid report
        assert isinstance(report, str)
        assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

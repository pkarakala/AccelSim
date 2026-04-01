"""
Performance Analysis Engine.

Computes detailed performance metrics from simulation data.

Metrics computed:
- Cycle breakdown: Compute, memory, idle cycles with percentages
- Instruction metrics: Latency, throughput
- Memory metrics: Bandwidth, traffic volume
- Buffer metrics: Utilization, capacity pressure

Analysis Framework:
1. Categorize cycles by operation type (compute vs. memory)
2. Compute utilization percentages
3. Derive bandwidth from traffic and cycle time
4. Calculate per-instruction statistics

Used for:
- Identifying bottlenecks (compute vs. memory bound)
- Predicting performance scaling
- Guiding optimization efforts
- Comparing different kernel implementations
"""

from typing import List, Dict, Any
from ir.instructions import OpCode
from runtime.scheduler import ExecutionEntry
from runtime.stats import SimulationStats


class PerformanceAnalyzer:
    """
    Computes performance metrics from simulation results.
    
    Takes simulator output (execution timeline + aggregated statistics)
    and derives detailed performance metrics useful for:
    - Bottleneck identification
    - Performance prediction
    - Hardware-algorithm co-design decisions
    
    Key insights provided:
    - Utilization breakdown (compute vs. memory vs. idle)
    - Critical resources (where time is spent)
    - Efficiency metrics (throughput, bandwidth, latency)
    - Buffer analysis (utilization, pressure)
    """
    
    def __init__(self, stats: SimulationStats, timeline: List[ExecutionEntry]):
        """
        Initialize analyzer with simulation data.
        
        Args:
            stats: SimulationStats from simulator containing:
                   - total_cycles, instruction_count
                   - bytes_loaded, bytes_stored
                   - peak_buffer_usage, buffer_capacity
            timeline: Execution timeline from simulator containing:
                      - ExecutionEntry records for each instruction
                      - start/end cycles and operation type
        
        Stores both for independent analysis of:
        - Instruction-level behavior (timeline)
        - Aggregate statistics (stats)
        """
        self.stats = stats
        self.timeline = timeline
    
    def analyze(self) -> Dict[str, Any]:
        """
        Compute comprehensive performance metrics.
        
        Analysis Pipeline:
        1. Cycle Categorization:
           - Count compute cycles (MATMUL, ADD, RELU operations)
           - Count memory cycles (LOAD, STORE operations)
           - Calculate idle cycles (total - compute - memory)
        
        2. Utilization Analysis:
           - Compute utilization (compute_cycles / total_cycles)
           - Memory utilization (memory_cycles / total_cycles)
           - Idle utilization (idle_cycles / total_cycles)
        
        3. Efficiency Metrics:
           - Average latency per instruction (total_cycles / instr_count)
           - Throughput (instr_count / total_cycles)
           - Memory bandwidth (bytes / memory_cycles)
        
        4. Buffer Analysis:
           - Peak usage vs. capacity
           - Utilization percentage
        
        Returns:
            Dictionary mapping metric names to computed values:
            - total_cycles, compute_cycles, memory_cycles, idle_cycles
            - compute_utilization, memory_utilization, idle_utilization (%)
            - instruction_count, average_latency_per_instruction (cycles)
            - throughput_instructions_per_cycle (instr/cycle)
            - memory_traffic, memory_bandwidth (bytes/cycle)
            - peak_buffer_usage, buffer_utilization (%)
        
        Example usage:
            metrics = analyzer.analyze()
            print(f"Compute utilization: {metrics['compute_utilization']:.1f}%")
            print(f"Memory bandwidth: {metrics['memory_bandwidth']:.0f} B/cycle")
        """
        metrics = {}
        
        # Categorize cycles by operation type
        compute_cycles = self._count_compute_cycles()
        memory_cycles = self._count_memory_cycles()
        total_cycles = self.stats.total_cycles
        idle_cycles = total_cycles - compute_cycles - memory_cycles
        
        metrics['total_cycles'] = total_cycles
        metrics['compute_cycles'] = compute_cycles
        metrics['memory_cycles'] = memory_cycles
        metrics['idle_cycles'] = max(0, idle_cycles)
        
        # Utilization metrics
        metrics['compute_utilization'] = (
            100.0 * compute_cycles / max(1, total_cycles)
        )
        metrics['memory_utilization'] = (
            100.0 * memory_cycles / max(1, total_cycles)
        )
        metrics['idle_utilization'] = (
            100.0 * max(0, idle_cycles) / max(1, total_cycles)
        )
        
        # Memory bandwidth
        total_traffic = self.stats.total_memory_traffic
        metrics['memory_traffic'] = total_traffic
        if memory_cycles > 0:
            metrics['memory_bandwidth'] = total_traffic / memory_cycles
        else:
            metrics['memory_bandwidth'] = 0.0
        
        # Instruction metrics
        instr_count = self.stats.instruction_count
        metrics['instruction_count'] = instr_count
        metrics['average_latency_per_instruction'] = (
            total_cycles / max(1, instr_count)
        )
        metrics['throughput_instructions_per_cycle'] = (
            instr_count / max(1, total_cycles)
        )
        
        # Buffer metrics
        metrics['peak_buffer_usage'] = self.stats.peak_buffer_usage
        metrics['buffer_capacity'] = self.stats.buffer_capacity
        metrics['buffer_utilization'] = (
            100.0 * self.stats.peak_buffer_usage / max(1, self.stats.buffer_capacity)
        )
        
        return metrics
    
    def _count_compute_cycles(self) -> int:
        """
        Count cycles spent on compute operations.
        
        Inspects execution timeline and sums duration of compute-intensive
        instructions: MATMUL (10 cycles), ADD (2 cycles), RELU (1 cycle).
        
        Returns:
            Total cycles spent in compute operations.
        
        Example:
            - MATMUL (10c) + ADD (2c) + RELU (1c) = 13 compute cycles
        """
        count = 0
        for entry in self.timeline:
            if entry.instruction.opcode in [OpCode.MATMUL, OpCode.ADD, OpCode.RELU]:
                count += entry.duration
        return count
    
    def _count_memory_cycles(self) -> int:
        """
        Count cycles spent on memory operations.
        
        Inspects execution timeline and sums duration of memory-related
        instructions: LOAD (1 cycle), STORE (1 cycle).
        
        Returns:
            Total cycles spent moving data to/from memory.
        
        Note:
            Memory cycles only count time spent in LOAD/STORE operations.
            Compute operations that read/write from buffer are not counted.
        
        Example:
            - LOAD (1c) + LOAD (1c) + STORE (1c) = 3 memory cycles
        """
        count = 0
        for entry in self.timeline:
            if entry.instruction.opcode in [OpCode.LOAD, OpCode.STORE]:
                count += entry.duration
        return count
    
    def summary(self) -> str:
        """
        Generate human-readable performance summary.
        
        Calls analyze() to compute all metrics, then formats them
        into a nicely organized text report with sections:
        - Cycle breakdown (with percentages)
        - Instruction metrics
        - Memory metrics
        - Buffer metrics
        
        Returns:
            Formatted string suitable for printing or logging.
            Includes headers, formatting, and units.
        
        Example output:
            ======= PERFORMANCE ANALYSIS =======
            CYCLE BREAKDOWN
            Total Cycles:        27
              Compute Cycles:     21 (77.8%)
              Memory Cycles:       6 (22.2%)
              Idle Cycles:         0 (0.0%)
            ...
        """
        metrics = self.analyze()
        
        lines = [
            "\n" + "=" * 70,
            "PERFORMANCE ANALYSIS",
            "=" * 70,
            "",
            "CYCLE BREAKDOWN",
            "-" * 70,
            f"Total Cycles:                {metrics['total_cycles']:>8,}",
            f"  Compute Cycles:            {metrics['compute_cycles']:>8,}  "
            f"({metrics['compute_utilization']:>5.1f}%)",
            f"  Memory Cycles:             {metrics['memory_cycles']:>8,}  "
            f"({metrics['memory_utilization']:>5.1f}%)",
            f"  Idle Cycles:               {metrics['idle_cycles']:>8,}  "
            f"({metrics['idle_utilization']:>5.1f}%)",
            "",
            "INSTRUCTION METRICS",
            "-" * 70,
            f"Instructions Executed:       {metrics['instruction_count']:>8,}",
            f"Avg Latency per Instr:       {metrics['average_latency_per_instruction']:>8.2f} cycles",
            f"Throughput:                  {metrics['throughput_instructions_per_cycle']:>8.4f} instr/cycle",
            "",
            "MEMORY METRICS",
            "-" * 70,
            f"Memory Traffic:              {metrics['memory_traffic']:>8,} bytes",
            f"Memory Bandwidth:            {metrics['memory_bandwidth']:>8.2f} bytes/cycle",
            f"Bytes Loaded:                {self.stats.bytes_loaded:>8,}",
            f"Bytes Stored:                {self.stats.bytes_stored:>8,}",
            "",
            "BUFFER METRICS",
            "-" * 70,
            f"Peak Buffer Usage:           {metrics['peak_buffer_usage']:>8,} / {metrics['buffer_capacity']:>8,} bytes",
            f"Buffer Utilization:          {metrics['buffer_utilization']:>8.1f}%",
            "=" * 70,
        ]
        
        return "\n".join(lines)

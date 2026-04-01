"""
Simulator: Main execution engine combining scheduler and memory system.

Executes instruction streams and collects performance metrics.
"""

from typing import List
from ir.instructions import Instruction, OpCode
from runtime.scheduler import Scheduler, ExecutionEntry
from runtime.memory_system import MemorySystem
from runtime.stats import StatsCollector, SimulationStats


class Simulator:
    """
    Accelerator Runtime Simulator.
    
    Orchestrates the complete execution of an instruction stream by combining:
    - Scheduler: Determines when each instruction executes
    - Memory System: Tracks buffer usage and memory traffic
    - Statistics Collector: Aggregates performance metrics
    
    Execution Flow:
    1. Schedule all instructions with latency model
    2. Process memory operations (LOAD/STORE) to track buffer state
    3. Record performance statistics (cycles, memory traffic, buffer usage)
    4. Return aggregated stats and execution timeline
    """
    
    def __init__(self, buffer_capacity: int = 1024 * 1024):
        """
        Initialize simulator with memory and scheduler.
        
        Args:
            buffer_capacity: On-chip buffer capacity in bytes (default: 1 MB)
        
        Creates:
            - Memory system with specified capacity
            - Scheduler for instruction timing
            - Stats collector for metrics aggregation
        """
        self.memory = MemorySystem(buffer_capacity=buffer_capacity)
        self.scheduler = Scheduler()
        self.stats_collector = StatsCollector()
        self.execution_log: List[ExecutionEntry] = []
    
    def simulate(self, instructions: List[Instruction]) -> SimulationStats:
        """
        Execute instruction stream and return statistics.
        
        Performs a complete simulation cycle:
        1. Resets memory and scheduler state
        2. Schedules all instructions with cycle timing
        3. Simulates LOAD/STORE memory operations
        4. Tracks buffer usage (peak and total)
        5. Aggregates performance statistics
        
        Args:
            instructions: List of accelerator instructions to execute
        
        Returns:
            SimulationStats containing:
                - total_cycles: Total execution time
                - instruction_count: Number of instructions
                - bytes_loaded/stored: Memory traffic
                - peak_buffer_usage: Maximum buffer occupancy
        
        Raises:
            RuntimeError: If buffer capacity exceeded during LOAD or underflow during STORE
        """
        # Reset state
        self.memory.reset()
        self.scheduler.reset()
        
        # Execute instructions with scheduler
        self.execution_log = self.scheduler.execute(instructions)
        
        # Process memory operations to track buffer usage
        for entry in self.execution_log:
            instr = entry.instruction
            
            if instr.opcode == OpCode.LOAD:
                traffic = instr.memory_traffic
                success = self.memory.load(traffic)
                if not success:
                    raise RuntimeError(
                        f"Memory overflow during LOAD: "
                        f"needed {traffic} bytes, available "
                        f"{self.memory.buffer_capacity - self.memory.current_usage}"
                    )
            
            elif instr.opcode == OpCode.STORE:
                traffic = instr.memory_traffic
                success = self.memory.store(traffic)
                if not success:
                    raise RuntimeError(
                        f"Memory underflow during STORE: "
                        f"tried to store {traffic} bytes, "
                        f"current usage {self.memory.current_usage}"
                    )
        
        # Collect statistics
        mem_stats = self.memory.get_stats()
        
        self.stats_collector.record_cycles(self.scheduler.get_total_cycles())
        self.stats_collector.record_instruction_count(len(instructions))
        self.stats_collector.record_memory_load(mem_stats.bytes_loaded)
        self.stats_collector.record_memory_store(mem_stats.bytes_stored)
        self.stats_collector.record_peak_buffer_usage(
            mem_stats.peak_usage,
            mem_stats.buffer_capacity
        )
        
        return self.stats_collector.get_stats()
    
    def get_execution_timeline(self) -> List[ExecutionEntry]:
        """
        Get execution timeline with cycle-accurate timing.
        
        Returns:
            List of ExecutionEntry records, each containing:
                - instruction: The scheduled instruction
                - start_cycle: When instruction starts
                - end_cycle: When instruction completes
                - duration: Latency in cycles
        """
        return self.execution_log

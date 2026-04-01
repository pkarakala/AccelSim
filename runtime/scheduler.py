"""
Instruction Scheduler: Implements sequential instruction execution.

Models a simple sequential (non-pipelined) execution model where:
- Instructions execute one-at-a-time in program order
- Each instruction has a fixed latency (in cycles)
- Latency varies by operation type:
  LOAD: 1 cycle (memory access)
  MATMUL: 10 cycles (compute intensive)
  ADD: 2 cycles (simple compute)
  RELU: 1 cycle (element-wise)
  STORE: 1 cycle (memory write)

Future extensions could add:
- Pipelining (overlap instruction execution)
- Parallel execution (multiple functional units)
- Resource conflicts (contention for shared resources)
"""

from typing import List, Dict
from ir.instructions import Instruction


class ExecutionEntry:
    """
    Execution record for a single instruction.
    
    Tracks timing information for one instruction execution:
    - instruction: The instruction that was executed
    - start_cycle: When instruction starts execution
    - end_cycle: When instruction completes
    - duration: end_cycle - start_cycle
    
    Used to generate execution timelines and identify critical paths.
    """
    
    def __init__(self, instruction: Instruction, start_cycle: int, end_cycle: int):
        self.instruction = instruction
        self.start_cycle = start_cycle
        self.end_cycle = end_cycle
        self.duration = end_cycle - start_cycle
    
    def __repr__(self) -> str:
        return (f"[{self.start_cycle:4d}-{self.end_cycle:4d}] "
                f"{self.instruction.opcode.value:8s} "
                f"{str(self.instruction.inputs):30s} "
                f"-> {self.instruction.output}")


class Scheduler:
    """
    Sequential Instruction Scheduler.
    
    Schedules instructions for execution using a simple model:
    1. Instructions execute in program order (no reordering)
    2. One instruction executes at a time (no pipelining)
    3. Each instruction contributes its latency to total time
    4. Total cycles = sum of all instruction latencies
    
    Execution Time Calculation:
        Instruction 0: [cycle 0 - cycle latency0)
        Instruction 1: [cycle latency0 - cycle latency0 + latency1)
        ...
        Total: sum(latencies)
    
    This model is realistic for simple accelerators and provides
    a baseline for understanding instruction-level bottlenecks.
    """
    
    def __init__(self):
        self.current_cycle = 0
        self.execution_log: List[ExecutionEntry] = []
    
    def execute(self, instructions: List[Instruction]) -> List[ExecutionEntry]:
        """
        Execute a sequence of instructions sequentially.
        
        Algorithm:
        1. Initialize current_cycle to 0
        2. For each instruction in order:
           a. Get instruction latency
           b. Record: start = current_cycle
           c. Record: end = current_cycle + latency
           d. Create ExecutionEntry with timing info
           e. Update current_cycle = end
        3. Return execution log with timing for all instructions
        
        Args:
            instructions: List of instructions to execute
        
        Returns:
            List of ExecutionEntry records with cycle-accurate timing
        """
        self.current_cycle = 0
        self.execution_log = []
        
        for instr in instructions:
            latency = instr.latency
            start = self.current_cycle
            end = start + latency
            
            entry = ExecutionEntry(instr, start, end)
            self.execution_log.append(entry)
            
            self.current_cycle = end
        
        return self.execution_log
    
    def get_total_cycles(self) -> int:
        """
        Get total cycles for complete execution.
        
        Returns:
            Total number of cycles needed to execute all instructions.
            This is the end_cycle of the last instruction.
        """
        return self.current_cycle
    
    def reset(self) -> None:
        """
        Reset scheduler state for a new execution run.
        
        Clears:
            - current_cycle counter
            - execution_log history
        
        Call before running a new simulation.
        """
        self.current_cycle = 0
        self.execution_log = []

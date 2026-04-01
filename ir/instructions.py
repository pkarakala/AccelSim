"""
Instruction definitions for the neural accelerator.

Supported opcodes:
- LOAD: Load data into buffers
- MATMUL: Matrix multiplication
- ADD: Element-wise addition
- RELU: Rectified linear unit
- STORE: Store results to memory
"""

from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


class OpCode(Enum):
    """Instruction opcodes."""
    LOAD = "load"
    MATMUL = "matmul"
    ADD = "add"
    RELU = "relu"
    STORE = "store"


@dataclass
class Instruction:
    """
    A single accelerator instruction.
    
    Attributes:
        opcode: Operation type (LOAD, MATMUL, ADD, RELU, STORE)
        inputs: List of input tensor names
        output: Output tensor name
        shape: Shape of the tensor (batch, height, width, channels)
        attributes: Dictionary of operation-specific attributes
                   (e.g., bias for MATMUL, alpha for RELU)
    """
    opcode: OpCode
    inputs: List[str]
    output: str
    shape: Tuple[int, ...]
    attributes: dict = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def __repr__(self) -> str:
        attrs_str = f", {self.attributes}" if self.attributes else ""
        return (f"Instr({self.opcode.value} {self.inputs} -> {self.output} "
                f"shape={self.shape}{attrs_str})")
    
    @property
    def latency(self) -> int:
        """Estimated latency in cycles for this instruction."""
        # Base latencies (simplified)
        latencies = {
            OpCode.LOAD: 1,
            OpCode.STORE: 1,
            OpCode.MATMUL: 10,  # Most expensive
            OpCode.ADD: 2,
            OpCode.RELU: 1,
        }
        return latencies[self.opcode]
    
    @property
    def memory_traffic(self) -> int:
        """Estimated bytes of memory traffic (simplified)."""
        # Calculate bytes based on shape (assume float32 = 4 bytes)
        num_elements = 1
        for dim in self.shape:
            num_elements *= dim
        bytes_per_elem = 4
        
        if self.opcode == OpCode.LOAD:
            return num_elements * bytes_per_elem  # Read
        elif self.opcode == OpCode.STORE:
            return num_elements * bytes_per_elem  # Write
        else:
            # Assume 2 inputs + 1 output for compute ops
            return num_elements * bytes_per_elem * 3

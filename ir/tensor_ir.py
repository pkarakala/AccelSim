"""
Minimal Tensor IR definition.

Provides a simple abstraction for neural network operations
that can be compiled down to accelerator instructions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum


class TensorOpType(Enum):
    """Types of tensor operations."""
    LINEAR = "linear"
    RELU = "relu"
    ADD = "add"
    MATMUL = "matmul"


@dataclass
class TensorOp:
    """A single tensor operation in the IR."""
    op_type: TensorOpType
    input_shapes: Dict[str, Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    name: str
    attributes: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"TensorOp({self.op_type.value} {self.name} -> {self.output_shape})"


@dataclass
class TensorIR:
    """
    Container for a tensor IR program.
    Represents a computation graph as a sequence of operations.
    """
    ops: List[TensorOp] = field(default_factory=list)
    input_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    output_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    
    def add_op(self, op: TensorOp) -> None:
        """Add an operation to the IR."""
        self.ops.append(op)
    
    def set_input_shapes(self, shapes: Dict[str, Tuple[int, ...]]) -> None:
        """Set input tensor shapes."""
        self.input_shapes.update(shapes)
    
    def set_output_shapes(self, shapes: Dict[str, Tuple[int, ...]]) -> None:
        """Set output tensor shapes."""
        self.output_shapes.update(shapes)
    
    def __repr__(self) -> str:
        return f"TensorIR(ops={len(self.ops)})"
    
    def summary(self) -> str:
        """Return a string summary of the IR."""
        lines = [f"TensorIR Program ({len(self.ops)} ops)"]
        lines.append(f"Inputs: {self.input_shapes}")
        lines.append(f"Outputs: {self.output_shapes}")
        lines.append("Operations:")
        for i, op in enumerate(self.ops):
            lines.append(f"  {i}: {op}")
        return "\n".join(lines)

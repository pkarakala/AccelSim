"""
Codegen: Convert TensorIR operations into accelerator instructions.

This module lowers high-level tensor operations into low-level
accelerator instructions (LOAD, MATMUL, ADD, RELU, STORE).
"""

from typing import List, Dict
from ir.instructions import Instruction, OpCode
from ir.tensor_ir import TensorIR, TensorOpType


class Codegen:
    """Compiles TensorIR to accelerator instructions."""
    
    def __init__(self):
        self.instructions: List[Instruction] = []
        self.tensor_counter = 0
    
    def _get_tensor_name(self, prefix: str) -> str:
        """Generate a unique tensor name."""
        name = f"{prefix}_{self.tensor_counter}"
        self.tensor_counter += 1
        return name
    
    def compile(self, ir: TensorIR) -> List[Instruction]:
        """
        Compile TensorIR to instructions.
        
        Returns a list of accelerator instructions.
        """
        self.instructions = []
        self.tensor_counter = 0
        
        # Create LOAD instructions for inputs
        tensor_map = {}
        for input_name, shape in ir.input_shapes.items():
            load_instr = Instruction(
                opcode=OpCode.LOAD,
                inputs=[input_name],
                output=input_name,
                shape=shape,
            )
            self.instructions.append(load_instr)
            tensor_map[input_name] = input_name
        
        # Process each operation
        for op in ir.ops:
            if op.op_type == TensorOpType.MATMUL:
                # MATMUL: y = wx + b (linear layer)
                # Expects inputs: [weight, input, bias?]
                instr = Instruction(
                    opcode=OpCode.MATMUL,
                    inputs=list(op.input_shapes.keys()),
                    output=op.name,
                    shape=op.output_shape,
                    attributes=op.attributes,
                )
                self.instructions.append(instr)
                tensor_map[op.name] = op.name
            
            elif op.op_type == TensorOpType.RELU:
                # RELU: y = max(0, x)
                input_tensors = list(op.input_shapes.keys())
                instr = Instruction(
                    opcode=OpCode.RELU,
                    inputs=input_tensors,
                    output=op.name,
                    shape=op.output_shape,
                )
                self.instructions.append(instr)
                tensor_map[op.name] = op.name
            
            elif op.op_type == TensorOpType.ADD:
                # ADD: y = x1 + x2
                instr = Instruction(
                    opcode=OpCode.ADD,
                    inputs=list(op.input_shapes.keys()),
                    output=op.name,
                    shape=op.output_shape,
                )
                self.instructions.append(instr)
                tensor_map[op.name] = op.name
            
            elif op.op_type == TensorOpType.LINEAR:
                # LINEAR is a composite: MATMUL + ADD (if bias) or just MATMUL
                # For simplicity, emit as MATMUL with attributes
                instr = Instruction(
                    opcode=OpCode.MATMUL,
                    inputs=list(op.input_shapes.keys()),
                    output=op.name,
                    shape=op.output_shape,
                    attributes=op.attributes,
                )
                self.instructions.append(instr)
                tensor_map[op.name] = op.name
        
        # Create STORE instructions for outputs
        for output_name, shape in ir.output_shapes.items():
            if output_name in tensor_map:
                store_instr = Instruction(
                    opcode=OpCode.STORE,
                    inputs=[output_name],
                    output=output_name,
                    shape=shape,
                )
                self.instructions.append(store_instr)
        
        return self.instructions

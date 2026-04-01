"""
Pretty printing utilities for displaying simulation results.
"""

from typing import List
from ir.instructions import Instruction
from runtime.scheduler import ExecutionEntry


def print_instruction_stream(instructions: List[Instruction]) -> None:
    """Pretty print instruction stream."""
    print("\n" + "=" * 80)
    print("INSTRUCTION STREAM")
    print("=" * 80)
    
    for i, instr in enumerate(instructions):
        print(f"{i:3d}: {instr}")
    
    print(f"\nTotal instructions: {len(instructions)}")


def print_execution_timeline(timeline: List[ExecutionEntry]) -> None:
    """Pretty print execution timeline with cycle information."""
    print("\n" + "=" * 80)
    print("EXECUTION TIMELINE")
    print("=" * 80)
    print()
    
    for entry in timeline:
        print(entry)
    
    if timeline:
        total_cycles = timeline[-1].end_cycle
        print(f"\nTotal cycles: {total_cycles}")


def print_tensor_ir_summary(ir_summary: str) -> None:
    """Pretty print tensor IR summary."""
    print("\n" + "=" * 80)
    print("TENSOR IR")
    print("=" * 80)
    print(ir_summary)

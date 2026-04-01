"""
Memory Usage Visualization: Plot buffer usage over time.

Tracks:
- Buffer usage at each cycle
- Capacity limits
- Load/store operations
"""

from typing import List
import matplotlib.pyplot as plt
from ir.instructions import OpCode
from runtime.scheduler import ExecutionEntry


def plot_memory_usage(timeline: List[ExecutionEntry], 
                      buffer_capacity: int,
                      output_path: str = None) -> None:
    """
    Plot memory (buffer) usage over time.
    
    Args:
        timeline: Execution timeline from simulator
        buffer_capacity: Maximum buffer capacity in bytes
        output_path: Path to save figure (if None, displays plot)
    """
    if not timeline:
        print("Timeline is empty, skipping plot")
        return
    
    # Simulate buffer usage over time
    max_cycle = max(e.end_cycle for e in timeline)
    cycles = []
    buffer_usage = []
    
    current_usage = 0
    instr_idx = 0
    
    for cycle in range(max_cycle + 1):
        cycles.append(cycle)
        
        # Find which instructions are executing at this cycle
        while instr_idx < len(timeline) and timeline[instr_idx].end_cycle <= cycle:
            instr = timeline[instr_idx].instruction
            
            # Update buffer based on operation type
            if instr.opcode == OpCode.LOAD:
                current_usage += instr.memory_traffic
            elif instr.opcode == OpCode.STORE:
                current_usage = max(0, current_usage - instr.memory_traffic)
            
            instr_idx += 1
        
        buffer_usage.append(current_usage)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Buffer usage line plot
    ax1.plot(cycles, buffer_usage, linewidth=2, color='#4ECDC4', label='Buffer Usage')
    ax1.axhline(buffer_capacity, color='red', linestyle='--', linewidth=2, label='Capacity')
    ax1.fill_between(cycles, 0, buffer_usage, alpha=0.3, color='#4ECDC4')
    
    ax1.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Buffer Usage (bytes)', fontsize=11, fontweight='bold')
    ax1.set_title('Buffer Usage Over Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, buffer_capacity * 1.1)
    
    # Plot 2: Memory operations stacked area
    load_traffic = []
    store_traffic = []
    
    for entry in timeline:
        if entry.instruction.opcode == OpCode.LOAD:
            load_traffic.append(entry.instruction.memory_traffic)
            store_traffic.append(0)
        elif entry.instruction.opcode == OpCode.STORE:
            load_traffic.append(0)
            store_traffic.append(entry.instruction.memory_traffic)
        else:
            load_traffic.append(0)
            store_traffic.append(0)
    
    x_pos = range(len(timeline))
    ax2.bar(x_pos, load_traffic, label='Load Bytes', color='#FF6B6B', alpha=0.7)
    ax2.bar(x_pos, store_traffic, bottom=load_traffic, label='Store Bytes', 
           color='#98D8C8', alpha=0.7)
    
    ax2.set_xlabel('Instruction', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Memory Traffic (bytes)', fontsize=11, fontweight='bold')
    ax2.set_title('Memory Traffic per Instruction', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"I{i}" for i in x_pos], fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Memory plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

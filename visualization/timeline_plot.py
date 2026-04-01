"""
Timeline Visualization: Plot instruction execution as a Gantt chart.

Uses matplotlib to visualize:
- Instruction execution timeline
- Operation types (color coded)
- Start and end cycles
"""

from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ir.instructions import OpCode
from runtime.scheduler import ExecutionEntry


def plot_timeline(timeline: List[ExecutionEntry], output_path: str = None) -> None:
    """
    Plot instruction execution timeline as a Gantt chart.
    
    Args:
        timeline: Execution timeline from simulator
        output_path: Path to save figure (if None, displays plot)
    """
    if not timeline:
        print("Timeline is empty, skipping plot")
        return
    
    # Color map for operation types
    color_map = {
        OpCode.LOAD: '#FF6B6B',    # Red
        OpCode.MATMUL: '#4ECDC4',  # Teal
        OpCode.ADD: '#45B7D1',     # Blue
        OpCode.RELU: '#FFA07A',    # Light salmon
        OpCode.STORE: '#98D8C8',   # Mint
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each instruction as a horizontal bar
    for idx, entry in enumerate(timeline):
        opcode = entry.instruction.opcode
        color = color_map.get(opcode, '#CCCCCC')
        
        # Bar from start to end cycle
        ax.barh(idx, entry.duration, left=entry.start_cycle, height=0.8,
                color=color, edgecolor='black', linewidth=0.5)
        
        # Add label in the middle of the bar
        mid_cycle = entry.start_cycle + entry.duration / 2
        ax.text(mid_cycle, idx, f"{opcode.value}", 
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax.set_ylabel('Instruction', fontsize=12, fontweight='bold')
    ax.set_title('Instruction Execution Timeline', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(timeline)))
    ax.set_yticklabels([f"I{i}" for i in range(len(timeline))])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Create legend
    legend_patches = [
        mpatches.Patch(color=color, label=opcode.value.upper())
        for opcode, color in color_map.items()
        if any(e.instruction.opcode == opcode for e in timeline)
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
    
    # Set x-axis limits
    max_cycle = max(e.end_cycle for e in timeline)
    ax.set_xlim(0, max_cycle * 1.05)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Timeline plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

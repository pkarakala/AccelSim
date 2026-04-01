"""
Stats: Aggregates simulation statistics.

Tracks performance metrics:
- Total cycles
- Instruction count
- Memory traffic (load + store)
- Peak buffer usage
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SimulationStats:
    """Aggregated simulation statistics."""
    total_cycles: int = 0
    instruction_count: int = 0
    bytes_loaded: int = 0
    bytes_stored: int = 0
    total_memory_traffic: int = 0
    peak_buffer_usage: int = 0
    buffer_capacity: int = 0
    
    def __str__(self) -> str:
        """Human-readable statistics summary."""
        lines = [
            "=" * 60,
            "SIMULATION STATISTICS",
            "=" * 60,
            f"Total Cycles:           {self.total_cycles:,}",
            f"Instruction Count:      {self.instruction_count:,}",
            f"Cycles per Instruction: {self.total_cycles / max(1, self.instruction_count):.2f}",
            "",
            "MEMORY STATISTICS",
            "-" * 60,
            f"Bytes Loaded:           {self.bytes_loaded:,}",
            f"Bytes Stored:           {self.bytes_stored:,}",
            f"Total Memory Traffic:   {self.total_memory_traffic:,}",
            f"Peak Buffer Usage:      {self.peak_buffer_usage:,} / {self.buffer_capacity:,}",
            f"Buffer Utilization:     {100.0 * self.peak_buffer_usage / max(1, self.buffer_capacity):.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


class StatsCollector:
    """Collects and aggregates simulation statistics."""
    
    def __init__(self):
        self.stats: Dict[str, int] = {
            'total_cycles': 0,
            'instruction_count': 0,
            'bytes_loaded': 0,
            'bytes_stored': 0,
            'peak_buffer_usage': 0,
            'buffer_capacity': 0,
        }
    
    def record_cycles(self, cycles: int) -> None:
        """Record total cycles."""
        self.stats['total_cycles'] = cycles
    
    def record_instruction_count(self, count: int) -> None:
        """Record number of instructions."""
        self.stats['instruction_count'] = count
    
    def record_memory_load(self, bytes_loaded: int) -> None:
        """Record bytes loaded."""
        self.stats['bytes_loaded'] = bytes_loaded
    
    def record_memory_store(self, bytes_stored: int) -> None:
        """Record bytes stored."""
        self.stats['bytes_stored'] = bytes_stored
    
    def record_peak_buffer_usage(self, usage: int, capacity: int) -> None:
        """Record peak buffer usage."""
        self.stats['peak_buffer_usage'] = usage
        self.stats['buffer_capacity'] = capacity
    
    def get_stats(self) -> SimulationStats:
        """Get aggregated statistics."""
        return SimulationStats(
            total_cycles=self.stats['total_cycles'],
            instruction_count=self.stats['instruction_count'],
            bytes_loaded=self.stats['bytes_loaded'],
            bytes_stored=self.stats['bytes_stored'],
            total_memory_traffic=self.stats['bytes_loaded'] + self.stats['bytes_stored'],
            peak_buffer_usage=self.stats['peak_buffer_usage'],
            buffer_capacity=self.stats['buffer_capacity'],
        )

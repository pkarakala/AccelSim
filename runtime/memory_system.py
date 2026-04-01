"""
On-Chip Buffer Memory Model.

Models a single on-chip buffer (scratch pad) with fixed capacity.
Simulates the behavior of on-chip SRAM on accelerators like:
- Google TPU (Matrix Unit buffer)
- NVIDIA GPU (Shared memory)
- Apple Neural Engine (SRAM)

Tracking:
- current_usage: Bytes currently in buffer
- peak_usage: Maximum bytes used (for sizing analysis)
- bytes_loaded: Total bytes read from main memory
- bytes_stored: Total bytes written to main memory

The model is conservative:
- LOAD adds bytes to buffer (must fit or fails)
- STORE removes bytes from buffer
- Tracks peak to identify buffer pressure

Future extensions:
- Multiple banks/ports (parallel access)
- Cache hierarchies (L1, L2, L3)
- Prefetching and eviction policies
"""

from dataclasses import dataclass


@dataclass
class MemoryStats:
    """
    Memory operation statistics.
    
    Attributes:
        bytes_loaded: Total bytes loaded from main memory
        bytes_stored: Total bytes stored to main memory
        buffer_capacity: Maximum on-chip buffer capacity
        peak_usage: Highest buffer occupancy during execution
        current_usage: Current buffer occupancy (at end of execution)
    
    Used to analyze:
    - Memory bandwidth requirements
    - Buffer pressure and utilization
    - Opportunity for data reuse
    """
    bytes_loaded: int = 0
    bytes_stored: int = 0
    buffer_capacity: int = 0
    peak_usage: int = 0
    current_usage: int = 0


class MemorySystem:
    """
    Simulated on-chip buffer memory.
    
    Models a single flat buffer with:
    - Fixed capacity (default 1 MB on accelerators)
    - Load/store semantics for data movement
    - Peak usage tracking for bottleneck analysis
    - Byte-granular accounting
    
    Usage Pattern:
    1. LOAD: Read data from main memory → buffer
    2. COMPUTE: Read from buffer as needed
    3. STORE: Write results from buffer → main memory
    
    Constraints:
    - Buffer has fixed capacity (fails if exceeded)
    - STORE must not exceed current buffer usage
    - Peak tracking for identifying buffer pressure
    """
    
    def __init__(self, buffer_capacity: int = 1024 * 1024):
        """
        Initialize on-chip buffer memory.
        
        Args:
            buffer_capacity: Maximum buffer size in bytes (default: 1 MB)
                             Typical values:
                             - 1-4 MB for edge accelerators
                             - 4-8 MB for mobile (e.g., Apple Neural Engine)
                             - 16-32 MB for data center (e.g., TPU)
        
        Initializes:
            - Empty buffer (current_usage = 0)
            - Statistics tracking (bytes_loaded, bytes_stored)
            - Peak usage monitoring (peak_usage = 0)
        """
        self.buffer_capacity = buffer_capacity
        self.current_usage = 0
        self.peak_usage = 0
        self.bytes_loaded = 0
        self.bytes_stored = 0
    
    def load(self, num_bytes: int) -> bool:
        """
        Load data from main memory into buffer.
        
        Simulates LOAD instruction: reads data from off-chip (DRAM)
        and brings it into on-chip buffer (SRAM).
        
        Args:
            num_bytes: Amount of data to load (in bytes)
        
        Returns:
            True if load succeeded
            False if insufficient buffer space (buffer overflow)
        
        Effects on success:
            - Increases current_usage by num_bytes
            - Increments bytes_loaded counter
            - Updates peak_usage if current exceeds previous peak
        
        Note:
            Returns False on failure; caller should check and handle.
            In this simulator, failure raises an exception.
        """
        if self.current_usage + num_bytes > self.buffer_capacity:
            return False  # Insufficient space
        
        self.current_usage += num_bytes
        self.bytes_loaded += num_bytes
        self.peak_usage = max(self.peak_usage, self.current_usage)
        return True
    
    def store(self, num_bytes: int) -> bool:
        """
        Store data from buffer back to main memory.
        
        Simulates STORE instruction: writes results from on-chip buffer
        (SRAM) back to off-chip memory (DRAM).
        
        Args:
            num_bytes: Amount of data to store (in bytes)
        
        Returns:
            True if store succeeded
            False if insufficient data in buffer (buffer underflow)
        
        Effects on success:
            - Decreases current_usage by num_bytes
            - Increments bytes_stored counter
            - Does not affect peak_usage (only records if exceeded previously)
        
        Note:
            Returns False if trying to store more than buffer contains.
            This represents an error in instruction sequence.
        """
        if self.current_usage < num_bytes:
            return False  # Invalid operation
        
        self.current_usage -= num_bytes
        self.bytes_stored += num_bytes
        return True
    
    def reset(self) -> None:
        """
        Reset buffer state for a new execution.
        
        Clears current_usage but preserves:
            - peak_usage (for post-run analysis)
            - bytes_loaded/stored (for aggregate statistics)
        
        Call before running a new simulation.
        """
        self.current_usage = 0
    
    def get_stats(self) -> MemoryStats:
        """
        Get current memory statistics.
        
        Returns:
            MemoryStats object with:
            - bytes_loaded: Total bytes read from main memory
            - bytes_stored: Total bytes written to main memory
            - buffer_capacity: Maximum capacity
            - peak_usage: Highest occupancy observed
            - current_usage: Current occupancy (usually 0 after execution)
        
        Used for:
            - Compute memory bandwidth (bytes_loaded + bytes_stored)
            - Analyze buffer utilization (peak_usage / capacity)
            - Detect memory pressure (peak_usage near capacity)
        """
        return MemoryStats(
            bytes_loaded=self.bytes_loaded,
            bytes_stored=self.bytes_stored,
            buffer_capacity=self.buffer_capacity,
            peak_usage=self.peak_usage,
            current_usage=self.current_usage,
        )

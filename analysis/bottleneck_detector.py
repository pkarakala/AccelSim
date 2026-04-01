"""
Bottleneck Detector: Identifies likely performance bottlenecks.

Uses simple heuristics to detect:
- Memory bound workloads
- Compute bound workloads
- Poor buffer utilization
- Excessive loads/stores
"""

from typing import Dict, Any


class BottleneckDetector:
    """Detects performance bottlenecks in simulator results."""
    
    def __init__(self, metrics: Dict[str, Any]):
        """
        Initialize detector with performance metrics.
        
        Args:
            metrics: Dictionary from PerformanceAnalyzer.analyze()
        """
        self.metrics = metrics
        self.bottlenecks = []
    
    def detect(self) -> str:
        """
        Detect bottlenecks and return a report.
        
        Returns:
            Formatted text report of detected bottlenecks
        """
        self.bottlenecks = []
        
        # Check for memory bound behavior
        self._check_memory_bound()
        
        # Check for compute bound behavior
        self._check_compute_bound()
        
        # Check for poor buffer utilization
        self._check_buffer_utilization()
        
        # Check for excessive idle cycles
        self._check_idle_cycles()
        
        # Check for memory traffic imbalance
        self._check_memory_balance()
        
        # Generate report
        return self._generate_report()
    
    def _check_memory_bound(self) -> None:
        """Check if workload is memory bound."""
        memory_util = self.metrics.get('memory_utilization', 0)
        compute_util = self.metrics.get('compute_utilization', 0)
        
        if memory_util > 50 and memory_util > compute_util * 1.5:
            self.bottlenecks.append({
                'severity': 'HIGH',
                'type': 'Memory Bound',
                'description': (
                    f"Memory operations dominate execution ({memory_util:.1f}% of cycles). "
                    f"Consider prefetching, compression, or memory hierarchy optimization."
                ),
            })
    
    def _check_compute_bound(self) -> None:
        """Check if workload is compute bound."""
        compute_util = self.metrics.get('compute_utilization', 0)
        memory_util = self.metrics.get('memory_utilization', 0)
        
        if compute_util > 70:
            self.bottlenecks.append({
                'severity': 'MEDIUM',
                'type': 'Compute Bound',
                'description': (
                    f"Compute operations are dominant ({compute_util:.1f}% of cycles). "
                    f"Consider loop tiling, parallelization, or compute unit scaling."
                ),
            })
    
    def _check_buffer_utilization(self) -> None:
        """Check for poor buffer utilization."""
        buffer_util = self.metrics.get('buffer_utilization', 0)
        
        if buffer_util < 20:
            self.bottlenecks.append({
                'severity': 'LOW',
                'type': 'Underutilized Buffer',
                'description': (
                    f"Buffer utilization is low ({buffer_util:.1f}%). "
                    f"Consider batch size adjustments or kernel fusion to maximize buffer usage."
                ),
            })
        elif buffer_util > 85:
            self.bottlenecks.append({
                'severity': 'MEDIUM',
                'type': 'Buffer Pressure',
                'description': (
                    f"Buffer is nearly full ({buffer_util:.1f}%). "
                    f"Risk of spills; consider smaller batch sizes or optimization."
                ),
            })
    
    def _check_idle_cycles(self) -> None:
        """Check for excessive idle cycles."""
        idle_util = self.metrics.get('idle_utilization', 0)
        
        if idle_util > 10:
            self.bottlenecks.append({
                'severity': 'MEDIUM',
                'type': 'High Idle Cycles',
                'description': (
                    f"Idle cycles are significant ({idle_util:.1f}%). "
                    f"Consider instruction pipelining or scheduling optimization."
                ),
            })
    
    def _check_memory_balance(self) -> None:
        """Check for memory traffic imbalance."""
        bytes_loaded = self.metrics.get('bytes_loaded', 0)
        bytes_stored = self.metrics.get('bytes_stored', 0)
        
        if bytes_loaded == 0:
            return
        
        ratio = bytes_stored / bytes_loaded if bytes_loaded > 0 else 0
        
        if ratio < 0.01:
            self.bottlenecks.append({
                'severity': 'LOW',
                'type': 'Load-Heavy Pattern',
                'description': (
                    f"Load traffic far exceeds store traffic ({ratio:.4f} ratio). "
                    f"Typical for inference; optimize read bandwidth."
                ),
            })
        elif ratio > 10:
            self.bottlenecks.append({
                'severity': 'MEDIUM',
                'type': 'Store-Heavy Pattern',
                'description': (
                    f"Store traffic exceeds load traffic ({ratio:.4f} ratio). "
                    f"Consider on-chip buffering or reduce output generation."
                ),
            })
    
    def _generate_report(self) -> str:
        """Generate bottleneck report."""
        lines = [
            "\n" + "=" * 70,
            "BOTTLENECK ANALYSIS",
            "=" * 70,
        ]
        
        if not self.bottlenecks:
            lines.append("")
            lines.append("✓ No significant bottlenecks detected!")
            lines.append("")
        else:
            lines.append(f"\nDetected {len(self.bottlenecks)} bottleneck(s):\n")
            
            for i, bottleneck in enumerate(self.bottlenecks, 1):
                severity = bottleneck['severity']
                label = '⚠ ' if severity == 'MEDIUM' else '! ' if severity == 'HIGH' else '• '
                
                lines.append(f"{label}[{severity:6s}] {bottleneck['type']}")
                lines.append(f"         {bottleneck['description']}")
                lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)

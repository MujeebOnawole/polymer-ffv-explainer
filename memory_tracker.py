import torch
import psutil
from logger import get_logger

class MemoryTracker:
    """Utility for logging system and memory information."""
    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)

    def log_system_info(self):
        """Log basic system information and handle CUDA errors gracefully."""
        self.logger.info("[System    ] Logging hardware info")
        self.logger.info(f"[CPU Count ] {psutil.cpu_count(logical=True)}")
        try:
            if torch.cuda.is_available():
                try:
                    self.logger.info(f"[Cuda     ] {torch.cuda.get_device_name(0)}")
                    total_mem = torch.cuda.get_device_properties(0).total_memory
                    self.logger.info(f"[GPU Mem  ] Total: {total_mem / 1024**2:.1f} MB")
                except Exception as e:
                    self.logger.info(f"[Cuda     ] CUDA error: {str(e)}")
                    self.logger.info("[GPU Mem  ] Unable to query")
            else:
                self.logger.info("[Cuda     ] CUDA not available")
                self.logger.info("[GPU Mem  ] N/A")
        except Exception as e:
            # Catch errors related to CUDA initialization failure
            self.logger.info(f"[Cuda     ] CUDA error: {str(e)}")
            self.logger.info("[GPU Mem  ] Unable to query")

    def log_memory_stats(self, tag=""):
        """Log current CPU and GPU memory usage."""
        mem = psutil.virtual_memory()
        self.logger.info(
            f"[Memory CPU] {tag} {mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB"
        )
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                self.logger.info(
                    f"[Memory GPU] {tag} Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
                )
            except Exception as e:
                self.logger.info(f"[Memory GPU] {tag} query error: {str(e)}")

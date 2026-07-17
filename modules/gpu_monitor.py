"""AMD ROCm GPU Monitor — live GPU metrics for the booth's secondary screen."""

import subprocess
import torch


class GPUMonitor:
    def __init__(self):
        self.has_rocm = torch.cuda.is_available()
        self.device_name = (
            torch.cuda.get_device_name(0) if self.has_rocm else "CPU only"
        )

    def get_status(self):
        """Returns a dict with GPU status info."""
        if not self.has_rocm:
            return {
                "device": "CPU",
                "utilization": "N/A",
                "memory_used": "N/A",
                "memory_total": "N/A",
                "temperature": "N/A",
            }

        info = {
            "device": self.device_name,
            "utilization": "N/A",
            "memory_used": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
            "temperature": "N/A",
        }

        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showtemp", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                if "card0" in data:
                    card = data["card0"]
                    if "GPU use (%)" in card:
                        info["utilization"] = f"{card['GPU use (%)']}%"
                    if "Temperature (C)" in card:
                        info["temperature"] = f"{card['Temperature (C)']}°C"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return info

    def get_status_text(self):
        """Returns a formatted string for display."""
        s = self.get_status()
        lines = [
            f"GPU: {s['device']}",
            f"Utilization: {s['utilization']}",
            f"Memory: {s['memory_used']} / {s['memory_total']}",
            f"Temperature: {s['temperature']}",
        ]
        return "\n".join(lines)

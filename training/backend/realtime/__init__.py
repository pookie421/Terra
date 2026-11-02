"""Global real-time terrain and landslide observation ingestion."""

from .dem_processor import DEMProcessor, TerrainCell, TerrainScan
from .device import detect_compute_device
from .global_ingestor import GlobalIngestor, GlobalScan

__all__ = [
    "DEMProcessor",
    "TerrainCell",
    "TerrainScan",
    "GlobalIngestor",
    "GlobalScan",
    "detect_compute_device",
]

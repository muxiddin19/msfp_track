"""
LITE++ Tracker Integration

Provides tracker wrappers that integrate LITE++ features with popular
multi-object tracking algorithms (DeepSORT, ByteTrack, OC-SORT, etc.).
"""

from .litepp_tracker import LITEPlusPlusTracker

__all__ = ["LITEPlusPlusTracker"]

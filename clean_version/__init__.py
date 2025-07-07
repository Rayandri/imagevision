"""
Package d'analyse de compression pour moteur de recherche d'images
"""

from compression_engine import (
    CompressionMethod,
    PNGCompression, 
    JPEGCompression,
    HaarCompression,
    DCTCompression,
    CompressionEvaluator
)

from search_engine import (
    ImageSearchEngine,
    FeatureExtractor,
    CompressedImageSearchEngine
)

from analysis_tools import CompressionImpactAnalyzer

__version__ = "1.0.0"

__all__ = [
    'CompressionMethod',
    'PNGCompression',
    'JPEGCompression', 
    'HaarCompression',
    'DCTCompression',
    'CompressionEvaluator',
    'ImageSearchEngine',
    'FeatureExtractor',
    'CompressedImageSearchEngine',
    'CompressionImpactAnalyzer'
] 

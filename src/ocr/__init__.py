"""
OCR Module - Document OCR processing and content classification.

This module provides:
- PaddleOCRProcessor: Main OCR processor using PaddleOCR
- ContentClassifier: Classify OCR output into content types (table, form, figure, text)
- BBoxVisualizer: Visualize bounding boxes with color-coded content types
- DocumentLayoutAnalyzer: Analyze document layout structure
- TokenClassifier: Classify individual OCR tokens
"""

from .ocr_processor import PaddleOCRProcessor
from .content_classifier import (
    ContentClassifier,
    ContentType,
    ClassifiedRegion,
    ClassificationResult,
    classify_ocr_output
)
from .bbox_visualizer import (
    BBoxVisualizer,
    visualize_classification,
    create_demo_output
)
from .layout_analyzer import DocumentLayoutAnalyzer
from .token_classifier import TokenClassifier

__all__ = [
    # Main processor
    'PaddleOCRProcessor',
    
    # Content classification
    'ContentClassifier',
    'ContentType',
    'ClassifiedRegion', 
    'ClassificationResult',
    'classify_ocr_output',
    
    # Visualization
    'BBoxVisualizer',
    'visualize_classification',
    'create_demo_output',
    
    # Layout analysis
    'DocumentLayoutAnalyzer',
    
    # Token classification
    'TokenClassifier',
]

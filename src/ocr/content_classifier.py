"""
Content Classifier: Phân loại nội dung OCR output theo loại.

Types:
- table: Bảng dữ liệu có cột và hàng
- form: Form với cặp key:value  
- figure: Chart/plot/biểu đồ
- text: Đoạn văn bản thông thường

Module này cung cấp unified API để phân loại từ OCR output.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .layout_analyzer import DocumentLayoutAnalyzer


class ContentType(Enum):
    """Enum cho các loại nội dung."""
    TABLE = "table"
    FORM = "form"
    FIGURE = "figure"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedRegion:
    """Data class cho một region đã được phân loại."""
    
    region_id: int
    content_type: ContentType
    confidence: float
    bbox: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text_content: str
    lines: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'region_id': self.region_id,
            'content_type': self.content_type.value,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'text_content': self.text_content,
            'lines': self.lines,
            'metadata': self.metadata
        }


@dataclass
class ClassificationResult:
    """Data class cho kết quả phân loại toàn bộ document."""
    
    success: bool
    regions: List[ClassifiedRegion]
    summary: Dict[str, int]  # Count by type
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'regions': [r.to_dict() for r in self.regions],
            'summary': self.summary,
            'error': self.error
        }
    
    def get_regions_by_type(self, content_type: ContentType) -> List[ClassifiedRegion]:
        """Lấy các regions theo loại."""
        return [r for r in self.regions if r.content_type == content_type]
    
    def get_tables(self) -> List[ClassifiedRegion]:
        """Lấy tất cả regions là table."""
        return self.get_regions_by_type(ContentType.TABLE)
    
    def get_forms(self) -> List[ClassifiedRegion]:
        """Lấy tất cả regions là form."""
        return self.get_regions_by_type(ContentType.FORM)
    
    def get_figures(self) -> List[ClassifiedRegion]:
        """Lấy tất cả regions là figure."""
        return self.get_regions_by_type(ContentType.FIGURE)
    
    def get_text_blocks(self) -> List[ClassifiedRegion]:
        """Lấy tất cả regions là text."""
        return self.get_regions_by_type(ContentType.TEXT)


class ContentClassifier:
    """
    Classifier để phân loại nội dung OCR output.
    
    Sử dụng DocumentLayoutAnalyzer để detect và phân loại các vùng trong document
    thành 4 loại chính: table, form, figure, text.
    """
    
    # Confidence thresholds cho từng loại
    DEFAULT_THRESHOLDS = {
        ContentType.TABLE: 0.25,
        ContentType.FORM: 0.20,
        ContentType.FIGURE: 0.25,
        ContentType.TEXT: 0.15,
    }
    
    # Colors cho visualization
    TYPE_COLORS = {
        ContentType.TABLE: '#FF0000',    # Red
        ContentType.FORM: '#0000FF',     # Blue
        ContentType.FIGURE: '#00FF00',   # Green
        ContentType.TEXT: '#FFA500',     # Orange
        ContentType.UNKNOWN: '#808080',  # Gray
    }
    
    def __init__(
        self,
        layout_analyzer: Optional[DocumentLayoutAnalyzer] = None,
        confidence_thresholds: Optional[Dict[ContentType, float]] = None
    ):
        """
        Khởi tạo classifier.
        
        Args:
            layout_analyzer: DocumentLayoutAnalyzer instance (tạo mới nếu None)
            confidence_thresholds: Custom thresholds cho từng loại
        """
        self.layout_analyzer = layout_analyzer or DocumentLayoutAnalyzer()
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        
        if confidence_thresholds:
            self.thresholds.update(confidence_thresholds)
    
    def classify(self, ocr_result: Dict[str, Any]) -> ClassificationResult:
        """
        Phân loại nội dung từ OCR result.
        
        Args:
            ocr_result: Kết quả từ PaddleOCRProcessor.run_ocr()
                       Cần có keys: 'success', 'details' (list tokens với text, box)
        
        Returns:
            ClassificationResult với danh sách regions đã phân loại
        """
        if not ocr_result.get('success', False):
            return ClassificationResult(
                success=False,
                regions=[],
                summary={},
                error=ocr_result.get('error', 'OCR failed')
            )
        
        if not ocr_result.get('details'):
            return ClassificationResult(
                success=False,
                regions=[],
                summary={},
                error='No OCR details found'
            )
        
        try:
            # Run layout analysis
            layout_result = self.layout_analyzer.analyze_layout(ocr_result)
            
            if not layout_result or not layout_result.get('regions'):
                return ClassificationResult(
                    success=True,
                    regions=[],
                    summary={'table': 0, 'form': 0, 'figure': 0, 'text': 0},
                    error=None
                )
            
            # Convert layout regions to ClassifiedRegions
            classified_regions = []
            seen_blocks = set()  # Avoid duplicates
            
            for idx, region in enumerate(layout_result['regions']):
                block_id = region['block_id']
                
                # Skip if we already have this block with higher confidence
                if block_id in seen_blocks:
                    continue
                
                seen_blocks.add(block_id)
                
                # Map region type to ContentType
                region_type = region['region_type']
                content_type = self._map_region_type(region_type)
                
                # Get confidence score
                confidence = region['score']
                
                # Skip if below threshold
                threshold = self.thresholds.get(content_type, 0.2)
                if confidence < threshold:
                    continue
                
                # Extract block info
                block = region['block']
                bbox = block.get('bbox')
                
                # Get text content from lines
                lines = [line['text'] for line in block.get('lines', [])]
                text_content = '\n'.join(lines)
                
                # Build metadata
                metadata = self._build_metadata(region, content_type)
                
                classified_region = ClassifiedRegion(
                    region_id=idx,
                    content_type=content_type,
                    confidence=confidence,
                    bbox=bbox,
                    text_content=text_content,
                    lines=lines,
                    metadata=metadata
                )
                
                classified_regions.append(classified_region)
            
            # Sort by position (top to bottom, left to right)
            classified_regions.sort(key=lambda r: self._sort_key(r.bbox))
            
            # Re-assign region_id after sorting
            for idx, region in enumerate(classified_regions):
                region.region_id = idx
            
            # Build summary
            summary = {
                'table': len([r for r in classified_regions if r.content_type == ContentType.TABLE]),
                'form': len([r for r in classified_regions if r.content_type == ContentType.FORM]),
                'figure': len([r for r in classified_regions if r.content_type == ContentType.FIGURE]),
                'text': len([r for r in classified_regions if r.content_type == ContentType.TEXT]),
            }
            
            return ClassificationResult(
                success=True,
                regions=classified_regions,
                summary=summary,
                error=None
            )
            
        except Exception as e:
            return ClassificationResult(
                success=False,
                regions=[],
                summary={},
                error=str(e)
            )
    
    def classify_with_layout(
        self, 
        ocr_result: Dict[str, Any]
    ) -> Tuple[ClassificationResult, Dict[str, Any]]:
        """
        Phân loại và trả về cả layout result.
        
        Args:
            ocr_result: Kết quả từ OCR
            
        Returns:
            Tuple (ClassificationResult, layout_result)
        """
        if not ocr_result.get('success', False):
            return (
                ClassificationResult(
                    success=False,
                    regions=[],
                    summary={},
                    error=ocr_result.get('error', 'OCR failed')
                ),
                None
            )
        
        layout_result = self.layout_analyzer.analyze_layout(ocr_result)
        classification_result = self.classify(ocr_result)
        
        return classification_result, layout_result
    
    def _map_region_type(self, region_type: str) -> ContentType:
        """Map string region type to ContentType enum."""
        type_mapping = {
            'table': ContentType.TABLE,
            'form': ContentType.FORM,
            'figure': ContentType.FIGURE,
            'text': ContentType.TEXT,
        }
        return type_mapping.get(region_type.lower(), ContentType.UNKNOWN)
    
    def _build_metadata(
        self, 
        region: Dict[str, Any], 
        content_type: ContentType
    ) -> Dict[str, Any]:
        """Build metadata cho region dựa trên loại."""
        metadata = {}
        raw_metadata = region.get('metadata', {})
        
        if content_type == ContentType.TABLE:
            metadata = {
                'num_columns': raw_metadata.get('num_cols', 0),
                'column_stability': raw_metadata.get('col_stability', 0.0),
                'row_spacing_variance': raw_metadata.get('row_spacing_var', 0.0),
            }
        
        elif content_type == ContentType.FORM:
            metadata = {
                'has_keyvalue_pattern': raw_metadata.get('has_keyvalue_pattern', False),
                'keyvalue_ratio': raw_metadata.get('keyvalue_ratio', 0.0),
                'colon_ratio': raw_metadata.get('colon_ratio', 0.0),
                'left_alignment': raw_metadata.get('left_alignment', 0.0),
                'right_alignment': raw_metadata.get('right_alignment', 0.0),
            }
        
        elif content_type == ContentType.FIGURE:
            metadata = {
                'empty_center_ratio': raw_metadata.get('empty_center_ratio', 0.0),
                'tick_like_numbers': raw_metadata.get('tick_like_numbers', 0),
                'has_legend': raw_metadata.get('legend_cluster', False),
            }
        
        elif content_type == ContentType.TEXT:
            metadata = {
                'avg_line_length': raw_metadata.get('avg_line_length', 0.0),
                'spacing_uniformity': raw_metadata.get('spacing_uniformity', 0.0),
                'justified_boost': raw_metadata.get('justified_boost', 0.0),
            }
        
        return metadata
    
    def _sort_key(self, bbox: List[List[float]]) -> Tuple[float, float]:
        """Generate sort key từ bbox (top to bottom, left to right)."""
        if not bbox:
            return (float('inf'), float('inf'))
        
        y_coords = [p[1] for p in bbox]
        x_coords = [p[0] for p in bbox]
        return (min(y_coords), min(x_coords))
    
    @staticmethod
    def get_type_color(content_type: ContentType) -> str:
        """Lấy màu cho loại content."""
        return ContentClassifier.TYPE_COLORS.get(content_type, '#808080')
    
    @staticmethod
    def get_type_name(content_type: ContentType) -> str:
        """Lấy tên hiển thị cho loại content."""
        names = {
            ContentType.TABLE: "Table (Bảng)",
            ContentType.FORM: "Form (Key:Value)",
            ContentType.FIGURE: "Figure (Chart/Plot)",
            ContentType.TEXT: "Text (Paragraph)",
            ContentType.UNKNOWN: "Unknown",
        }
        return names.get(content_type, "Unknown")


def classify_ocr_output(
    ocr_result: Dict[str, Any],
    confidence_thresholds: Optional[Dict[str, float]] = None
) -> ClassificationResult:
    """
    Convenience function để phân loại OCR output.
    
    Args:
        ocr_result: Kết quả từ PaddleOCRProcessor.run_ocr()
        confidence_thresholds: Optional thresholds {'table': 0.3, 'form': 0.2, ...}
    
    Returns:
        ClassificationResult
    
    Example:
        >>> from src.ocr.ocr_processor import PaddleOCRProcessor
        >>> from src.ocr.content_classifier import classify_ocr_output
        >>> 
        >>> processor = PaddleOCRProcessor()
        >>> ocr_result = processor.run_ocr("document.png")
        >>> classification = classify_ocr_output(ocr_result)
        >>> 
        >>> print(f"Found {classification.summary['table']} tables")
        >>> for table in classification.get_tables():
        ...     print(f"Table: {table.text_content[:50]}...")
    """
    # Convert string keys to ContentType
    thresholds = None
    if confidence_thresholds:
        thresholds = {}
        for key, value in confidence_thresholds.items():
            if isinstance(key, str):
                content_type = ContentType(key) if key in [e.value for e in ContentType] else None
                if content_type:
                    thresholds[content_type] = value
            elif isinstance(key, ContentType):
                thresholds[key] = value
    
    classifier = ContentClassifier(confidence_thresholds=thresholds)
    return classifier.classify(ocr_result)

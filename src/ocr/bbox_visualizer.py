"""
Bounding Box Visualizer: Vẽ bounding boxes với màu sắc theo loại nội dung.

Module này cung cấp các hàm để:
- Vẽ bounding boxes lên ảnh với màu sắc theo loại (table, form, figure, text)
- Vẽ labels và confidence scores
- Export ảnh với annotations
"""

import os
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .content_classifier import (
    ContentType, 
    ClassifiedRegion, 
    ClassificationResult,
    ContentClassifier
)


class BBoxVisualizer:
    """
    Visualizer để vẽ bounding boxes với màu sắc theo loại nội dung.
    """
    
    # Default colors (RGB format for PIL)
    DEFAULT_COLORS = {
        ContentType.TABLE: (255, 0, 0),      # Red
        ContentType.FORM: (0, 0, 255),       # Blue
        ContentType.FIGURE: (0, 200, 0),     # Green
        ContentType.TEXT: (255, 165, 0),     # Orange
        ContentType.UNKNOWN: (128, 128, 128), # Gray
    }
    
    # Color names for legend
    COLOR_NAMES = {
        ContentType.TABLE: "Red",
        ContentType.FORM: "Blue",
        ContentType.FIGURE: "Green",
        ContentType.TEXT: "Orange",
        ContentType.UNKNOWN: "Gray",
    }
    
    def __init__(
        self,
        colors: Optional[Dict[ContentType, Tuple[int, int, int]]] = None,
        line_width: int = 3,
        font_size: int = 16,
        show_confidence: bool = True,
        show_label: bool = True,
        alpha: float = 0.3
    ):
        """
        Khởi tạo visualizer.
        
        Args:
            colors: Custom colors cho từng loại (RGB tuples)
            line_width: Độ dày của bounding box
            font_size: Kích thước font cho labels
            show_confidence: Có hiển thị confidence score không
            show_label: Có hiển thị label (type) không
            alpha: Độ trong suốt của fill (0-1)
        """
        self.colors = self.DEFAULT_COLORS.copy()
        if colors:
            self.colors.update(colors)
        
        self.line_width = line_width
        self.font_size = font_size
        self.show_confidence = show_confidence
        self.show_label = show_label
        self.alpha = alpha
        
        # Try to load a good font
        self._font = self._load_font(font_size)
    
    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load font, fallback to default if not available."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "arial.ttf",
            "Arial.ttf",
        ]
        
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except (IOError, OSError):
                continue
        
        # Fallback to default
        try:
            return ImageFont.load_default()
        except:
            return None
    
    def draw_classified_regions(
        self,
        image: Union[str, Image.Image, np.ndarray],
        classification_result: ClassificationResult,
        fill: bool = False
    ) -> Image.Image:
        """
        Vẽ các classified regions lên ảnh.
        
        Args:
            image: Đường dẫn ảnh, PIL Image, hoặc numpy array
            classification_result: Kết quả từ ContentClassifier
            fill: Có fill màu bên trong bbox không
            
        Returns:
            PIL Image với bounding boxes
        """
        # Load image
        img = self._load_image(image)
        
        if not classification_result.success or not classification_result.regions:
            return img
        
        # Create drawing context
        draw = ImageDraw.Draw(img, 'RGBA')
        
        for region in classification_result.regions:
            self._draw_region(draw, region, fill)
        
        return img
    
    def draw_regions_by_type(
        self,
        image: Union[str, Image.Image, np.ndarray],
        classification_result: ClassificationResult,
        content_types: List[ContentType],
        fill: bool = False
    ) -> Image.Image:
        """
        Vẽ chỉ các regions của một số loại cụ thể.
        
        Args:
            image: Input image
            classification_result: Classification result
            content_types: List các loại cần vẽ
            fill: Có fill không
            
        Returns:
            PIL Image
        """
        img = self._load_image(image)
        
        if not classification_result.success:
            return img
        
        draw = ImageDraw.Draw(img, 'RGBA')
        
        for region in classification_result.regions:
            if region.content_type in content_types:
                self._draw_region(draw, region, fill)
        
        return img
    
    def draw_single_region(
        self,
        image: Union[str, Image.Image, np.ndarray],
        region: ClassifiedRegion,
        fill: bool = False
    ) -> Image.Image:
        """
        Vẽ một region đơn lẻ.
        
        Args:
            image: Input image
            region: ClassifiedRegion to draw
            fill: Có fill không
            
        Returns:
            PIL Image
        """
        img = self._load_image(image)
        draw = ImageDraw.Draw(img, 'RGBA')
        self._draw_region(draw, region, fill)
        return img
    
    def create_comparison_view(
        self,
        image: Union[str, Image.Image, np.ndarray],
        classification_result: ClassificationResult
    ) -> Image.Image:
        """
        Tạo ảnh so sánh: Original | OCR Boxes | Classified Regions.
        
        Args:
            image: Input image
            classification_result: Classification result
            
        Returns:
            PIL Image với 3 panels
        """
        img = self._load_image(image)
        width, height = img.size
        
        # Create composite image (3 panels)
        composite = Image.new('RGB', (width * 3 + 20, height + 80), color=(255, 255, 255))
        
        # Panel 1: Original
        composite.paste(img, (0, 40))
        
        # Panel 2: OCR Boxes (all boxes same color)
        img_ocr = img.copy()
        draw_ocr = ImageDraw.Draw(img_ocr)
        for region in classification_result.regions:
            if region.bbox:
                points = [(int(p[0]), int(p[1])) for p in region.bbox]
                draw_ocr.polygon(points, outline=(100, 100, 100), width=2)
        composite.paste(img_ocr, (width + 10, 40))
        
        # Panel 3: Classified (color-coded)
        img_classified = self.draw_classified_regions(img.copy(), classification_result, fill=True)
        composite.paste(img_classified, (width * 2 + 20, 40))
        
        # Add titles
        draw_composite = ImageDraw.Draw(composite)
        title_font = self._load_font(20) or self._font
        
        titles = ["Original", "OCR Detected", "Content Classified"]
        x_positions = [width // 2, width + 10 + width // 2, width * 2 + 20 + width // 2]
        
        for title, x in zip(titles, x_positions):
            bbox = draw_composite.textbbox((0, 0), title, font=title_font) if title_font else (0, 0, len(title) * 10, 20)
            text_width = bbox[2] - bbox[0]
            draw_composite.text((x - text_width // 2, 10), title, fill=(0, 0, 0), font=title_font)
        
        # Add legend at bottom
        self._draw_legend(draw_composite, 10, height + 50, classification_result)
        
        return composite
    
    def create_type_grid(
        self,
        image: Union[str, Image.Image, np.ndarray],
        classification_result: ClassificationResult
    ) -> Image.Image:
        """
        Tạo grid view với mỗi loại content trên một panel riêng.
        
        Args:
            image: Input image
            classification_result: Classification result
            
        Returns:
            PIL Image với 2x2 grid (table, form, figure, text)
        """
        img = self._load_image(image)
        width, height = img.size
        
        # Create 2x2 grid
        grid_width = width * 2 + 20
        grid_height = height * 2 + 80
        grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
        
        types_and_positions = [
            (ContentType.TABLE, 0, 40, "TABLE (Red)"),
            (ContentType.FORM, width + 20, 40, "FORM (Blue)"),
            (ContentType.FIGURE, 0, height + 60, "FIGURE (Green)"),
            (ContentType.TEXT, width + 20, height + 60, "TEXT (Orange)"),
        ]
        
        for content_type, x, y, title in types_and_positions:
            # Draw only regions of this type
            panel = self.draw_regions_by_type(img.copy(), classification_result, [content_type], fill=True)
            grid.paste(panel, (x, y))
            
            # Add title
            draw = ImageDraw.Draw(grid)
            title_font = self._load_font(16) or self._font
            draw.text((x + 10, y - 25), title, fill=self.colors[content_type], font=title_font)
        
        return grid
    
    def _draw_region(
        self,
        draw: ImageDraw.Draw,
        region: ClassifiedRegion,
        fill: bool = False
    ) -> None:
        """Vẽ một region lên draw context."""
        if not region.bbox:
            return
        
        # Get color
        color = self.colors.get(region.content_type, self.DEFAULT_COLORS[ContentType.UNKNOWN])
        
        # Convert bbox to points
        points = [(int(p[0]), int(p[1])) for p in region.bbox]
        
        # Draw fill if requested
        if fill:
            fill_color = (*color, int(self.alpha * 255))  # RGBA
            draw.polygon(points, fill=fill_color)
        
        # Draw outline
        draw.polygon(points, outline=color, width=self.line_width)
        
        # Draw label
        if self.show_label or self.show_confidence:
            label_parts = []
            
            if self.show_label:
                label_parts.append(region.content_type.value.upper())
            
            if self.show_confidence:
                label_parts.append(f"{region.confidence:.2f}")
            
            label = " | ".join(label_parts)
            
            # Position label INSIDE the bbox at top-left corner
            x_min = min(p[0] for p in points)
            y_min = min(p[1] for p in points)
            
            # Label position: inside the box with small padding
            label_x = x_min + 4
            label_y = y_min + 4
            
            # Draw label background
            if self._font:
                bbox = draw.textbbox((label_x, label_y), label, font=self._font)
                padding = 2
                draw.rectangle(
                    [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
                    fill=(255, 255, 255, 220)
                )
                draw.text((label_x, label_y), label, fill=color, font=self._font)
            else:
                draw.text((label_x, label_y), label, fill=color)
    
    def _draw_legend(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        classification_result: ClassificationResult
    ) -> None:
        """Vẽ legend cho các loại content."""
        legend_items = [
            (ContentType.TABLE, f"Table ({classification_result.summary.get('table', 0)})"),
            (ContentType.FORM, f"Form ({classification_result.summary.get('form', 0)})"),
            (ContentType.FIGURE, f"Figure ({classification_result.summary.get('figure', 0)})"),
            (ContentType.TEXT, f"Text ({classification_result.summary.get('text', 0)})"),
        ]
        
        current_x = x
        for content_type, label in legend_items:
            color = self.colors[content_type]
            
            # Draw color box
            draw.rectangle([current_x, y, current_x + 20, y + 20], fill=color, outline=(0, 0, 0))
            
            # Draw label
            if self._font:
                draw.text((current_x + 25, y + 2), label, fill=(0, 0, 0), font=self._font)
                bbox = draw.textbbox((current_x + 25, y), label, font=self._font)
                current_x = bbox[2] + 30
            else:
                draw.text((current_x + 25, y), label, fill=(0, 0, 0))
                current_x += 25 + len(label) * 8 + 30
    
    def _load_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Load và convert image sang PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                # Grayscale
                return Image.fromarray(image).convert('RGB')
            elif image.shape[2] == 4:
                # RGBA
                return Image.fromarray(image).convert('RGB')
            else:
                # BGR to RGB (OpenCV format)
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def save_visualization(
        self,
        image: Union[str, Image.Image, np.ndarray],
        classification_result: ClassificationResult,
        output_path: str,
        fill: bool = True
    ) -> str:
        """
        Vẽ và lưu visualization ra file.
        
        Args:
            image: Input image
            classification_result: Classification result
            output_path: Đường dẫn output
            fill: Có fill regions không
            
        Returns:
            Đường dẫn file đã lưu
        """
        img = self.draw_classified_regions(image, classification_result, fill=fill)
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        img.save(output_path)
        return output_path
    
    def save_comparison(
        self,
        image: Union[str, Image.Image, np.ndarray],
        classification_result: ClassificationResult,
        output_path: str
    ) -> str:
        """
        Tạo và lưu comparison view.
        
        Args:
            image: Input image
            classification_result: Classification result
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn file đã lưu
        """
        img = self.create_comparison_view(image, classification_result)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        img.save(output_path)
        return output_path


def visualize_classification(
    image_path: str,
    ocr_result: Dict[str, Any],
    output_path: Optional[str] = None,
    show: bool = False
) -> Tuple[Image.Image, ClassificationResult]:
    """
    Convenience function để visualize classification results.
    
    Args:
        image_path: Đường dẫn đến ảnh gốc
        ocr_result: Kết quả từ PaddleOCRProcessor.run_ocr()
        output_path: Đường dẫn lưu output (optional)
        show: Có hiển thị ảnh không (requires matplotlib)
        
    Returns:
        Tuple (PIL Image, ClassificationResult)
    
    Example:
        >>> from src.ocr.ocr_processor import PaddleOCRProcessor
        >>> from src.ocr.bbox_visualizer import visualize_classification
        >>> 
        >>> processor = PaddleOCRProcessor()
        >>> ocr_result = processor.run_ocr("document.png")
        >>> img, result = visualize_classification(
        ...     "document.png", 
        ...     ocr_result,
        ...     output_path="output/classified.png"
        ... )
        >>> print(result.summary)
    """
    from .content_classifier import ContentClassifier
    
    # Classify
    classifier = ContentClassifier()
    classification_result = classifier.classify(ocr_result)
    
    # Visualize
    visualizer = BBoxVisualizer()
    img = visualizer.draw_classified_regions(image_path, classification_result, fill=True)
    
    # Save if requested
    if output_path:
        visualizer.save_visualization(image_path, classification_result, output_path)
        print(f"✅ Saved visualization to: {output_path}")
    
    # Show if requested
    if show:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title("Content Classification Result")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available for display")
    
    return img, classification_result


def create_demo_output(
    image_path: str,
    ocr_result: Dict[str, Any],
    output_dir: str
) -> Dict[str, str]:
    """
    Tạo bộ demo outputs đầy đủ.
    
    Args:
        image_path: Đường dẫn ảnh gốc
        ocr_result: OCR result
        output_dir: Thư mục output
        
    Returns:
        Dict với paths của các output files
    """
    from .content_classifier import ContentClassifier
    
    # Classify
    classifier = ContentClassifier()
    classification_result = classifier.classify(ocr_result)
    
    # Create visualizer
    visualizer = BBoxVisualizer()
    
    # Ensure output dir exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get base filename
    base_name = Path(image_path).stem
    
    outputs = {}
    
    # 1. Simple visualization
    simple_path = os.path.join(output_dir, f"{base_name}_classified.png")
    visualizer.save_visualization(image_path, classification_result, simple_path, fill=True)
    outputs['classified'] = simple_path
    
    # 2. Comparison view
    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    visualizer.save_comparison(image_path, classification_result, comparison_path)
    outputs['comparison'] = comparison_path
    
    # 3. Type grid
    grid_path = os.path.join(output_dir, f"{base_name}_type_grid.png")
    grid_img = visualizer.create_type_grid(image_path, classification_result)
    grid_img.save(grid_path)
    outputs['type_grid'] = grid_path
    
    # 4. Save JSON result
    import json
    json_path = os.path.join(output_dir, f"{base_name}_classification.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(classification_result.to_dict(), f, indent=2, ensure_ascii=False)
    outputs['json'] = json_path
    
    print(f"\n✅ Demo outputs created in: {output_dir}")
    print(f"   - Classified: {simple_path}")
    print(f"   - Comparison: {comparison_path}")
    print(f"   - Type Grid: {grid_path}")
    print(f"   - JSON: {json_path}")
    
    return outputs

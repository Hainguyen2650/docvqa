"""
Module phân tích layout của document từ kết quả OCR.
Bao gồm:
- Group token → line → block
- Phát hiện region types (table, form, figure, text)
- Trích xuất cấu trúc theo từng loại region

Classification Method:
- TABLE: Multiple columns aligned along x-axis, evenly spaced rows
- FORM: Key:value pairs (Branch A: explicit pattern, Branch B: alignment-based)
- FIGURE: Charts/plots with empty center, tick-like numbers, legend clusters
- TEXT: Running paragraphs with uniform spacing, few columns
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import DBSCAN
import re


class DocumentLayoutAnalyzer:
    """Class phân tích layout document từ OCR results."""
    
    def __init__(
        self,
        y_overlap_threshold: float = 0.5,
        line_height_tolerance: float = 0.3,
        max_x_gap_ratio: float = 3.0,
        block_vertical_gap: float = 20,
        block_x_overlap_threshold: float = 0.3
    ):
        """
        Khởi tạo analyzer.
        
        Args:
            y_overlap_threshold: Ngưỡng overlap theo y để gom token vào line
            line_height_tolerance: Độ sai lệch y_center so với median height
            max_x_gap_ratio: Tỉ lệ khoảng cách x tối đa (x median char width)
            block_vertical_gap: Khoảng cách vertical tối đa (pixels) để gom lines vào block
            block_x_overlap_threshold: Ngưỡng overlap x để gom lines vào block
        """
        self.y_overlap_threshold = y_overlap_threshold
        self.line_height_tolerance = line_height_tolerance
        self.max_x_gap_ratio = max_x_gap_ratio
        self.block_vertical_gap = block_vertical_gap
        self.block_x_overlap_threshold = block_x_overlap_threshold
    
    @staticmethod
    def bbox_overlap_y(bbox1: List, bbox2: List) -> float:
        """Tính overlap ratio theo trục y của 2 bbox."""
        y1_min = min(p[1] for p in bbox1)
        y1_max = max(p[1] for p in bbox1)
        y2_min = min(p[1] for p in bbox2)
        y2_max = max(p[1] for p in bbox2)
        
        overlap_start = max(y1_min, y2_min)
        overlap_end = min(y1_max, y2_max)
        overlap = max(0, overlap_end - overlap_start)
        
        h1 = y1_max - y1_min
        h2 = y2_max - y2_min
        
        if h1 == 0 or h2 == 0:
            return 0.0
        
        return overlap / min(h1, h2)
    
    @staticmethod
    def bbox_center(bbox: List) -> Tuple[float, float]:
        """Tính center (x, y) của bbox."""
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return (sum(x_coords) / 4, sum(y_coords) / 4)
    
    @staticmethod
    def bbox_bounds(bbox: List) -> Tuple[float, float, float, float]:
        """Trả về (x_min, y_min, x_max, y_max) của bbox."""
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    @staticmethod
    def estimate_char_width(tokens: List[Dict]) -> float:
        """Ước tính độ rộng trung bình của một ký tự."""
        char_widths = []
        for tok in tokens:
            if tok.get('box') and tok.get('text'):
                x_min, _, x_max, _ = DocumentLayoutAnalyzer.bbox_bounds(tok['box'])
                width = x_max - x_min
                num_chars = len(tok['text'])
                if num_chars > 0:
                    char_widths.append(width / num_chars)
        
        return np.median(char_widths) if char_widths else 10.0
    
    @staticmethod
    def estimate_line_height(tokens: List[Dict]) -> float:
        """Ước tính chiều cao trung bình của line."""
        heights = []
        for tok in tokens:
            if tok.get('box'):
                _, y_min, _, y_max = DocumentLayoutAnalyzer.bbox_bounds(tok['box'])
                heights.append(y_max - y_min)
        
        return np.median(heights) if heights else 20.0
    
    @staticmethod
    def detect_keyvalue_pattern(lines: List[Dict]) -> Dict:
        """
        Phát hiện pattern "key: value" trong lines.
        
        Returns:
            {
                'has_pattern': bool,
                'keyvalue_ratio': float,
                'avg_key_length': float,
                'keyvalue_lines': List[str]
            }
        """
        if not lines:
            return {'has_pattern': False, 'keyvalue_ratio': 0.0, 'avg_key_length': 0.0, 'keyvalue_lines': []}
        
        # Pattern: "key: value" hoặc "key : value"
        keyvalue_pattern = re.compile(r'^([^:]{1,50}):\s*(.+)$')
        
        keyvalue_lines = []
        key_lengths = []
        
        for line in lines:
            text = line.get('text', '').strip()
            match = keyvalue_pattern.match(text)
            
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                if len(key) > 0 and len(key) < 50 and len(value) > 0:
                    keyvalue_lines.append(text)
                    key_lengths.append(len(key))
        
        keyvalue_ratio = len(keyvalue_lines) / len(lines) if lines else 0.0
        avg_key_length = np.mean(key_lengths) if key_lengths else 0.0
        
        # Pattern detected if >= 40% lines are key:value with at least 2 matches
        has_pattern = (keyvalue_ratio >= 0.4 and len(keyvalue_lines) >= 2)
        
        return {
            'has_pattern': has_pattern,
            'keyvalue_ratio': keyvalue_ratio,
            'avg_key_length': avg_key_length,
            'keyvalue_lines': keyvalue_lines
        }
    
    def group_tokens_to_lines(self, ocr_result: Dict[str, Any]) -> List[Dict]:
        """Gom các token thành lines dựa trên y-overlap và x-distance."""
        if not ocr_result.get('success') or not ocr_result.get('details'):
            return []
        
        tokens = ocr_result['details']
        n_tokens = len(tokens)
        
        median_height = self.estimate_line_height(tokens)
        median_char_width = self.estimate_char_width(tokens)
        max_x_gap = self.max_x_gap_ratio * median_char_width
        
        # Union-find
        parent = list(range(n_tokens))
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj
        
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                if not tokens[i].get('box') or not tokens[j].get('box'):
                    continue
                
                y_overlap = self.bbox_overlap_y(tokens[i]['box'], tokens[j]['box'])
                cx_i, cy_i = self.bbox_center(tokens[i]['box'])
                cx_j, cy_j = self.bbox_center(tokens[j]['box'])
                y_diff = abs(cy_i - cy_j)
                
                x_i_min, _, x_i_max, _ = self.bbox_bounds(tokens[i]['box'])
                x_j_min, _, x_j_max, _ = self.bbox_bounds(tokens[j]['box'])
                x_gap = min(abs(x_i_max - x_j_min), abs(x_j_max - x_i_min))
                
                if (y_overlap >= self.y_overlap_threshold or 
                    y_diff < self.line_height_tolerance * median_height) and x_gap < max_x_gap:
                    union(i, j)
        
        groups = defaultdict(list)
        for i in range(n_tokens):
            groups[find(i)].append(i)
        
        lines = []
        for group_indices in groups.values():
            line_tokens = [tokens[i] for i in sorted(group_indices, 
                key=lambda idx: self.bbox_center(tokens[idx]['box'])[0] if tokens[idx].get('box') else 0)]
            
            line_text = ' '.join(tok.get('text', '') for tok in line_tokens)
            
            all_coords = []
            for tok in line_tokens:
                if tok.get('box'):
                    all_coords.extend(tok['box'])
            
            if all_coords:
                x_coords = [p[0] for p in all_coords]
                y_coords = [p[1] for p in all_coords]
                line_bbox = [
                    [min(x_coords), min(y_coords)],
                    [max(x_coords), min(y_coords)],
                    [max(x_coords), max(y_coords)],
                    [min(x_coords), max(y_coords)]
                ]
            else:
                line_bbox = None
            
            lines.append({
                'text': line_text,
                'bbox': line_bbox,
                'tokens': line_tokens,
                'token_ids': group_indices
            })
        
        lines.sort(key=lambda line: self.bbox_center(line['bbox'])[1] if line.get('bbox') else 0)
        return lines
    
    def group_lines_to_blocks(self, lines: List[Dict]) -> List[Dict]:
        """Gom các lines thành blocks dựa trên vertical gap và x-overlap."""
        if not lines:
            return []
        
        n_lines = len(lines)
        vertical_gap_threshold = self.block_vertical_gap
        
        line_widths = []
        for line in lines:
            if line.get('bbox'):
                x_min, _, x_max, _ = self.bbox_bounds(line['bbox'])
                line_widths.append(x_max - x_min)
        
        median_line_width = np.median(line_widths) if line_widths else 100.0
        x1_close_threshold = 0.05 * median_line_width
        
        adj = defaultdict(list)
        
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                if not lines[i].get('bbox') or not lines[j].get('bbox'):
                    continue
                
                xi_min, yi_min, xi_max, yi_max = self.bbox_bounds(lines[i]['bbox'])
                xj_min, yj_min, xj_max, yj_max = self.bbox_bounds(lines[j]['bbox'])
                
                v_gap = min(abs(yi_max - yj_min), abs(yj_max - yi_min))
                
                x_overlap_start = max(xi_min, xj_min)
                x_overlap_end = min(xi_max, xj_max)
                x_overlap = max(0, x_overlap_end - x_overlap_start)
                min_width = min(xi_max - xi_min, xj_max - xj_min)
                x_overlap_ratio = x_overlap / min_width if min_width > 0 else 0
                
                x1_close = abs(xi_min - xj_min) < x1_close_threshold
                
                if v_gap < vertical_gap_threshold and \
                   (x_overlap_ratio >= self.block_x_overlap_threshold or x1_close):
                    adj[i].append(j)
                    adj[j].append(i)
        
        visited = [False] * n_lines
        blocks = []
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(n_lines):
            if not visited[i]:
                component = []
                dfs(i, component)
                
                block_lines = [lines[idx] for idx in sorted(component, 
                    key=lambda idx: self.bbox_center(lines[idx]['bbox'])[1] if lines[idx].get('bbox') else 0)]
                
                all_coords = []
                for line in block_lines:
                    if line.get('bbox'):
                        all_coords.extend(line['bbox'])
                
                if all_coords:
                    x_coords = [p[0] for p in all_coords]
                    y_coords = [p[1] for p in all_coords]
                    block_bbox = [
                        [min(x_coords), min(y_coords)],
                        [max(x_coords), min(y_coords)],
                        [max(x_coords), max(y_coords)],
                        [min(x_coords), max(y_coords)]
                    ]
                else:
                    block_bbox = None
                
                keyvalue_info = self.detect_keyvalue_pattern(block_lines)
                
                blocks.append({
                    'lines': block_lines,
                    'bbox': block_bbox,
                    'line_ids': component,
                    'keyvalue_pattern': keyvalue_info
                })
        
        return blocks
    
    def detect_table_region(self, block: Dict) -> Dict:
        """
        Phát hiện TABLE region từ block.
        
        Algorithm:
        1. Collect all x_centers of tokens within block
        2. Use DBSCAN clustering to estimate columns:
           - eps = max(30, 3 * median_char_width) (adaptive to font size)
           - num_cols_est = number of clusters (excluding noise -1)
        3. Calculate col_stability: ratio of lines with >= 2 tokens aligned to columns
           - alignment_threshold = max(30, 3 * median_char_width)
        4. Calculate row_spacing_var: variance of row spacing (std/mean of y_gaps)
        
        Table conditions:
        - num_cols_est >= 3
        - col_stability >= 0.6
        - row_spacing_var < 0.3
        
        Score: increases with columns and alignment, decreases with spacing variance
        """
        lines = block.get('lines', [])
        if len(lines) < 2:
            return {'is_table': False, 'score': 0.0, 'num_cols': 0, 'col_stability': 0.0, 'row_spacing_var': 1.0}
        
        # Collect all token x_centers
        x_centers = []
        all_tokens = []
        for line in lines:
            for tok in line.get('tokens', []):
                if tok.get('box'):
                    cx, _ = self.bbox_center(tok['box'])
                    x_centers.append(cx)
                    all_tokens.append(tok)
        
        if len(x_centers) < 5:
            return {'is_table': False, 'score': 0.0, 'num_cols': 0, 'col_stability': 0.0, 'row_spacing_var': 1.0}
        
        # Adaptive eps based on font size
        median_char_width = self.estimate_char_width(all_tokens)
        clustering_eps = max(30, 3 * median_char_width)
        
        # DBSCAN clustering on x_centers to find columns
        X = np.array(x_centers).reshape(-1, 1)
        clustering = DBSCAN(eps=clustering_eps, min_samples=2).fit(X)
        labels = clustering.labels_
        num_cols_est = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Calculate column centers
        col_centers = []
        for label in set(labels):
            if label != -1:
                cluster_points = X[labels == label]
                col_centers.append(np.mean(cluster_points))
        col_centers = sorted(col_centers)
        
        # Alignment threshold (same as eps)
        alignment_threshold = max(30, 3 * median_char_width)
        
        # col_stability: proportion of lines where >= 2 tokens are "near" column centers
        aligned_lines = 0
        for line in lines:
            aligned_tokens_count = 0
            for tok in line.get('tokens', []):
                if tok.get('box') and col_centers:
                    cx, _ = self.bbox_center(tok['box'])
                    closest_dist = min(abs(cx - cc) for cc in col_centers)
                    if closest_dist < alignment_threshold:
                        aligned_tokens_count += 1
            
            if aligned_tokens_count >= 2:
                aligned_lines += 1
        
        col_stability = aligned_lines / len(lines) if lines else 0.0
        
        # row_spacing_var: std/mean of y_gaps
        y_centers = [self.bbox_center(line['bbox'])[1] for line in lines if line.get('bbox')]
        if len(y_centers) >= 2:
            y_centers_sorted = sorted(y_centers)
            y_gaps = [y_centers_sorted[i+1] - y_centers_sorted[i] for i in range(len(y_centers_sorted)-1)]
            mean_gap = np.mean(y_gaps)
            row_spacing_var = np.std(y_gaps) / mean_gap if mean_gap > 0 else 1.0
        else:
            row_spacing_var = 1.0
        
        # Table conditions
        is_table = (num_cols_est >= 3 and col_stability >= 0.6 and row_spacing_var < 0.3)
        
        # Score: increases with columns and alignment, decreases with spacing variance
        # Normalize num_cols contribution (cap at 10 columns)
        cols_score = min(1.0, num_cols_est / 10.0)
        variance_penalty = 1.0 - min(1.0, row_spacing_var)
        score = cols_score * 0.3 + col_stability * 0.5 + variance_penalty * 0.2
        
        return {
            'is_table': is_table,
            'score': score,
            'num_cols': num_cols_est,
            'col_stability': col_stability,
            'row_spacing_var': row_spacing_var,
            'alignment_threshold': alignment_threshold
        }
    
    def detect_form_region(self, block: Dict) -> Dict:
        """
        Phát hiện FORM region (key:value pairs) từ block.
        
        Two branches:
        
        Branch A (high priority): 
        - If keyvalue_pattern.has_pattern=True (detected "key: value" patterns)
        - Immediately conclude form with high score (~0.9+)
        - Set text_boost_score to negative to avoid misclassification as text
        
        Branch B (fallback):
        - Measure left_alignment and right_alignment (1 - std/mean of x_min/x_max)
        - Statistics on line length (chars), short/long line ratios, variance
        - Distinguish Form vs Justified text:
          * Justified text: both margins aligned, many long lines, high avg_line_length
          * Form: many short lines, varying length (short key, long value), contains ":"
        
        Returns text_boost_score for use by text detector.
        """
        lines = block.get('lines', [])
        if len(lines) < 2:
            return {
                'is_form': False, 'score': 0.0, 'text_boost_score': 0.0,
                'has_keyvalue_pattern': False, 'keyvalue_ratio': 0.0,
                'colon_ratio': 0.0, 'left_alignment': 0.0, 'right_alignment': 0.0,
                'avg_line_length': 0.0, 'line_length_variance': 0.0,
                'short_line_ratio': 0.0, 'long_line_ratio': 0.0, 'is_justified_text': False
            }
        
        # ===== BRANCH A: Key:Value Pattern Detection =====
        keyvalue_info = block.get('keyvalue_pattern', {})
        has_keyvalue_pattern = keyvalue_info.get('has_pattern', False)
        keyvalue_ratio = keyvalue_info.get('keyvalue_ratio', 0.0)
        
        if has_keyvalue_pattern:
            # High confidence form detection
            return {
                'is_form': True,
                'score': 0.9 + keyvalue_ratio * 0.1,  # Score ~0.9-1.0
                'text_boost_score': -0.3,  # Negative penalty for text detection
                'has_keyvalue_pattern': True,
                'keyvalue_ratio': keyvalue_ratio,
                'colon_ratio': keyvalue_ratio,
                'left_alignment': 1.0,
                'right_alignment': 0.0,
                'avg_line_length': keyvalue_info.get('avg_key_length', 0.0),
                'line_length_variance': 0.0,
                'short_line_ratio': 0.0,
                'long_line_ratio': 0.0,
                'is_justified_text': False
            }
        
        # ===== BRANCH B: Alignment-based Detection =====
        # Calculate colon ratio from keyvalue_ratio
        colon_ratio = keyvalue_ratio
        
        # Left alignment: stability of x_min positions (1 - std/mean)
        left_x = [self.bbox_bounds(line['bbox'])[0] for line in lines if line.get('bbox')]
        left_alignment = 0.0
        if len(left_x) >= 2:
            left_mean = np.mean(left_x)
            left_std = np.std(left_x)
            left_alignment = 1.0 - min(1.0, left_std / max(left_mean, 1.0))
        
        # Right alignment: stability of x_max positions (1 - std/mean)
        right_x = [self.bbox_bounds(line['bbox'])[2] for line in lines if line.get('bbox')]
        right_alignment = 0.0
        if len(right_x) >= 2:
            right_mean = np.mean(right_x)
            right_std = np.std(right_x)
            right_alignment = 1.0 - min(1.0, right_std / max(right_mean, 1.0))
        
        # Line length statistics (in characters)
        line_lengths = [len(line.get('text', '')) for line in lines]
        avg_line_length = np.mean(line_lengths) if line_lengths else 0.0
        line_length_variance = np.std(line_lengths) / max(avg_line_length, 1.0) if line_lengths else 0.0
        
        # Short/long line ratios
        short_lines = sum(1 for l in line_lengths if l < 40)
        long_lines = sum(1 for l in line_lengths if l > 60)
        short_line_ratio = short_lines / len(lines) if lines else 0.0
        long_line_ratio = long_lines / len(lines) if lines else 0.0
        
        # ===== Distinguish Form vs Justified Text =====
        # Justified text: both margins aligned, many long lines, high avg_line_length
        is_justified_text = (
            left_alignment >= 0.8 and
            right_alignment >= 0.7 and
            long_line_ratio >= 0.5 and
            avg_line_length > 50
        )
        
        # Form: many short lines, varying length, contains colons, right not aligned
        is_true_form = (
            # Option 1: Many colons + short lines + varying length
            (colon_ratio >= 0.25 and short_line_ratio >= 0.3 and line_length_variance > 0.25)
            or
            # Option 2: Some colons + left aligned + right NOT aligned + short lines
            (colon_ratio >= 0.1 and left_alignment >= 0.65 and right_alignment < 0.65 and short_line_ratio >= 0.25)
        )
        
        # Calculate scores and text_boost_score
        text_boost_score = 0.0
        
        if is_justified_text:
            # This is justified text, not a form
            is_form = False
            score = 0.0
            # Positive boost for text detection
            text_boost_score = (
                (left_alignment + right_alignment) / 2 * 0.4 +
                long_line_ratio * 0.3 +
                min(1.0, avg_line_length / 100.0) * 0.3
            )
        elif is_true_form:
            is_form = True
            score = (
                colon_ratio * 0.4 +
                short_line_ratio * 0.3 +
                line_length_variance * 0.2 +
                left_alignment * 0.1
            )
            # Negative boost for text detection
            text_boost_score = -0.2
        else:
            is_form = False
            score = colon_ratio * 0.3 + left_alignment * 0.2
            text_boost_score = 0.0
        
        return {
            'is_form': is_form,
            'score': score,
            'text_boost_score': text_boost_score,
            'has_keyvalue_pattern': False,
            'keyvalue_ratio': keyvalue_ratio,
            'colon_ratio': colon_ratio,
            'left_alignment': left_alignment,
            'right_alignment': right_alignment,
            'avg_line_length': avg_line_length,
            'line_length_variance': line_length_variance,
            'short_line_ratio': short_line_ratio,
            'long_line_ratio': long_line_ratio,
            'is_justified_text': is_justified_text
        }
    
    def detect_figure_region(self, block: Dict) -> Dict:
        """
        Phát hiện FIGURE region (chart/plot) từ block.
        
        Heuristic based on token density:
        1. Create 20×20 grid covering block's bbox
        2. Mark cells occupied by tokens
        3. Calculate:
           - empty_center_ratio: proportion of empty cells in center area (middle of chart often empty)
           - tick_like_numbers: count of short numeric tokens (<=6 chars) resembling axis ticks
           - legend_cluster: tokens located on right side (x > 70% width), suggesting legend
        
        Figure conditions:
        - empty_center_ratio >= 0.3 AND (tick_like_numbers >= 5 OR legend_cluster=True)
        
        Score = empty_center * 0.5 + tick_density * 0.3 + legend_bonus * 0.2
        """
        lines = block.get('lines', [])
        bbox = block.get('bbox')
        
        if not bbox:
            return {
                'is_figure': False, 'score': 0.0,
                'empty_center_ratio': 0.0, 'tick_like_numbers': 0, 'legend_cluster': False
            }
        
        x_min, y_min, x_max, y_max = self.bbox_bounds(bbox)
        width = x_max - x_min
        height = y_max - y_min
        
        if width == 0 or height == 0:
            return {
                'is_figure': False, 'score': 0.0,
                'empty_center_ratio': 0.0, 'tick_like_numbers': 0, 'legend_cluster': False
            }
        
        # Create 20×20 grid
        grid_size = 20
        grid = np.zeros((grid_size, grid_size))
        
        # Mark cells occupied by tokens
        for line in lines:
            for tok in line.get('tokens', []):
                if tok.get('box'):
                    tx_min, ty_min, tx_max, ty_max = self.bbox_bounds(tok['box'])
                    
                    gx1 = int((tx_min - x_min) / width * grid_size)
                    gy1 = int((ty_min - y_min) / height * grid_size)
                    gx2 = int((tx_max - x_min) / width * grid_size)
                    gy2 = int((ty_max - y_min) / height * grid_size)
                    
                    for gx in range(max(0, gx1), min(grid_size, gx2 + 1)):
                        for gy in range(max(0, gy1), min(grid_size, gy2 + 1)):
                            grid[gy, gx] = 1
        
        # empty_center_ratio: center area (middle 50%)
        center_start = grid_size // 4  # 5
        center_end = 3 * grid_size // 4  # 15
        center_area = grid[center_start:center_end, center_start:center_end]
        empty_center_ratio = 1.0 - (np.sum(center_area) / center_area.size)
        
        # tick_like_numbers: short numeric tokens (<=6 chars)
        tick_like_numbers = 0
        for line in lines:
            for tok in line.get('tokens', []):
                text = tok.get('text', '').strip()
                if re.match(r'^-?\d+\.?\d*$', text) and len(text) <= 6:
                    tick_like_numbers += 1
        
        # legend_cluster: tokens on right side (x > 70% width)
        right_tokens = 0
        for line in lines:
            for tok in line.get('tokens', []):
                if tok.get('box'):
                    tx_min, _, _, _ = self.bbox_bounds(tok['box'])
                    if tx_min > x_min + 0.7 * width:
                        right_tokens += 1
        
        legend_cluster = right_tokens >= 3
        
        # Figure conditions
        is_figure = (empty_center_ratio >= 0.3 and (tick_like_numbers >= 5 or legend_cluster))
        
        # Score calculation
        tick_density = min(1.0, tick_like_numbers / 10.0)
        legend_bonus = 0.2 if legend_cluster else 0.0
        score = empty_center_ratio * 0.5 + tick_density * 0.3 + legend_bonus
        
        return {
            'is_figure': is_figure,
            'score': score,
            'empty_center_ratio': empty_center_ratio,
            'tick_like_numbers': tick_like_numbers,
            'legend_cluster': legend_cluster
        }
    
    def detect_text_region(self, block: Dict, form_boost: float = 0.0) -> Dict:
        """
        Phát hiện TEXT region (running paragraph) từ block.
        
        Text features:
        - Multiple lines with sufficient average length (avg_line_length)
        - Relatively even line spacing (spacing_uniformity)
        - Not many columns (few_columns) - estimated using DBSCAN on x_centers
        - Boost if 'justified text' is recognized (aligned on both margins)
        
        Also receives form_boost from detect_form:
        - If detect_form detected 'justified text', it passes positive boost to text
        - If detect_form detected true form, it passes negative boost
        
        Args:
            block: Block to analyze
            form_boost: Score adjustment from form detection
        """
        lines = block.get('lines', [])
        if len(lines) < 2:
            return {
                'is_text': False, 'score': 0.0,
                'avg_line_length': 0.0, 'spacing_uniformity': 0.0,
                'few_columns': True, 'num_cols': 0,
                'justified_boost': 0.0, 'form_boost': form_boost
            }
        
        # Average line length (in characters)
        line_lengths = [len(line.get('text', '')) for line in lines]
        avg_line_length = np.mean(line_lengths) if line_lengths else 0.0
        
        # Spacing uniformity: 1 - std/mean of y_gaps
        y_centers = [self.bbox_center(line['bbox'])[1] for line in lines if line.get('bbox')]
        if len(y_centers) >= 2:
            y_centers_sorted = sorted(y_centers)
            y_gaps = [y_centers_sorted[i+1] - y_centers_sorted[i] for i in range(len(y_centers_sorted)-1)]
            mean_gap = np.mean(y_gaps)
            spacing_uniformity = 1.0 - min(1.0, np.std(y_gaps) / max(mean_gap, 1.0))
        else:
            spacing_uniformity = 0.0
        
        # Few columns check using DBSCAN on x_centers
        x_centers = []
        all_tokens = []
        for line in lines:
            for tok in line.get('tokens', []):
                if tok.get('box'):
                    cx, _ = self.bbox_center(tok['box'])
                    x_centers.append(cx)
                    all_tokens.append(tok)
        
        num_cols = 0
        few_columns = True
        if x_centers:
            median_char_width = self.estimate_char_width(all_tokens)
            clustering_eps = max(30, 3 * median_char_width)
            
            X = np.array(x_centers).reshape(-1, 1)
            clustering = DBSCAN(eps=clustering_eps, min_samples=2).fit(X)
            num_cols = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            few_columns = (num_cols <= 2)
        
        # Check for justified text pattern (boost)
        left_x = [self.bbox_bounds(line['bbox'])[0] for line in lines if line.get('bbox')]
        right_x = [self.bbox_bounds(line['bbox'])[2] for line in lines if line.get('bbox')]
        
        left_alignment = 0.0
        if len(left_x) >= 2:
            left_mean = np.mean(left_x)
            left_std = np.std(left_x)
            left_alignment = 1.0 - min(1.0, left_std / max(left_mean, 1.0))
        
        right_alignment = 0.0
        if len(right_x) >= 2:
            right_mean = np.mean(right_x)
            right_std = np.std(right_x)
            right_alignment = 1.0 - min(1.0, right_std / max(right_mean, 1.0))
        
        long_lines = sum(1 for l in line_lengths if l > 60)
        long_line_ratio = long_lines / len(lines) if lines else 0.0
        
        # Justified text boost
        justified_boost = 0.0
        if (left_alignment >= 0.8 and right_alignment >= 0.7 and
            long_line_ratio >= 0.5 and avg_line_length > 50):
            justified_boost = 0.3
        
        # Text score calculation
        # Base score from line length and spacing
        base_score = min(1.0, avg_line_length / 100.0) * 0.5 + spacing_uniformity * 0.5
        
        # Total score with boosts
        score = base_score + justified_boost + form_boost
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        # Text conditions
        is_text = (
            (avg_line_length >= 30 and spacing_uniformity >= 0.6 and few_columns)
            or (justified_boost > 0)
            or (form_boost > 0)
        )
        
        return {
            'is_text': is_text,
            'score': score,
            'avg_line_length': avg_line_length,
            'spacing_uniformity': spacing_uniformity,
            'few_columns': few_columns,
            'num_cols': num_cols,
            'justified_boost': justified_boost,
            'form_boost': form_boost,
            'left_alignment': left_alignment,
            'right_alignment': right_alignment
        }
    
    def analyze_layout(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phân tích toàn bộ layout của document.
        
        Stage I: Neutral grouping (token → line → block)
        Stage II: Region detection for each block
        
        Each detector returns {is_xxx, score, ...metadata...}
        Scores are based on heuristics + geometric statistics from bbox/tokens.
        """
        # Stage I: Neutral grouping
        lines = self.group_tokens_to_lines(ocr_result)
        blocks = self.group_lines_to_blocks(lines)
        
        # Stage II: Region detection
        regions = []
        
        for block_idx, block in enumerate(blocks):
            # Detect all hypotheses
            table_result = self.detect_table_region(block)
            form_result = self.detect_form_region(block)
            figure_result = self.detect_figure_region(block)
            
            # Pass form_boost to text detection
            form_boost = form_result.get('text_boost_score', 0.0)
            text_result = self.detect_text_region(block, form_boost=form_boost)
            
            # Create hypotheses list
            hypotheses = [
                {'type': 'table', 'score': table_result['score'], 'metadata': table_result},
                {'type': 'form', 'score': form_result['score'], 'metadata': form_result},
                {'type': 'figure', 'score': figure_result['score'], 'metadata': figure_result},
                {'type': 'text', 'score': text_result['score'], 'metadata': text_result},
            ]
            
            # Sort by score (descending) and keep top-2
            hypotheses.sort(key=lambda h: h['score'], reverse=True)
            top_hypotheses = hypotheses[:2]
            
            # Minimum score thresholds for each type
            min_scores = {
                'table': 0.25,   # Table needs high confidence (columns, alignment)
                'form': 0.20,    # Form threshold
                'figure': 0.25,  # Figure needs empty center or ticks
                'text': 0.15     # Text is easier to detect
            }
            
            # Add qualifying regions
            for hyp in top_hypotheses:
                min_score = min_scores.get(hyp['type'], 0.2)
                if hyp['score'] >= min_score:
                    regions.append({
                        'block_id': block_idx,
                        'block': block,
                        'region_type': hyp['type'],
                        'score': hyp['score'],
                        'metadata': hyp['metadata']
                    })
        
        return {
            'lines': lines,
            'blocks': blocks,
            'regions': regions
        }

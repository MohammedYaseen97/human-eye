from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from enum import Enum
from sentence_transformers import SentenceTransformer
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class Platform(Enum):
    ANDROID = "android"
    IOS = "ios"
    DESKTOP = "desktop"

@dataclass
class UIElement:
    id: str
    element_type: str
    bounds: Dict[str, float]
    visual_properties: Dict[str, float]
    platform_specific: Dict[str, bool]

    def center_point(self) -> Tuple[float, float]:
        return (
            (self.bounds["x1"] + self.bounds["x2"]) / 2,
            (self.bounds["y1"] + self.bounds["y2"]) / 2
        )

    def size(self) -> float:
        return (self.bounds["x2"] - self.bounds["x1"]) * (self.bounds["y2"] - self.bounds["y1"])


@dataclass
class AttentionCandidate:
    position: Tuple[float, float]  # x, y coordinates (normalized 0-1)
    candidate_type: str  # 'ui_element' or 'platform_hotspot'
    element_id: Optional[str] = None  # Only for UI elements
    element: Optional[UIElement] = None  # Only for UI elements


class UIAttentionPredictor:
    def __init__(self, platform: Platform, tech_savviness: int):
        """
        Initialize the predictor with platform and user tech savviness (1-10)
        """
        self.platform = platform
        self.tech_savviness = max(1, min(10, tech_savviness))
        
        # Platform-specific priority zones with attention weights (0-1)
        self.platform_zones = {
            Platform.ANDROID: {
                "top_left": 0.9,    # System back button area
                "bottom": 0.8,       # Bottom navigation bar with main app actions
                "bottom_right": 0.7, # Floating Action Button (circular primary action)
                "top_right": 0.6,    # Overflow menu (3-dot menu) with additional options
                "center": 0.5,       # Main content area
            },
            Platform.IOS: {
                "left_edge": 0.9,    # Back gesture swipe area
                "bottom": 0.8,       # Tab bar for main navigation
                "top_right": 0.7,    # Action buttons (Share, Edit etc)
                "top": 0.6,          # Pull-down for notifications/search
                "bottom_center": 0.7, # Home indicator grab area
                "top_center": 0.6,   # Dynamic Island / Notch interactive area
                "center": 0.5        # Main content area
            },
            Platform.DESKTOP: {
                "top_left": 0.9,     # Application menu and main navigation
                "top_right": 0.8,    # User profile, settings, notifications
                "left": 0.7,         # Navigation sidebar/tree
                "center": 0.6,       # Primary content area
                "right": 0.5         # Secondary sidebar (details, properties etc)
            }
        }

        # Initialize models for text and image understanding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize BLIP-2 from transformers
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        self._task_score_cache = {}  # Simple cache for task scores


    def _calculate_visual_attraction_score(self, candidate: AttentionCandidate) -> float:
        """
        Calculate how visually attractive a point is to the eye.
        For UI elements, considers their visual properties.
        For platform hotspots (no UI element), returns 0.
        """
        if candidate.candidate_type != 'ui_element':
            return 0.0
        
        element = candidate.element
        
        # Visual property weights
        weights = {
            "size": 0.3,        # Larger elements draw more attention
            "contrast": 0.3,    # High contrast elements pop out
            "color": 0.2,      # Bright/saturated colors attract attention
            "motion": 0.1,     # Animated elements draw eye movement
            "isolation": 0.1   # Elements with more whitespace around them stand out
        }
        
        scores = {
            "size": element.visual_properties.get("size", 0.0),
            "contrast": element.visual_properties.get("contrast", 0.0),
            "color": element.visual_properties.get("color_intensity", 0.0),
            "motion": element.visual_properties.get("is_animated", 0.0),
            "isolation": element.visual_properties.get("whitespace", 0.0)
        }
        
        visual_score = sum(weights[k] * scores[k] for k in weights.keys())
        
        # Inverse tech savviness modifier for visual attraction
        # Less tech-savvy users rely more on visual properties
        tech_factor = (11 - self.tech_savviness) / 5.0  # 1-10 scale becomes 2.0-0.2
        
        return visual_score * tech_factor
    
    
    def _calculate_base_position_score(self, position: Tuple[float, float]) -> float:
        """
        Calculate base position score for any candidate based on platform conventions
        """
        x, y = position
        zones = self.platform_zones[self.platform]
        score = 0.0
        
        # Common platform patterns with their base attention weights
        if self.platform == Platform.ANDROID:
            # Back button area
            if x < 0.2 and y < 0.2:
                score = zones.get("top_left", 0.5)
            # Overflow menu area
            elif x > 0.8 and y < 0.2:
                score = zones.get("top_right", 0.5)
            # FAB area
            elif x > 0.8 and y > 0.8:
                score = zones.get("bottom_right", 0.5)
            # Bottom navigation
            elif 0.2 < x < 0.8 and y > 0.8:
                score = zones.get("bottom", 0.5)
            # Center content
            elif 0.2 < x < 0.8 and 0.2 < y < 0.8:
                score = zones.get("center", 0.5)
        
        elif self.platform == Platform.IOS:
            # Back gesture area
            if x < 0.1:
                score = zones.get("left_edge", 0.5)
            # Action buttons area
            elif x > 0.8 and y < 0.2:
                score = zones.get("top_right", 0.5)
            # Tab bar area
            elif y > 0.8:
                score = zones.get("bottom", 0.5)
            # Dynamic Island area
            elif 0.4 < x < 0.6 and y < 0.1:
                score = zones.get("top_center", 0.5)
            # Pull-down area
            elif y < 0.1:
                score = zones.get("top", 0.5)
            # Home indicator area
            elif 0.4 < x < 0.6 and y > 0.9:
                score = zones.get("bottom_center", 0.5)
            # Main content
            elif 0.1 < x < 0.9 and 0.2 < y < 0.8:
                score = zones.get("center", 0.5)
        
        elif self.platform == Platform.DESKTOP:
            # Application menu area
            if x < 0.2 and y < 0.2:
                score = zones.get("top_left", 0.5)
            # User profile/settings area
            elif x > 0.8 and y < 0.2:
                score = zones.get("top_right", 0.5)
            # Navigation sidebar
            elif x < 0.2 and 0.2 < y < 0.8:
                score = zones.get("left", 0.5)
            # Secondary sidebar
            elif x > 0.8 and 0.2 < y < 0.8:
                score = zones.get("right", 0.5)
            # Main content area
            elif 0.2 < x < 0.8 and 0.2 < y < 0.8:
                score = zones.get("center", 0.5)
        
        # Apply tech savviness modifier to position score
        tech_factor = self.tech_savviness / 5.0  # 1-10 scale becomes 0.2-2.0
        return score * tech_factor


    def _calculate_base_task_score(self, position: Tuple[float, float], task: str) -> float:
        """Calculate task-based score with caching"""
        cache_key = (position, task.lower())
        if cache_key in self._task_score_cache:
            return self._task_score_cache[cache_key]
        
        x, y = position
        task_lower = task.lower()
        score = 0.5  # Default score
        
        # Common task-position associations
        if self.platform == Platform.ANDROID:
            if "back" in task_lower or "previous" in task_lower:
                # Back action is usually top-left
                score = 0.9 if x < 0.2 and y < 0.2 else 0.1
            elif "menu" in task_lower or "options" in task_lower:
                # Menu/options are usually top-right
                score = 0.9 if x > 0.8 and y < 0.2 else 0.2
            elif "add" in task_lower or "create" in task_lower:
                # FAB actions are bottom-right
                score = 0.9 if x > 0.8 and y > 0.8 else 0.3
            elif any(nav in task_lower for nav in ["home", "search", "profile", "navigate"]):
                # Navigation tasks focus on bottom bar
                score = 0.8 if y > 0.8 else 0.3

        elif self.platform == Platform.IOS:
            if "back" in task_lower or "previous" in task_lower:
                # Back gesture area on left edge
                score = 0.9 if x < 0.1 else 0.1
            elif "share" in task_lower or "action" in task_lower:
                # Share/action buttons usually top-right
                score = 0.9 if x > 0.8 and y < 0.2 else 0.2
            elif "notification" in task_lower or "search" in task_lower:
                # Pull-down area at top
                score = 0.9 if y < 0.1 else 0.2
            elif "home" in task_lower or "switch" in task_lower:
                # Home indicator/app switching at bottom
                score = 0.9 if y > 0.9 else 0.2
            elif any(nav in task_lower for nav in ["tab", "navigate", "section"]):
                # Tab bar navigation at bottom
                score = 0.8 if y > 0.8 else 0.3
            elif "control" in task_lower or "media" in task_lower:
                # Dynamic Island interactions
                score = 0.8 if 0.4 < x < 0.6 and y < 0.1 else 0.2

        elif self.platform == Platform.DESKTOP:
            if "menu" in task_lower or "file" in task_lower:
                # Main menu in top-left
                score = 0.9 if x < 0.2 and y < 0.2 else 0.1
            elif "profile" in task_lower or "account" in task_lower or "settings" in task_lower:
                # User/settings area in top-right
                score = 0.9 if x > 0.8 and y < 0.2 else 0.2
            elif "navigation" in task_lower or "sidebar" in task_lower:
                # Main navigation sidebar on left
                score = 0.8 if x < 0.2 and 0.2 < y < 0.8 else 0.2
            elif "details" in task_lower or "properties" in task_lower:
                # Secondary sidebar on right
                score = 0.8 if x > 0.8 and 0.2 < y < 0.8 else 0.2
            elif "content" in task_lower or "main" in task_lower:
                # Main content area in center
                score = 0.7 if 0.2 < x < 0.8 and 0.2 < y < 0.8 else 0.3
        
        # Tech-savvy users have stronger position-task associations
        tech_factor = self.tech_savviness / 5.0  # 1-10 scale becomes 0.2-2.0
        self._task_score_cache[cache_key] = score * tech_factor
        return self._task_score_cache[cache_key]


    def _merge_overlapping_candidates(self, candidates: List[AttentionCandidate], proximity_threshold: float = 0.15) -> List[AttentionCandidate]:
        """
        Merge or filter candidates that are too close to each other.
        Prefer UI elements over platform hotspots when there's overlap.
        proximity_threshold: normalized distance (0-1) to consider candidates as overlapping
        """
        merged_candidates = []
        
        # Sort candidates to process UI elements first
        sorted_candidates = sorted(
            candidates,
            key=lambda c: 0 if c.candidate_type == 'ui_element' else 1
        )
        
        for candidate in sorted_candidates:
            # Check if this candidate is too close to any existing merged candidate
            is_redundant = False
            for existing in merged_candidates:
                distance = np.sqrt(
                    (candidate.position[0] - existing.position[0])**2 +
                    (candidate.position[1] - existing.position[1])**2
                )
                
                if distance < proximity_threshold:
                    # If current is UI element and existing is hotspot, replace existing
                    if (candidate.candidate_type == 'ui_element' and 
                        existing.candidate_type == 'platform_hotspot'):
                        merged_candidates.remove(existing)
                        merged_candidates.append(candidate)
                    # If both are UI elements or current is hotspot, skip current
                    is_redundant = True
                    break
            
            if not is_redundant:
                merged_candidates.append(candidate)
        
        return merged_candidates

    def _extract_ui_elements_from_image(self, ui_image: Image) -> List[AttentionCandidate]:
        """
        Extract UI elements from the image and convert them to attention candidates.
        Uses BLIP-2 and other models to detect and analyze UI elements.
        
        Args:
            ui_image: PIL Image of the UI screenshot
            
        Returns:
            List of AttentionCandidate objects representing detected UI elements
        """
        pass

    def predict_attention(self, 
                         ui_image: Image,
                         task: str,
                         top_k: int = 3) -> Dict[str, any]:
        """
        Predict attention points for a UI screenshot.
        
        Args:
            ui_image: PIL Image of the UI screenshot
            task: Description of the user's current task
            top_k: Number of attention points to return
            
        Returns:
            Dictionary containing primary focus, secondary focuses, and attention distribution
        """
        # Extract UI elements from image
        attention_candidates = self._extract_ui_elements_from_image(ui_image)
        
        # Add platform hotspots
        attention_candidates.extend(self._generate_platform_hotspots())
        
        # First, merge overlapping candidates
        merged_candidates = self._merge_overlapping_candidates(attention_candidates)
        
        attention_points = []
        
        for candidate in merged_candidates:
            # Base scores for all candidates
            position_score = self._calculate_base_position_score(candidate.position)
            task_score = self._calculate_base_task_score(candidate.position, task)
            visual_score = self._calculate_visual_attraction_score(candidate)
            
            # Weight the scores based on tech savviness
            # Tech savvy users: position & task matter more
            # Non-tech savvy users: visual attraction matters more
            if self.tech_savviness >= 7:
                final_score = (
                    position_score * 0.4 +
                    task_score * 0.4 +
                    visual_score * 0.2
                )
            elif self.tech_savviness <= 3:
                final_score = (
                    position_score * 0.2 +
                    task_score * 0.2 +
                    visual_score * 0.6
                )
            else:  # Medium tech savviness
                final_score = (
                    position_score * 0.33 +
                    task_score * 0.33 +
                    visual_score * 0.34
                )
            
            attention_points.append({
                "element_id": candidate.element_id,
                "position": candidate.position,
                "score": final_score,
                "component_scores": {
                    "position": position_score,
                    "task": task_score,
                    "visual": visual_score
                }
            })
        
        # Sort by score
        attention_points.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top_k points and normalize their scores to get confidence distribution
        top_points = attention_points[:top_k]
        scores = np.array([point["score"] for point in top_points])
        
        # Apply softmax to get confidence distribution
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        confidences = exp_scores / exp_scores.sum()
        
        # Add confidence to top points
        for point, confidence in zip(top_points, confidences):
            point["confidence"] = float(confidence)
            point["reasoning"] = self._generate_reasoning_v2(
                next(c for c in merged_candidates if c.position == point["position"]),
                point["component_scores"]["position"],
                point["component_scores"]["task"],
                point["component_scores"]["visual"]
            )
        
        return {
            "primary_focus": top_points[0] if top_points else None,
            "secondary_focuses": top_points[1:],
            "attention_distribution": top_points
        }


    def _generate_reasoning_v2(self,
                             candidate: AttentionCandidate,
                             position_score: float,
                             task_score: float,
                             visual_score: float) -> str:
        """Generate improved reasoning for the prediction"""
        reasons = []
        
        if position_score > 0.7:
            reasons.append(f"Strong platform convention for {self.platform.value}")
        if task_score > 0.7:
            reasons.append("Highly relevant position for task")
        if visual_score > 0.7:
            reasons.append("Visually prominent")
        
        if candidate.candidate_type == 'ui_element':
            # Add UI element specific reasoning
            if candidate.element.visual_properties.get("contrast", 0) > 0.7:
                reasons.append("High visual contrast")
            if candidate.element.platform_specific.get("is_platform_pattern", False):
                reasons.append("Follows platform pattern")
        
        return ", ".join(reasons) if reasons else "Based on general UI principles"


    def _generate_platform_hotspots(self) -> List[AttentionCandidate]:
        """Generate default platform hotspots based on current platform"""
        hotspots = []
        
        if self.platform == Platform.ANDROID:
            hotspots.extend([
                AttentionCandidate(position=(0.1, 0.1), candidate_type='platform_hotspot'),  # Back button
                AttentionCandidate(position=(0.9, 0.1), candidate_type='platform_hotspot'),  # Menu
                AttentionCandidate(position=(0.9, 0.9), candidate_type='platform_hotspot'),  # FAB
                AttentionCandidate(position=(0.5, 0.95), candidate_type='platform_hotspot'), # Nav bar
            ])
        elif self.platform == Platform.IOS:
            hotspots.extend([
                AttentionCandidate(position=(0.05, 0.5), candidate_type='platform_hotspot'), # Back gesture
                AttentionCandidate(position=(0.9, 0.1), candidate_type='platform_hotspot'),  # Actions
                AttentionCandidate(position=(0.5, 0.05), candidate_type='platform_hotspot'), # Pull down
                AttentionCandidate(position=(0.5, 0.95), candidate_type='platform_hotspot'), # Home indicator
                AttentionCandidate(position=(0.5, 0.05), candidate_type='platform_hotspot'), # Dynamic Island
            ])
        elif self.platform == Platform.DESKTOP:
            hotspots.extend([
                AttentionCandidate(position=(0.1, 0.1), candidate_type='platform_hotspot'),  # Menu
                AttentionCandidate(position=(0.9, 0.1), candidate_type='platform_hotspot'),  # User/settings
                AttentionCandidate(position=(0.1, 0.5), candidate_type='platform_hotspot'),  # Nav sidebar
                AttentionCandidate(position=(0.9, 0.5), candidate_type='platform_hotspot'),  # Secondary sidebar
            ])
        
        return hotspots

    def visualize_attention(self, 
                           attention_result: Dict[str, any],
                           ui_image: Image,
                           alpha: float = 0.6) -> Image:
        """
        Visualize top attention points overlaid on the UI screenshot.
        
        Args:
            attention_result: Result from predict_attention
            ui_image: PIL Image of the UI screenshot
            alpha: Transparency of the overlay (0-1)
        
        Returns:
            PIL Image with attention visualization overlaid
        """
        from PIL import Image, ImageDraw
        
        # Convert to RGBA if not already
        ui_image = ui_image.convert('RGBA')
        
        # Create a transparent overlay for the heatmap
        overlay = Image.new('RGBA', ui_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Get image dimensions for coordinate conversion
        width, height = ui_image.size
        
        # Function to convert normalized coordinates to pixel coordinates
        def norm_to_pixel(x: float, y: float) -> Tuple[int, int]:
            return (int(x * width), int(y * height))
        
        # Get the top attention points (these already have normalized confidence scores)
        points = [attention_result["primary_focus"]] + attention_result["secondary_focuses"]
        
        # Draw attention circles
        for point in points:
            x, y = point["position"]
            confidence = point["confidence"]  # Already normalized by softmax
            
            # Convert to pixel coordinates
            px, py = norm_to_pixel(x, y)
            
            # Calculate radius based on image size (e.g., 5% of width)
            radius = int(width * 0.05)
            
            # Color intensity based only on confidence
            intensity = int(255 * confidence * alpha)
            color = (255, 0, 0, intensity)  # Red with varying alpha
            
            # Single solid circle for each attention point
            draw.ellipse(
                [(px - radius, py - radius), (px + radius, py + radius)],
                fill=color,
                outline=None
            )
        
        # Blend the overlay with the original image
        result = Image.alpha_composite(ui_image, overlay)
        
        # Add small confidence labels
        draw = ImageDraw.Draw(result)
        for point in points:
            x, y = point["position"]
            px, py = norm_to_pixel(x, y)
            confidence = point["confidence"]
            
            # Draw white label with confidence percentage
            label = f'{confidence*100:.0f}%'
            draw.text(
                (px + 10, py + 10),
                label,
                fill=(255, 255, 255, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=2
            )
        
        return result


# Example usage:
"""
# Initialize predictor
predictor = UIAttentionPredictor(
    platform=Platform.ANDROID,
    tech_savviness=3
)

# Create attention candidates from both UI elements and platform hotspots
candidates = [
    # UI Elements
    AttentionCandidate(
        position=(0.15, 0.15),
        candidate_type='ui_element',
        element_id='btn_1',
        element=UIElement(
            id="btn_1",
            element_type="button",
            bounds={"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
            visual_properties={
                "size": 0.8,
                "contrast": 0.9,
                "text_emphasis": 0.7
            },
            platform_specific={
                "is_back_button": True,
                "is_menu": False,
                "is_primary_action": True,
                "is_navigation": False
            }
        )
    ),
    
    # Platform Hotspots (no UI elements)
    AttentionCandidate(
        position=(0.1, 0.1),  # Top-left corner (Android back button area)
        candidate_type='platform_hotspot'
    ),
    AttentionCandidate(
        position=(0.9, 0.95),  # Bottom-right corner (Android FAB area)
        candidate_type='platform_hotspot'
    ),
    AttentionCandidate(
        position=(0.5, 0.95),  # Bottom center (Navigation bar area)
        candidate_type='platform_hotspot'
    )
]

# Get prediction
result = predictor.predict_attention(
    ui_image=Image.open("path_to_ui_image.png"),
    task="Find the settings menu"
)

print(result)
""" 
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

class UIAttentionPredictor:
    def __init__(self, platform: Platform, tech_savviness: int):
        """
        Initialize the predictor with platform and user tech savviness (1-10)
        """
        self.platform = platform
        self.tech_savviness = max(1, min(10, tech_savviness))
        
        # Platform-specific priority zones
        self.platform_zones = {
            Platform.ANDROID: {
                "top_left": 0.9,    # Back button
                "bottom": 0.8,       # Navigation
                "bottom_right": 0.7, # FAB
                "top_right": 0.6,    # Overflow
            },
            Platform.IOS: {
                "left_edge": 0.9,    # Back gesture
                "bottom": 0.8,       # Tab bar
                "top_right": 0.7,    # Actions
                "top": 0.6,          # Pull-down
            },
            Platform.DESKTOP: {
                "top_left": 0.9,     # Menu
                "top_right": 0.8,    # User area
                "left": 0.7,         # Sidebar
                "center": 0.6,       # Main content
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

    def _calculate_tech_savviness_modifiers(self, element: UIElement) -> float:
        """
        Calculate modifiers based on tech savviness level
        """
        is_novice = self.tech_savviness <= 3
        is_expert = self.tech_savviness >= 8
        
        modifier = 1.0
        
        if is_novice:
            # Novice users prefer explicit UI elements
            if element.element_type == "button":
                modifier *= 1.7
            if element.element_type == "text":
                modifier *= 1.5
            if self._is_center_element(element):
                modifier *= 1.3
            if element.platform_specific.get("is_gesture_area", False):
                modifier *= 0.5
                
        elif is_expert:
            # Expert users prefer efficient paths
            if element.platform_specific.get("is_shortcut", False):
                modifier *= 1.7
            if element.platform_specific.get("is_gesture_area", False):
                modifier *= 1.5
            if self._is_edge_element(element):
                modifier *= 1.3
            if element.platform_specific.get("is_platform_pattern", False):
                modifier *= 1.4
                
        return modifier

    def _calculate_visual_score(self, element: UIElement) -> float:
        """
        Calculate base visual score for an element
        """
        weights = {
            "size": 0.3,
            "contrast": 0.3,
            "position": 0.2,
            "text_emphasis": 0.2
        }
        
        size_score = element.visual_properties["size"]
        contrast_score = element.visual_properties["contrast"]
        position_score = self._calculate_position_score(element)
        text_emphasis = element.visual_properties.get("text_emphasis", 0.5)
        
        return (
            weights["size"] * size_score +
            weights["contrast"] * contrast_score +
            weights["position"] * position_score +
            weights["text_emphasis"] * text_emphasis
        )

    def _calculate_position_score(self, element: UIElement) -> float:
        """
        Calculate position-based score using platform zones
        """
        center = element.center_point()
        zones = self.platform_zones[self.platform]
        
        # Simple zone checking - can be made more sophisticated
        if center[0] < 0.2 and center[1] < 0.2:  # Top left
            return zones.get("top_left", 0.5)
        elif center[0] > 0.8 and center[1] < 0.2:  # Top right
            return zones.get("top_right", 0.5)
        # Add more zone checks based on platform
        
        return 0.5  # Default score

    def _is_center_element(self, element: UIElement) -> bool:
        """Check if element is in the center region"""
        center = element.center_point()
        return 0.3 <= center[0] <= 0.7 and 0.3 <= center[1] <= 0.7

    def _is_edge_element(self, element: UIElement) -> bool:
        """Check if element is near any edge"""
        center = element.center_point()
        return (
            center[0] <= 0.1 or
            center[0] >= 0.9 or
            center[1] <= 0.1 or
            center[1] >= 0.9
        )

    def _get_element_description(self, element: UIElement) -> str:
        """
        Generate a textual description of the element combining its properties
        """
        descriptions = []
        
        # Add element type and role
        descriptions.append(f"{element.element_type}")
        if "role" in element.visual_properties:
            descriptions.append(element.visual_properties["role"])
        
        # Add any text content if available
        if "text_content" in element.visual_properties:
            descriptions.append(element.visual_properties["text_content"])
        
        # Add semantic labels if available
        if "semantic_label" in element.visual_properties:
            descriptions.append(element.visual_properties["semantic_label"])
            
        # Add image caption if available, but with lower confidence
        if "screenshot" in element.visual_properties:
            image = element.visual_properties["screenshot"]
            inputs = self.processor(
                image, 
                "Describe this UI element and its purpose", # Prompt to focus on UI context
                return_tensors="pt"
            ).to(self.device)
            
            try:
                with torch.no_grad():
                    generated_ids = self.blip_model.generate(
                        **inputs,
                        max_new_tokens=30,  # Shorter descriptions for UI elements
                        num_beams=3,
                        min_length=5,
                        length_penalty=0.8  # Prefer concise descriptions
                    )
                caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                # Only add caption if it seems relevant to UI
                if any(ui_term in caption.lower() for ui_term in [
                    "button", "icon", "menu", "text", "image", "link", "input",
                    "select", "checkbox", "radio", "toggle", "slider", "dialog"
                ]):
                    descriptions.append(caption)
            except Exception as e:
                # Fallback gracefully if image processing fails
                pass
            
        return " ".join(descriptions)

    def _calculate_task_relevance(self, element: UIElement, task: str) -> float:
        """
        Calculate task relevance using text similarity and image understanding
        """
        element_description = self._get_element_description(element)
        
        # Weight different aspects of the element
        relevance_scores = []
        
        # Direct text match is most reliable
        if "text_content" in element.visual_properties:
            text_embedding = self.text_model.encode(element.visual_properties["text_content"], convert_to_tensor=True)
            task_embedding = self.text_model.encode(task, convert_to_tensor=True)
            text_similarity = float(torch.nn.functional.cosine_similarity(task_embedding, text_embedding, dim=0))
            relevance_scores.append(text_similarity * 0.5)  # Higher weight for text content
        
        # Element type and semantic information
        base_desc = f"{element.element_type}"
        if "semantic_label" in element.visual_properties:
            base_desc += f" {element.visual_properties['semantic_label']}"
        base_embedding = self.text_model.encode(base_desc, convert_to_tensor=True)
        task_embedding = self.text_model.encode(task, convert_to_tensor=True)
        base_similarity = float(torch.nn.functional.cosine_similarity(task_embedding, base_embedding, dim=0))
        relevance_scores.append(base_similarity * 0.3)  # Medium weight for element type/semantics
        
        # Image-based caption (lowest confidence)
        if "screenshot" in element.visual_properties and len(element_description.split()) > len(base_desc.split()):
            caption_embedding = self.text_model.encode(element_description, convert_to_tensor=True)
            caption_similarity = float(torch.nn.functional.cosine_similarity(task_embedding, caption_embedding, dim=0))
            relevance_scores.append(caption_similarity * 0.2)  # Lower weight for image-based understanding
        
        # Combine scores
        if relevance_scores:
            task_relevance = sum(relevance_scores)
            return max(0.0, min(1.0, task_relevance))
        return 0.5  # Default relevance if no scores available

    def _calculate_confidence(self, 
                            element: UIElement, 
                            task: str, 
                            visual_score: float, 
                            tech_modifier: float) -> float:
        """
        Calculate confidence score for an element
        """
        task_relevance = self._calculate_task_relevance(element, task)
        
        platform_convention_match = 1.0
        if self.platform == Platform.ANDROID:
            if element.platform_specific.get("is_back_button", False):
                platform_convention_match = 0.9
        # Add more platform-specific conventions
        
        return (
            task_relevance * 0.4 +
            visual_score * 0.3 +
            platform_convention_match * 0.2 +
            tech_modifier * 0.1
        )

    def predict_attention(self, 
                         elements: List[UIElement], 
                         task: str) -> Dict[str, any]:
        """
        Predict attention points for given UI elements and task
        """
        attention_points = []
        
        for element in elements:
            visual_score = self._calculate_visual_score(element)
            tech_modifier = self._calculate_tech_savviness_modifiers(element)
            
            final_score = visual_score * tech_modifier
            confidence = self._calculate_confidence(
                element, task, visual_score, tech_modifier
            )
            
            attention_points.append({
                "element_id": element.id,
                "position": element.center_point(),
                "score": final_score,
                "confidence": confidence,
                "reasoning": self._generate_reasoning(
                    element, visual_score, tech_modifier, confidence
                )
            })
        
        # Sort by score
        attention_points.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "primary_focus": attention_points[0] if attention_points else None,
            "secondary_focuses": attention_points[1:3],
            "all_points": attention_points
        }

    def _generate_reasoning(self, 
                          element: UIElement, 
                          visual_score: float, 
                          tech_modifier: float, 
                          confidence: float) -> str:
        """Generate human-readable reasoning for the prediction"""
        reasons = []
        
        if visual_score > 0.7:
            reasons.append("High visual prominence")
        if tech_modifier > 1.3:
            reasons.append(
                f"{'Novice' if self.tech_savviness <= 3 else 'Expert'} user preference"
            )
        if element.platform_specific.get("is_platform_pattern", False):
            reasons.append("Follows platform convention")
            
        return ", ".join(reasons) if reasons else "Based on general UI principles"

# Example usage:
"""
# Initialize predictor
predictor = UIAttentionPredictor(
    platform=Platform.ANDROID,
    tech_savviness=3
)

# Your UI element detection system should provide elements in this format
elements = [
    UIElement(
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
    ),
    # ... more elements ...
]

# Get prediction
result = predictor.predict_attention(
    elements=elements,
    task="Find the settings menu"
)

print(result)
""" 
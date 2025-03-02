# UI Attention Prediction with Visual Language Models

This project implements a Visual Language Model (VLM) based approach to predict where users would first look in a UI when trying to accomplish a specific task. It takes into account:
- The user's tech savviness level
- The platform (Android/iOS/Desktop)
- The specific task description
- UI element visual characteristics

## Features

- Uses CLIP for zero-shot UI element classification
- Employs Meta's Segment Anything Model (SAM) for UI element detection
- Generates attention heatmaps
- Considers platform-specific UI conventions
- Adjusts predictions based on user tech savviness
- Provides confidence scores and reasoning for predictions

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the SAM checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

```python
from vlm_attention_model import VLMAttentionModel, Platform

# Initialize model
model = VLMAttentionModel()

# Make predictions
predictions = model.predict_attention(
    image_path="path/to/screenshot.png",
    task_description="Find the settings button to change the theme",
    platform=Platform.ANDROID,
    tech_level=3  # 1-10, where 1 is novice and 10 is expert
)

# The predictions will be sorted by confidence
for pred in predictions[:3]:
    print(f"Position: {pred.position}")
    print(f"Confidence: {pred.confidence:.2f}")
    print(f"Element Type: {pred.element.element_type}")
    print(f"Reasoning: {pred.reasoning}")
```

## Output Format

The model produces:
1. A list of `AttentionPredictionVLM` objects containing:
   - Position (x, y coordinates)
   - Confidence score
   - Element type
   - Reasoning for the prediction
2. A heatmap visualization
3. A JSON file with detailed predictions

## How It Works

1. **UI Element Detection**: Uses SAM to segment the UI into distinct elements
2. **Element Classification**: Uses CLIP to classify each element into UI component types
3. **Task Understanding**: Embeds the task description using CLIP
4. **Attention Prediction**: Combines:
   - Visual element characteristics
   - Task relevance
   - Platform-specific conventions
   - User tech savviness
5. **Confidence Calculation**: Weights multiple factors to determine final confidence scores

## Limitations

- Requires GPU for optimal performance
- SAM and CLIP models can be memory-intensive
- Currently doesn't support text recognition (OCR) - planned for future
- Predictions are heuristic-based and may need calibration for specific use cases

## Contributing

Feel free to open issues or submit pull requests for improvements! 
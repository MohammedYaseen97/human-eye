import gradio as gr
import numpy as np
from PIL import Image
from ui_attention_predictor import UIAttentionPredictor, Platform
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def process_image(image_path, age, tech_savviness, platform_str):
    # Convert platform string to enum
    platform_map = {
        "Android": Platform.ANDROID,
        "iOS": Platform.IOS,
        "Desktop": Platform.DESKTOP
    }
    platform = platform_map[platform_str]
    
    # Initialize predictor
    predictor = UIAttentionPredictor(platform=platform, tech_savviness=tech_savviness)
    
    # Load and process image
    image = Image.open(image_path)
    
    # Mock elements data for now - in real implementation this would come from your detection system
    elements_data = [
        {
            "type": "ui_element",
            "text": "",
            "bounds": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}
        }
        # Add more mock elements as needed
    ]
    
    # Get predictions
    attention_result = predictor.predict_attention(image, f"User age {age}", elements_data)
    
    # Generate visualizations for different timesteps
    timesteps = [0.2, 0.4, 0.6, 0.8, 1.0]  # Example timesteps
    visualizations = []
    
    for i, alpha in enumerate(timesteps):
        vis_image = predictor.visualize_attention(
            attention_result,
            image,
            alpha=alpha,
            top_k=int(5 * (i + 1))  # Show more points over time
        )
        visualizations.append((f"Timestep {alpha:.1f}s", vis_image))
    
    return visualizations

def create_ui():
    with gr.Blocks() as app:
        gr.Markdown("# UI Attention Prediction")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                image_input = gr.Image(type="filepath", label="Upload UI Screenshot")
                age_input = gr.Slider(minimum=1, maximum=100, value=25, step=1, label="User Age")
                tech_input = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Tech Savviness (1-10)")
                platform_input = gr.Dropdown(
                    choices=["Android", "iOS", "Desktop"],
                    value="Desktop",
                    label="Platform"
                )
                submit_btn = gr.Button("Predict Attention")
            
            with gr.Column(scale=2):
                # Output gallery with scrollbar
                output_gallery = gr.Gallery(
                    label="Attention Predictions Over Time",
                    show_label=True,
                    elem_id="gallery",
                    columns=1,
                    rows=1,
                    height=600
                )
        
        # Handle submission
        submit_btn.click(
            fn=process_image,
            inputs=[image_input, age_input, tech_input, platform_input],
            outputs=output_gallery
        )
        
        # Add CSS for gallery scrolling
        gr.HTML("""
            <style>
            #gallery {
                overflow-y: auto;
                max-height: 600px;
            }
            </style>
        """)
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True) 
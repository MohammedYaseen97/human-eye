from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from ui_attention_predictor import Platform
import matplotlib.pyplot as plt

cos = nn.CosineSimilarity(dim=0, eps=1e-8)

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess the input image for model inference
    """
    # Add your image preprocessing logic here
    # Example:
    # - Resize
    # - Normalize
    # - Convert to tensor
    pass


def postprocess_output(model_output: torch.Tensor) -> list:
    """
    Convert model output to the desired format
    """
    # Add your output processing logic here
    pass


def validate_inputs(age: int, platform: str, task: str, tech_saviness: int) -> bool:
    """
    Validate all input parameters
    """
    try:
        # Add validation logic
        assert isinstance(age, int) and 0 <= age <= 120
        assert isinstance(platform, Platform)  # platform should already be a Platform enum
        assert isinstance(tech_saviness, int) and 1 <= tech_saviness <= 10
        return True
    except:
        return False

def caption_single_image(cropped_image, model, processor, prompt=None):
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"
            
    device = model.device
    if model.device.type == 'cuda':
        inputs = processor(images=cropped_image, text=prompt, return_tensors="pt", do_resize=False).to(device=device, dtype=torch.float16)
    else:
        inputs = processor(images=cropped_image, text=prompt, return_tensors="pt").to(device=device)
    if 'florence' in model.config.name_or_path:
        generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=20,num_beams=1, do_sample=False)
    else:
        generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1)
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text.strip()


def evaluate_cropped_icon(model, processor, embed_model, cropped_icon, task_description, threshold=0.4):
    prompt = "List the UI elements in this image in less than 10 words."
    
    cropped_icon = cropped_icon.convert('RGB')
    cropped_icon.resize((64, 64), resample=Image.Resampling.LANCZOS)
    
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"
    
    device = model.device
    
    # Process single image
    if model.device.type == 'cuda':
        inputs = processor(images=cropped_icon, text=prompt, return_tensors="pt", do_resize=False).to(device=device, dtype=torch.float16)
    else:
        inputs = processor(images=cropped_icon, text=prompt, return_tensors="pt").to(device=device)
    
    if 'florence' in model.config.name_or_path:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=20,
            num_beams=1,
            do_sample=False
        )
    else:
        generated_ids = model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_return_sequences=1
        )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    embeddings = embed_model.encode([caption, task_description], convert_to_tensor=True)
    print(f"caption: {caption}, task_description: {task_description}")
    similarity = cos(embeddings[0], embeddings[1])
    print(f"similarity: {similarity.item()}")
    return similarity.item() > threshold


def draw_attention(attention_point, ui_image, alpha=0.9) -> Image:
    from PIL import Image, ImageDraw
        
    # Convert to RGBA if not already
    ui_image = ui_image.convert('RGBA')
    
    # Create a transparent overlay for the heatmap
    overlay = Image.new('RGBA', ui_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Get image dimensions for coordinate conversion
    width, height = ui_image.size
    
    # Function to convert normalized coordinates to pixel coordinates
    # Flip y-coordinate since PIL uses top-left origin
    def norm_to_pixel(x: float, y: float):
        return (int(x * width), int(y * height))  # Flip y coordinate
    
    x, y = attention_point["position"]
    
    # Convert to pixel coordinates (y is now flipped)
    px, py = norm_to_pixel(x, y)
    
    # Calculate radius based on image size (e.g., 10% of width)
    radius = int(width * 0.1)
    
    # Color intensity based only on confidence
    intensity = int(255 * alpha)
    
    # Choose color based on whether it's a hotspot or UI element
    color = (0, 255, 0, intensity) if attention_point["candidate_type"] == "platform_hotspot" else (255, 0, 0, intensity)  # Green for hotspots, Red for UI elements
    
    # Single solid circle for each attention point
    draw.ellipse(
        [(px - radius, py - radius), (px + radius, py + radius)],
        fill=color,
        outline=None
    )
    
    # Add this line to blend the overlay with the original image
    ui_image = Image.alpha_composite(ui_image, overlay)
    return ui_image


def create_spatial_grid(elements):
    # Sort elements by y-coordinate first (rows)
    sorted_by_y = sorted(elements, key=lambda x: x["position"][1])
    
    # Group elements into rows based on y-coordinate proximity
    rows = []
    current_row = []
    y_threshold = 0.05  # Adjust based on your UI layout
    
    for element in sorted_by_y:
        if not current_row or abs(element["position"][1] - current_row[0]["position"][1]) < y_threshold:
            current_row.append(element)
        else:
            current_row.sort(key=lambda x: x["position"][0])
            rows.append(current_row)
            current_row = [element]
    
    if current_row:
        current_row.sort(key=lambda x: x["position"][0])
        rows.append(current_row)
    
    return rows


def z_scan_pattern(elements_ref, last_element, debug):
    if not last_element:
        # Start from top-left
        return min(elements_ref, key=lambda x: (x["position"][1], x["position"][0]))
    
    spatial_grid = create_spatial_grid(elements_ref)
    current_row_idx = None
    current_col_idx = None
    
    # Find current position in grid
    for i, row in enumerate(spatial_grid):
        for j, element in enumerate(row):
            if element["element_id"] == last_element["element_id"]:
                current_row_idx = i
                current_col_idx = j
                if debug:
                    print(f"found last element in grid: {current_row_idx}, {current_col_idx}")
                break
        if current_row_idx is not None:
            break
    
    if current_row_idx is None:
        if debug:
            print("current_row_idx is None")
        return None
    
    # Z-pattern movement
    if current_row_idx % 2 == 0:  # Moving right
        if debug:
            print(f"length of current row: {len(spatial_grid[current_row_idx])}, moving right")
        if current_col_idx < len(spatial_grid[current_row_idx]) - 1:
            if debug:
                print(f"moving right, next element: {current_row_idx}, {current_col_idx + 1}")
            return spatial_grid[current_row_idx][current_col_idx + 1]
        elif current_row_idx < len(spatial_grid) - 1:
            # Move to next row, starting from right
            if debug:
                print(f"last element, moving down, next element: {current_row_idx + 1}, {current_col_idx - 1}")
            return spatial_grid[current_row_idx + 1][-1]
    else:  # Moving left
        if debug:
            print(f"length of current row: {len(spatial_grid[current_row_idx])}, moving left")
        if current_col_idx > 0:
            if debug:
                print(f"moving left, next element: {current_row_idx}, {current_col_idx - 1}")
            return spatial_grid[current_row_idx][current_col_idx - 1]
        elif current_row_idx < len(spatial_grid) - 1:
            # Move to next row, starting from left
            if debug:
                print(f"last element, moving down, next element: {current_row_idx + 1}, 0")
            return spatial_grid[current_row_idx + 1][0]
    
    return None


def find_current_position(last_element, spatial_grid):
    """Helper function to find element position in grid"""
    for i, row in enumerate(spatial_grid):
        for j, element in enumerate(row):
            if element["element_id"] == last_element["element_id"]:
                return i, j
    return None, None


def f_scan_pattern(elements_ref, last_element, debug):
    import random
    
    if not last_element:
        if debug:
            print("Starting from top-left element")
        return min(elements_ref, key=lambda x: (x["position"][1], x["position"][0]))
    
    spatial_grid = create_spatial_grid(elements_ref)
    current_row_idx, current_col_idx = find_current_position(last_element, spatial_grid)
    
    if debug:
        print(f"\nCurrent position: Row {current_row_idx}, Column {current_col_idx}")
    
    if current_row_idx is None:
        if debug:
            print("Element not found in grid")
        return None
        
    # If not in first column, chance to return to first columns
    if current_col_idx > 0:
        cols_in_row = len(spatial_grid[current_row_idx])
        jump_probability = 1/(cols_in_row + 3)
        if debug:
            print(f"Not in first column. Chance to return to first columns: {jump_probability:.2f}")
        
        if random.random() < jump_probability:
            target_col = random.randint(0, 1)
            if debug:
                print(f"Jumping back to column {target_col} in current row")
            if target_col < cols_in_row:
                return spatial_grid[current_row_idx][target_col]
            if debug:
                print("Jump failed - target column doesn't exist")
    
    # If in first two columns of lower rows, chance to jump to top rows
    if current_row_idx > 1 and current_col_idx < 2:
        jump_probability = 1/(len(spatial_grid) + 4)
        if debug:
            print(f"In first two columns of lower row. Chance to jump to top rows: {jump_probability:.2f}")
        
        if random.random() < jump_probability:
            target_row = random.randint(0, 1)
            if debug:
                print(f"Jumping up to row {target_row}")
            if current_col_idx < len(spatial_grid[target_row]):
                return spatial_grid[target_row][current_col_idx]
            if debug:
                print("Jump failed - target position doesn't exist")
    
    # Default sequential movement
    if debug:
        print("\nAttempting sequential movement:")
    if current_col_idx < len(spatial_grid[current_row_idx]) - 1:
        if debug:
            print(f"Moving right to column {current_col_idx + 1}")
        return spatial_grid[current_row_idx][current_col_idx + 1]
    elif current_row_idx < len(spatial_grid) - 1:
        if debug:
            print(f"Moving to next row {current_row_idx + 1}, column 0")
        return spatial_grid[current_row_idx + 1][0]
    
    if debug:
        print("No valid moves remaining")
    return None


def layered_scan_pattern(elements_ref, last_element, debug, layer_size=2):
    if not last_element:
        # Start from top-left
        return min(elements_ref, key=lambda x: (x["position"][1], x["position"][0]))
    
    spatial_grid = create_spatial_grid(elements_ref)
    current_row_idx = None
    current_col_idx = None
    
    # Find current position in grid
    for i, row in enumerate(spatial_grid):
        for j, element in enumerate(row):
            if element["element_id"] == last_element["element_id"]:
                current_row_idx = i
                current_col_idx = j
                if debug:
                    print(f"found last element in grid: {current_row_idx}, {current_col_idx}")
                break
        if current_row_idx is not None:
            break
    
    if current_row_idx is None:
        if debug:
            print("current_row_idx is None")
        return None
    
    # Determine which layer we're in
    current_layer = current_row_idx // layer_size
    layer_start = current_layer * layer_size
    layer_end = min(layer_start + layer_size, len(spatial_grid))
    
    if debug:
        print(f"\nCurrent position: Row {current_row_idx}, Col {current_col_idx}")
        print(f"Currently in Layer {current_layer} (Rows {layer_start}-{layer_end-1})")
    
    # Scan within current layer
    if current_col_idx < len(spatial_grid[current_row_idx]) - 1:
        if debug:
            print(f"Moving right within layer {current_layer}")
        return spatial_grid[current_row_idx][current_col_idx + 1]
    elif current_row_idx < layer_end - 1:
        if debug:
            print(f"Moving to start of next row within layer {current_layer}")
        return spatial_grid[current_row_idx + 1][0]
    elif layer_end < len(spatial_grid):
        if debug:
            print(f"Layer {current_layer} complete! Moving to next layer...")
        return spatial_grid[layer_end][0]
    
    if debug:
        print("Reached end of grid!")
    return None


def find_next_element_scan(elements_ref, pattern, last_element, debug=False):
    if pattern == "Spotted Pattern":
        visual_elements = sorted(elements_ref, key=lambda x: x["scores"], reverse=True)
        for i, element in enumerate(visual_elements):
            if element["element_id"] == last_element["element_id"]:
                return visual_elements[i+1]
    elif pattern == "Z-Pattern":
        return z_scan_pattern(elements_ref, last_element, debug)
    elif pattern == "F-Pattern":
        return f_scan_pattern(elements_ref, last_element, debug)
    elif pattern == "Layered Pattern":
        return layered_scan_pattern(elements_ref, last_element, debug)
    
    return None

def display_image(image: Image.Image, jup=True):
    if jup:
        image.show()
        return
    
    # Convert PIL Image to matplotlib format and display
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show(block=True)  # This will block until the window is closed
    plt.close()  # Explicitly close the figure
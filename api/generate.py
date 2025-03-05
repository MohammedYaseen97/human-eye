from http.client import BAD_REQUEST
from models.predictor import UIPredictor
from models.utils import validate_inputs
from PIL import Image
import io

# Initialize the predictor once - it will be reused across requests
predictor = UIPredictor()

def handle_request(request):
    """
    Handle incoming API requests
    """
    try:
        # Get form data
        form_data = request.form
        
        # Get and validate inputs
        age = int(form_data.get('age'))
        platform = form_data.get('platform')
        task = form_data.get('task')
        tech_saviness = int(form_data.get('tech_saviness'))
        
        # Validate inputs
        if not validate_inputs(age, platform, task, tech_saviness):
            return {"error": "Invalid inputs"}, BAD_REQUEST
            
        # Get and process image
        image_file = request.files.get('image')
        if not image_file:
            return {"error": "No image provided"}, BAD_REQUEST
            
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Get predictions
        result = predictor.predict(
            image=image,
            age=age,
            platform=platform,
            task=task,
            tech_saviness=tech_saviness
        )
        
        return result
        
    except Exception as e:
        return {"error": str(e)}, BAD_REQUEST 
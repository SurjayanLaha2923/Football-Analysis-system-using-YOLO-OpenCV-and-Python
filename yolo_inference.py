import torch
from ultralytics import YOLO
import numpy as np
import pandas as pd
import math

# Example: Handling NaN values
nan_value = np.nan
integer_value = np.nan_to_num(nan_value, nan=0)  # replace NaN with 0
print(integer_value)

nan_value = float('nan')
if math.isnan(nan_value):
    integer_value = 0  # or any other integer value
else:
    integer_value = int(nan_value)
print(integer_value)

df = pd.DataFrame({'values': [1, 2, np.nan, 4]})
df['values'] = pd.to_numeric(df['values'], errors='coerce').fillna(0)
print(df)

# Load the YOLO model
model = YOLO('yolov8x')

# Run prediction
results = model.predict('input_videos/58177_004239_Sideline.mp4', save=True)

# Preprocess the results to ensure no NaN values are present
def sanitize_boxes(results):
    for result in results:
        if hasattr(result, 'boxes'):
            for i, box in enumerate(result.boxes):
                # Convert to numpy array
                box_array = np.array(box)
                
                # Replace NaN with 0 and ensure no negative or out-of-bounds values
                box_array = np.nan_to_num(box_array, nan=0, posinf=0, neginf=0)
                
                # Clip values to ensure they are within valid ranges (e.g., non-negative)
                box_array = np.clip(box_array, a_min=0, a_max=None)
                
                # Update the box in the results with sanitized values
                result.boxes[i] = box_array

# Sanitize boxes in the results
sanitize_boxes(results)

# Print the results
for result in results:
    print(result)
    print('=======================================')
    for box in result.boxes:
        # Ensure no NaN values before conversion to integer
        box_array = np.nan_to_num(np.array(box), nan=0)
        p1, p2 = (int(box_array[0]), int(box_array[1])), (int(box_array[2]), int(box_array[3]))
        print(f"Box coordinates: p1={p1}, p2={p2}")


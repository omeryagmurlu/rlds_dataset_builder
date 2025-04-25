import os
import glob
from PIL import Image

def process_images(input_folder: str, output_folder: str, crop_params: dict):
    """
    Process and save all PNG images from input_folder into output_folder using the specified crop_params.

    crop_params should be a dictionary with keys: "left", "top", "right", "bottom" 
    where each value is the percentage (as a float between 0 and 1) of the corresponding edge.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))
    
    for img_path in image_paths:
        img = Image.open(img_path)
        width, height = img.size
        # Calculate crop coordinates based on percentages in crop_params
        left = int(width * crop_params["left"])
        top = int(height * crop_params["top"])
        right = int(width * crop_params["right"])
        bottom = int(height * crop_params["bottom"])
        
        cropped_img = img.crop((left, top, right, bottom))
        # Resize to 250x250 pixels using LANCZOS filter (for best quality)
        resized_img = cropped_img.resize((250, 250), Image.LANCZOS)
        
        # Create output filename by adding _processed suffix and saving as JPEG
        base_name = os.path.basename(img_path)
        name_without_ext, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_folder, f"{name_without_ext}_processed.jpeg")
        resized_img.save(output_path, "JPEG", quality=100, subsampling=0, optimize=True, progressive=True)
        print(f"Processed image saved to: {output_path}")

# Base directory where all collected data is stored.
base_dir = "/home/mnikolaus/code/data/collected_data"

# Iterate over each timestamp folder in base_dir
for timestamp_folder in os.listdir(base_dir):
    ts_folder_path = os.path.join(base_dir, timestamp_folder)
    if os.path.isdir(ts_folder_path):
        # Construct the path to the images folder within this timestamp folder.
        images_folder = os.path.join(ts_folder_path, "images")
        if os.path.exists(images_folder):
            # Process each camera folder that ends with "_orig"
            for folder_name in os.listdir(images_folder):
                if folder_name.endswith("cam_orig"):
                    input_folder = os.path.join(images_folder, folder_name)
                    # Derive the output folder name by replacing '_orig' with '_processed'
                    output_folder = os.path.join(images_folder, folder_name.replace("_orig", "_processed"))
                    
                    # Set crop parameters depending on camera (customize as needed)
                    # For example, for top_cam, we crop left=25%, right=90% and for side_cam, we keep full image.
                    if "top_cam" in folder_name:
                        crop_params = {"left": 0.25, "top": 0.00, "right": 0.90, "bottom": 1.0}
                    elif "side_cam" in folder_name:
                        crop_params = {"left": 0.00, "top": 0.00, "right": 1.0, "bottom": 1.0}
                    else:
                        raise ValueError("in wrong folder:", folder_name)
                    
                    print(f"Processing images in '{input_folder}' with crop {crop_params} -> saving to '{output_folder}'")
                    process_images(input_folder, output_folder, crop_params)

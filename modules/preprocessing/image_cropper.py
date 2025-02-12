import csv
import os
import pandas as pandas
import cv2

# For image tags
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

class ImageCropper():
    INDEX_FILE_COLUMNS = ["filepath", "annotation", "timestamp", "height", "width"]

    def __init__(self, output_root, base_input_dir, dirs=None, recurse=False):
        self.dirs = dirs
        self.output_root = output_root
        self.base_input_dir = base_input_dir
        self.recurse = recurse

    def find_image(self, image_name, raw_dirs):
        for raw_dir in raw_dirs:
            image_path = os.path.join(raw_dir, image_name)
            if os.path.exists(image_path):
                return image_path
        return None


    def get_image_timestamp(self, image_path):
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()

            if exif_data is None:
                return None

            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":  # Timestamp of the original image capture
                    # Format: YYYYMMDD_HHMMSS
                    return value.replace(":", "").replace(" ", "_")

        except Exception as e:
            print(f"Error extracting timestamp from {image_path}: {e}")

        return None


    def begin_cropping(self):
        # create output directory if it doesn't exist
        os.makedirs(self.output_root, exist_ok=True)

        self.crop_size = 500
        self.half_crop = self.crop_size // 2

        # Open csv file handle for index file generations
        with open(os.path.join(self.output_root, "index.csv"), "w") as index_file:
            index_file_writer = csv.writer(index_file)
            index_file_writer.writerow(self.INDEX_FILE_COLUMNS)

            for dir_path in self.dirs:
            # Iterate through each directory
                if not os.path.exists(dir_path):
                    print(f"Directory does not exist: {dir_path}")
                    return

                print(f"Processing directory: {dir_path}")

                # Process each CPCE annotation file
                if self.recurse:
                    for (current_dir, _, filenames) in os.walk(dir_path):
                        # Get the relative path based on the correct base input
                        full_current_dir = os.path.join(dir_path, current_dir)

                        relative_path = os.path.relpath(full_current_dir, self.base_input_dir)

                        # Create the corresponding output directory inside CROPPED-CORALS
                        cropped_output_dir = os.path.join(self.output_root, relative_path)
                        os.makedirs(cropped_output_dir, exist_ok=True)
                        
                        for annotation_file in filenames:
                            if not annotation_file.endswith(".cpc"):
                                continue

                            self.process_cpc_file(full_current_dir, annotation_file, relative_path, cropped_output_dir, index_file_writer)
                else:
                    # Get the relative path based on the correct base input
                    relative_path = os.path.relpath(dir_path, self.base_input_dir)

                    # Create the corresponding output directory inside CROPPED-CORALS
                    cropped_output_dir = os.path.join(self.output_root, relative_path)
                    os.makedirs(cropped_output_dir, exist_ok=True)

                    for annotation_file in os.listdir(dir_path):
                        if not annotation_file.endswith(".cpc"):
                            continue

                        self.process_cpc_file(dir_path, annotation_file, relative_path, cropped_output_dir, index_file_writer)
                    
        print("Cropping complete!")

    def process_cpc_file(self, dir_path, annotation_file, relative_path, cropped_output_dir, index_file_writer):
        annotation_path = os.path.join(dir_path, annotation_file)
        print(f"Processing: {annotation_path}")

        # Parse the annotation file
        with open(annotation_path, "r", encoding="ISO-8859-1") as file:
            lines = file.readlines()

        try:
            # Extract image name
            image_name = os.path.splitext(annotation_file)[0] + ".JPG"
            print(f"Extracted image name: {image_name}")

            # Set start_index to the 6th line (index 5)
            start_index = 5

            # Ensure the file has at least 6 lines
            if len(lines) <= start_index:
                print(f"Skipping file {annotation_file}: File has fewer than 6 lines.")
                return

            # Parse the number of annotations
            try:
                num_annotations = int(lines[start_index].strip())
            except ValueError:
                print(f"Skipping file {annotation_file} due to invalid number of annotations.")
                return

            if num_annotations <= 0:
                print(f"Skipping file {annotation_file} due to zero or negative annotations.")
                return

            # Extract points and labels
            points = []
            labels = []
            for i in range(num_annotations):
                point_line = lines[start_index + 1 + i].strip()
                label_line = lines[start_index +
                                1 + num_annotations + i].strip()

                try:
                    x, y = map(float, point_line.split(","))
                    points.append((x, y))

                    label = label_line.split(",")[1].strip('"')
                    labels.append(label)
                except (ValueError, IndexError):
                    print(f"Skipping invalid line in file {annotation_file}: {point_line} or {label_line}")

            print(f"Extracted coordinates for {annotation_file}: {points}")

        except Exception as e:
            print(f"Error parsing annotation file {annotation_file}: {e}")
            return

        # Find the corresponding image
        image_path = os.path.join(dir_path, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_name}")
            return

        # Extract timestamp
        timestamp = self.get_image_timestamp(image_path)
        if not timestamp:
            timestamp = "UNKNOWN"

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return

        # Get image dimensions
        image_height, image_width = image.shape[:2]

        # Scaling factors (adjust max_x, max_y if needed)
        max_x = 82080
        max_y = 54720
        scale_x = image_width / max_x
        scale_y = image_height / max_y

        print(f"Scaling factors: scale_x={scale_x}, scale_y={scale_y}")

        # Crop and save each annotation
        for i, (point, label) in enumerate(zip(points, labels)):
            x, y = point

            # Scale the coordinates to pixel space
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)

            # Define crop boundaries
            x_min = max(0, scaled_x - self.half_crop)
            y_min = max(0, scaled_y - self.half_crop)
            x_max = min(image_width, scaled_x + self.half_crop)
            y_max = min(image_height, scaled_y + self.half_crop)

            # Check for valid crop
            if x_min >= x_max or y_min >= y_max:
                print(f"Skipping invalid crop at ({scaled_x}, {scaled_y}) in image {image_name}. Crop is empty.")
                continue

            # Crop the image
            cropped = image[y_min:y_max, x_min:x_max]

            # Check if cropped image is empty
            if cropped.size == 0:
                print(f"Skipping empty crop for {label}_{os.path.splitext(image_name)[0]}_{i}.JPG")
                continue

            # Save cropped image inside the replicated folder structure
            cropped_output_filename = f"{label}_{os.path.splitext(image_name)[0]}_{timestamp}_{i}.JPG"
            cropped_output_path = os.path.join(
                cropped_output_dir, cropped_output_filename)
            cv2.imwrite(cropped_output_path, cropped)
            print(f"Saved: {cropped_output_path}")

            # Add image information to the index
            index_file_writer.writerow([os.path.join(relative_path, cropped_output_filename), label, timestamp, cropped.shape[0], cropped.shape[1]])



if __name__ == "__main__":
    dirs = [
        r"D:\ORIGINAL (aka MONITORING)\2024 (COMPLETE)\QUADRAT\IMAGE AND CPCE FILE\SHINE-1790_Min Ping Yu, Tubbataha, Cagayancillo\Q1",
        r"D:\ORIGINAL (aka MONITORING)\2024 (COMPLETE)\QUADRAT\IMAGE AND CPCE FILE\SHINE-1790_Min Ping Yu, Tubbataha, Cagayancillo\Q2",
        r"D:\ORIGINAL (aka MONITORING)\2024 (COMPLETE)\QUADRAT\IMAGE AND CPCE FILE\SHINE-1790_Min Ping Yu, Tubbataha, Cagayancillo\Q3",
        r"D:\ORIGINAL (aka MONITORING)\2024 (COMPLETE)\QUADRAT\IMAGE AND CPCE FILE\SHINE-1801_USS Guardian, Tubbataha, Cagayancillo\Q1",
        r"D:\ORIGINAL (aka MONITORING)\2024 (COMPLETE)\QUADRAT\IMAGE AND CPCE FILE\SHINE-1801_USS Guardian, Tubbataha, Cagayancillo\Q2",
        r"D:\ORIGINAL (aka MONITORING)\2024 (COMPLETE)\QUADRAT\IMAGE AND CPCE FILE\SHINE-1801_USS Guardian, Tubbataha, Cagayancillo\Q3",
    ]

    output_root = r"D:\CROPPED-CORALS"
    base_input_dir = r"D:\ORIGINAL (aka MONITORING)"

    ImageCropper(output_root, base_input_dir, dirs).begin_cropping()
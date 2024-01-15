import cv2
import os
from ultralytics import YOLO

# Function to perform YOLO predictions on an image
def yolo_predict(image, model):
    conf_thresh = 0.25
    hide_conf = True
    results = model.predict(image, stream=True, save=True, save_txt=True, project=" 123", name="yolo_test", exist_ok=True)
    predicted_rectangles = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            predicted_rectangles.append((r[0], r[1], r[2], r[3], 0))  # Assuming class label is 1

    return predicted_rectangles

# Function to draw rectangles on the image
def draw_rectangles(image, rectangles, selected_rectangle):
    img_copy = image.copy()
    for i, rect in enumerate(rectangles):
        if i == selected_rectangle:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255),
                          2)  # Highlight the selected rectangle in red
        else:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

        # Display the label on the rectangle
        label = rectangles[i][-1]
        cv2.putText(img_copy, f"Class {label}", (rect[0], rect[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img_copy

# Function to find the index of the rectangle that contains a point
def find_selected_rectangle(x, y, rectangles):
    for i, rect in enumerate(rectangles):
        if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
            return i
    return -1

# Callback function for mouse events
def draw_rectangle(event, x, y, flags, param):
    global drawing, top_left_pt, rectangles, image, selected_rectangle, current_class

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        rectangles.append((top_left_pt[0], top_left_pt[1], bottom_right_pt[0], bottom_right_pt[1], current_class))
        selected_rectangle = len(rectangles) - 1  # Select the last drawn rectangle by default

# Function to delete the selected annotation
def delete_selected_annotation(rectangles, selected_rectangle):
    if selected_rectangle != -1 and selected_rectangle < len(rectangles):
        rectangles.pop(selected_rectangle)
        # After deletion, select the previous rectangle
        selected_rectangle = max(selected_rectangle - 1, 0)
    return selected_rectangle

# Function to save the annotated image and its YOLO format annotation
def save_annotation(image, rectangles, image_path, save_dir):
    img_with_rectangles = draw_rectangles(image, rectangles, -1)  # -1 indicates no selected rectangle
    cv2.imwrite(os.path.join(save_dir, f"annotated_{os.path.basename(image_path)}"), img_with_rectangles)

    yolo_annotation_file = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    with open(yolo_annotation_file, 'w') as file:
        for rect in rectangles:
            # YOLO format: class x_center y_center width height
            x_center = (rect[0] + rect[2]) / (2 * image.shape[1])
            y_center = (rect[1] + rect[3]) / (2 * image.shape[0])
            width = (rect[2] - rect[0]) / image.shape[1]
            height = (rect[3] - rect[1]) / image.shape[0]
            file.write(f"{rect[4]} {x_center} {y_center} {width} {height}\n")

# Path to the directory containing images
image_directory = 'path_to_directory'  # Replace with your image directory path
output_directory = 'path_to_output_directory'  # Replace with the directory where you want to save annotated images and YOLO annotations

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Load YOLO model
model = YOLO('/Users/sambhavdixit/PycharmProjects/ 123/best_1.pt')

image_files = os.listdir(image_directory)
image_files.sort()  # Ensure images are sorted for consistent order

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)

    # Load an image
    image = cv2.imread(image_path)

    # Perform YOLO prediction
    rectangles = yolo_predict(image, model)

    drawing = False
    top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
    selected_rectangle = 0  # Set to 0 to highlight the first predicted annotation
    current_class = 0  # Default class label

    # Create a window and set the callback function for mouse events
    cv2.namedWindow('Image Annotation')
    cv2.setMouseCallback('Image Annotation', draw_rectangle)

    while True:
        # Display the image with drawn rectangles
        img_with_rectangles = draw_rectangles(image, rectangles, selected_rectangle)
        cv2.imshow('Image Annotation', img_with_rectangles)

        # Press '1' or '2' to set the label for the selected annotation
        # Press 's' to save and move to the next image
        # Press 'd' to delete the selected annotation
        # Press 'p' to select the previous annotation
        key = cv2.waitKey(1)
        if key == ord('0') or key == ord('1'):
            if selected_rectangle != -1:
                current_class = int(chr(key))
                # Update the label in the rectangles list
                rectangles[selected_rectangle] = (
                    rectangles[selected_rectangle][0],
                    rectangles[selected_rectangle][1],
                    rectangles[selected_rectangle][2],
                    rectangles[selected_rectangle][3],
                    current_class
                )
        elif key == ord('s'):
            save_annotation(image, rectangles, image_path, output_directory)
            print(f"Annotation saved for {os.path.basename(image_path)}")
            break
        elif key == ord('d'):
            selected_rectangle = delete_selected_annotation(rectangles, selected_rectangle)
        elif key == ord('p') and len(rectangles) > 1:
            selected_rectangle = (selected_rectangle - 1) % len(rectangles)
        # Press 'Esc' to exit the annotation process without saving
        elif key == 27:
            break

    cv2.destroyAllWindows()

# Inform the user that the annotation process is complete
print("Annotation process complete.")


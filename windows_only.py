 import cv2
import os
from ultralytics import YOLO
from PIL import Image, ImageTk  # Import Image and ImageTk from Pillow
import numpy as np
import easygui
import tkinter as tk
from tkinter import filedialog ,messagebox

do_up =False

def do_upload():
    global do_up
    result = messagebox.askyesno("Upload Images", "Do you want to upload images?")

    if result ==True :
        do_up = True
        print("True")

def open_file_dialog():
    if os.name == 'posix' and 'Darwin' in os.uname():
        file_paths = easygui.fileopenbox(title="Select Image(s)", filetypes=["*.jpg", "*.jpeg", "*.png"], multiple=True)
    else:
        root = tk.Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames(title="Select Image(s)", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        root.destroy()  # Close the Tkinter root window
    return file_paths

def yolo_predict(image, model):
    conf_thresh = 0.25
    hide_conf = True

    results = model.predict(image, stream=True, save=True, save_txt=True, project="123", name="yolo_test",
                            exist_ok=True)

    predicted_rectangles = []
    labels = []

    for result in results:
        boxes = result.boxes.cpu().numpy()

        for box in boxes:
            # Extract predictions as floats
            x1, y1, x2, y2 = box.xyxy[0]
            label = box.cls

            # Convert to ints
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = int(label)

            predicted_rectangles.append((x1, y1, x2, y2, label))
            labels.append(label)

    return predicted_rectangles, labels


def draw_rectangles(image, rectangles, selected_rectangle):
    img_copy = image.copy()

    for i, rect in enumerate(rectangles):
        if i == selected_rectangle:
            # Convert to ints
            x1, y1, x2, y2, label = map(int, rect)

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            x1, y1, x2, y2, label = map(int, rect)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(img_copy, f"Class {label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img_copy

def find_selected_rectangle(x, y, rectangles):
    for i, rect in enumerate(rectangles):
        if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
            return i

    return -1

def draw_rectangle(event, x, y, flags, param):
    global drawing, top_left_pt, rectangles, image, selected_rectangle, current_class

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        rectangles.append((top_left_pt[0], top_left_pt[1], bottom_right_pt[0], bottom_right_pt[1], current_class))
        selected_rectangle = len(rectangles) - 1

def delete_selected_annotation(rectangles, selected_rectangle):
    if selected_rectangle != -1 and selected_rectangle < len(rectangles):
        rectangles.pop(selected_rectangle)
        selected_rectangle = max(selected_rectangle - 1, 0)

    return selected_rectangle

def save_annotation(image, rectangles, image_path, save_dir):
    img_with_rectangles = draw_rectangles(image, rectangles, -1)
    cv2.imwrite(os.path.join(save_dir, f"annotated_{os.path.basename(image_path)}"), img_with_rectangles)

    yolo_annotation_file = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")

    with open(yolo_annotation_file, 'w') as file:
        for rect in rectangles:
            # Convert to ints
            x1, y1, x2, y2, label = map(int, rect)

            x_center = (x1 + x2) / (2 * image.shape[1])
            y_center = (y1 + y2) / (2 * image.shape[0])
            width = (x2 - x1) / image.shape[1]
            height = (y2 - y1) / image.shape[0]

            file.write(f"{label} {x_center} {y_center} {width} {height}\n")

do_upload()

# image_paths = open_file_dialog()

if do_up:
    image_paths = open_file_dialog()

    if not image_paths:
        print("No images selected. Exiting.")
    else:
        # Continue with the rest of the code for annotation
        output_directory = 'path_to_output_directory'
        os.makedirs(output_directory, exist_ok=True)

        model = YOLO('best_1.pt')

        for image_path in image_paths:
            # Use Pillow to open the image
            pillow_image = Image.open(image_path)
            # Convert the Pillow image to OpenCV format
            image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)

            rectangles, labels = yolo_predict(image, model)

            current_class = labels[0]

            drawing = False
            top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
            selected_rectangle = 0

            cv2.namedWindow('Image Annotation')
            cv2.setMouseCallback('Image Annotation', draw_rectangle)

            while True:
                img_with_rectangles = draw_rectangles(image, rectangles, selected_rectangle)

                cv2.imshow('Image Annotation', img_with_rectangles)

                key = cv2.waitKey(1)

                if key == ord('0') or key == ord('1'):
                    if selected_rectangle != -1:
                        # Update current_class based on the key pressed
                        current_class = int(chr(key))

                        # Convert the tuple to a list, update the label, and then convert back to a tuple
                        rect_list = list(rectangles[selected_rectangle])
                        rect_list[-1] = current_class
                        rectangles[selected_rectangle] = tuple(rect_list)

                elif key == ord('s'):
                    save_annotation(image, rectangles, image_path, output_directory)
                    print(f"Annotation saved for {os.path.basename(image_path)}")
                    break

                elif key == ord('d'):
                    selected_rectangle = delete_selected_annotation(rectangles, selected_rectangle)

                elif key == ord('p') and len(rectangles) > 1:
                    selected_rectangle = (selected_rectangle - 1) % len(rectangles)

                elif key == 27:
                    break

            cv2.destroyAllWindows()

        print("Annotation process complete.")


# if not image_paths:
#     print("No images selected. Exiting.")
#     # return
#
# output_directory = 'path_to_output_directory'
# os.makedirs(output_directory, exist_ok=True)
#
# model = YOLO('best_1.pt')
#
# for image_path in image_paths:
#     # Use Pillow to open the image
#     pillow_image = Image.open(image_path)
#     # Convert the Pillow image to OpenCV format
#     image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
#
#     rectangles, labels = yolo_predict(image, model)
#
#     current_class = labels[0]
#
#     drawing = False
#     top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
#     selected_rectangle = 0
#
#     cv2.namedWindow('Image Annotation')
#     cv2.setMouseCallback('Image Annotation', draw_rectangle)
#
#     while True:
#         img_with_rectangles = draw_rectangles(image, rectangles, selected_rectangle)
#
#         cv2.imshow('Image Annotation', img_with_rectangles)
#
#         key = cv2.waitKey(1)
#
#         if key == ord('0') or key == ord('1'):
#             if selected_rectangle != -1:
#                 # Update current_class based on the key pressed
#                 current_class = int(chr(key))
#
#                 # Convert the tuple to a list, update the label, and then convert back to a tuple
#                 rect_list = list(rectangles[selected_rectangle])
#                 rect_list[-1] = current_class
#                 rectangles[selected_rectangle] = tuple(rect_list)
#
#         elif key == ord('s'):
#             save_annotation(image, rectangles, image_path, output_directory)
#             print(f"Annotation saved for {os.path.basename(image_path)}")
#             break
#
#         elif key == ord('d'):
#             selected_rectangle = delete_selected_annotation(rectangles, selected_rectangle)
#
#         elif key == ord('p') and len(rectangles) > 1:
#             selected_rectangle = (selected_rectangle - 1) % len(rectangles)
#
#         elif key == 27:
#             break
#
#     cv2.destroyAllWindows()
#
# print("Annotation process complete.")
# # if __name__ == "__main__":
# #     main()
# #     # tk.mainloop()  # Run the Tkinter main loop after the main function

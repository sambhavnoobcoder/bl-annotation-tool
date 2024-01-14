from ultralytics import YOLO
import cv2
model = YOLO('/Users/sambhavdixit/PycharmProjects/ 123/best_1.pt')
conf_thresh = 0.25
hide_conf = True
img = cv2.imread('/Users/sambhavdixit/PycharmProjects/ 123/path_to_directory/test.jpg')
results = model.predict(img, stream=True, save=True,save_txt = True ,project=" 123" ,name="yolo_test" ,exist_ok=True)
for result in results:
    boxes = result.boxes.cpu().numpy() # get boxes on cpu in numpy
    for box in boxes: # iterate boxes
        r = box.xyxy[0].astype(int) # get corner points as int
        print(r) # print boxes
        cv2.rectangle(img, r[:2], r[2:], (0,255,0), 2) # draw boxes on img

cv2.imshow('Img with boxes ',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

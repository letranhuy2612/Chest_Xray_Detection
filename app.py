import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1


def obj_detection(my_img):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    column1, column2 = st.columns(2)

    column1.subheader("Input image")
    st.text("")
    plt.figure(figsize = (16,16))
    plt.imshow(my_img,cmap='gray')
    column1.pyplot(use_column_width=True)

    net = cv2.dnn.readNet("D:\\Workspace\\VINAI_Chest_Xray\\best.onnx")
    # Runs the forward pass to get output of the output layers
    classes = ["Aortic enlargement","Atelectasis","Calcification","Cardiomegaly","Consolidation","ILD","Infiltration","Lung Opacity","Nodule/Mass","Other lesion","Pleural effusion","Pleural thickening","Pneumothorax","Pulmonary fibrosis"]
    
    output_layers = net.getUnconnectedOutLayersNames()

    colors = np.random.uniform(0,255,size=(len(classes), 3))#Các giá trị #RGB được chọn ngẫu nhiên từ 0 đến 255    

    # Tải ảnh
    newImage = np.array(my_img.convert('RGB'))
    img = cv2.cvtColor(newImage,1)
    height,width,channels = img.shape

    # Objects detection 
    #Chuyển đổi hình ảnh thành các đốm màu
    blob = cv2.dnn.blobFromImage(img, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], True, crop=False)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

	# Rows.
    rows = outputs[0].shape[1]

    image_height, image_width = img.shape[:2]

	# Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

	# Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
		# Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

			# Get the index of max class score.
            class_id = np.argmax(classes_scores)

			#  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    items = []
    for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            color = colors[i-1]
            cv2.rectangle(img, (left, top), (left + width, top + height), color, 3*THICKNESS)
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
            text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
            dim, baseline = text_size[0], text_size[1]
            # Use text size to create a BLACK rectangle. 
            cv2.rectangle(img, (left, top), (left + dim[0], top + dim[1] + baseline), color)
            # Display text inside the rectangle.
            cv2.putText(img, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, color, THICKNESS,cv2.LINE_AA)
            items.append(label)
    st.text("")
    column2.subheader("Output image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    column2.pyplot(use_column_width=True)

    if len(indices)>1:
        st.success("Found {} Objects - {}".format(len(indices),[item for item in set(items)]))
    else:
        st.success("Found {} Object - {}".format(len(indices),[item for item in set(items)]))

def main():
    
    st.title("Streamlit app")
    st.write("You can view real-time object detection done using YOLO model here. Select one of the following options to proceed:")

    choice = st.radio("", ("See an illustration", "Choose an image of your choice"))
    #st.write()

    if choice == "Choose an image of your choice":
        #st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Upload", type=['jpg','png','jpeg'])

        if image_file is not None:
            my_img = Image.open(image_file)  
            obj_detection(my_img)

    elif choice == "See an illustration":
        my_img = Image.open("D:\\Workspace\\VINAI_Chest_Xray\\yolo_data\\images\\train\\0a2d01ecb9e01cf972c1e1d31ccacb98.jpg")
        obj_detection(my_img)

if __name__ == '__main__':
    main()
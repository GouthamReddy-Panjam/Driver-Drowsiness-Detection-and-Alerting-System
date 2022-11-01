# Driver-Drowsiness-Detection-and-Alerting-System
 
 ## **Abstract** ##
 
Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving.

The objective of this project is to build a drowsiness detection system that will detect that a person’s eyes are closed for a few seconds. This system will alert the driver when drowsiness is detected.

How the Drowsiness will detect

we will be using OpenCV for gathering the images from webcam and feed them into a deep learning model which will classify whether the person’s eyes are ‘Open’ or ‘Closed’.

Step 1 – Take image as input from a camera.

Step 2 – Detect the face in the image and create a Region of Interest (ROI).

Step 3 – Detect the eyes from ROI and feed it to the classifier.

Step 4 – Classifier will categorize whether eyes are open or closed.

Step 5 – Calculate score to check whether the person is drowsy

## **Introduction** ##

Accidents due to drivers falling asleep has become very common this could be due increased work, stress, etc. There are also a number of people that drive on the highway at night and can experience fatigue which can result in them falling asleep while driving and can lead to major accidents.According to research at least 20% to 30% of road accidents are due to fatigued drivers who fall asleep while on the road. This proves to be a threat for other vehicles on the road as well who
aren’t aware of the uncertainty of another vehicle going out of control. When the driver is tired and hasn’t slept in a while it can causes severe side effects like delay in reaction time, low concentration and their alertness when it comes to noticing the activities on the road. This minimal attention to the road can affect their decision-making time and also could lead to loss of speed control.

## **Hardware / Software Requirements**  ##

The requirement for the project is a webcam through which we will capture images. You need to have Python (3.6 version recommended) installed on your system. The following packages are required for the project:

* OpenCV –
OpenCV, a library of programming functions is used for detecting the face and facial features or in for this program, the eyes of the driver.

* TensorFlow –
TensorFlow is used as a backend and works with Keras.

* Keras –
Keras is a classification model that we will use to classify the eyes of the driver as either ‘open’ or ‘closed’.

* Pygame –
Pygame is used to sound the alarm as soon as it is detected that the driver has fallen
asleep.

## **Existing System** ##

A system has been developed that uses image processing technology to analyze images of the driver's face taken with a video camera. Diminished alertness is detected on the basis of the degree to which the driver's eyes are open or closed. This detection system provides a noncontact technique for judging various levels of driver alertness and facilitates early detection of a decline in alertness during driving. An efficient method to solve these problems for eye state identification for fatigue detection, in embedded system, which is based on image processing techniques, was also proposed. This method goes against the traditional way of driver alertness detection to make it real time, it utilizes face detection and eye detection to initialize the location of driver’s eyes; after that an object tracking method is used to keep track of the eyes.

## **Drawback / Limitations of Existing System** ##

The limitations of the existing system are, because the level of drowsiness is measured approximately every 5 min, sudden variations cannot be detected using subjective measures. Another limitation to using subjective ratings is that the self-introspection alerts the driver, thereby reducing their drowsiness level. There was no alarm to wake up the driver, which is best choice to wake up the drowsy and fatigued drivers fast.

## **Proposed/Developed model** ##

### Design ###

The model we used is built with Keras using **Convolutional Neural Networks (CNN)**. A convolutional neural network is a special type of deep neural network which performs extremely well for image classification purposes. A CNN basically consists of an input layer, an output layer and a hidden layer which can have multiple numbers of layers. A convolution operation is performed on these layers using a filter that performs 2D matrix multiplication on the layer and filter.

The CNN model architecture consists of the following layers:

Convolutional layer; 32 nodes, kernel size 3
* Convolutional layer; 32 nodes, kernel size 3
* Convolutional layer; 64 nodes, kernel size 3
* Fully connected layer; 128 nodes

## **Module Wise Description** ##

### Step 1 – Take Image as Input from a Camera ###

With a webcam, we will take images as input. So to access the webcam, we made an infinite loop that will capture each frame. We use the method provided by OpenCV,
cv2.VideoCapture(0) to access the camera and set the capture object (cap). cap.read() will read each frame and we store the image in a frame variable.

### Step 2 – Detect Face in the Image and Create a Region of Interest (ROI) ###

To detect the face in the image, we need to first convert the image into grayscale as the OpenCV algorithm for object detection takes gray images in the input. We don’t need colour information to detect the objects. We will be using haar cascade classifier to detect faces. This line is used to set our classifier face = cv2.CascadeClassifier(‘ path to our haar cascade xml file’). Then we perform the detection using faces = face. detect Multi Scale(gray). It returns an array of
detections with x,y coordinates, and height, the width of the boundary box of the object. Now we  can iterate over the faces and draw boundary boxes for each face.

for (x,y,w,h) in faces:

cv2.rectangle(frame, (x,y), (x+w, y+h), (100,100,100), 1 )

### Step 3 – Detect the eyes from ROI and feed it to the classifier ###

The same procedure to detect faces is used to detect eyes. First, we set the cascade classifier for eyes in leye and reye respectively then detect the eyes using left_eye = l eye. Detect Multi Scale(gray). Now we need to extract only the eyes data from the full image. This can be achieved by extracting the boundary box of the eye and then we can pull out the eye image from the frame with this code.

L eye = frame[ y : y+h, x : x+w ]

l_ eye only contains the image data of the eye. This will be fed into our CNN classifier which will predict if eyes are open or closed. Similarly, we will be extracting the right eye into r _eye.

### Step 4 – Classifier will Categorize whether Eyes are Open or Closed ###

We are using CNN classifier for predicting the eye status. To feed our image into the model, we need to perform certain operations because the model needs the correct dimensions to start with. First, we convert the colour image into grayscale using r_ eye = cv2.cvtColor(r_ eye,cv2.COLOR_BGR2GRAY). Then, we resize the image to 24*24 pixels as our model was trained on 24*24 pixel images cv2.resize(r_ eye, (24,24)). We normalize our data for better convergence r_ eye = r_ eye/255 (All values will be between 0-1). Expand the dimensions to feed into our classifier. We loaded our model using model = load_model(‘models/cnnCat2.h5’). Now we predict each eye with our model.

l pred = model. predict_ classes (l_ eye). If the value of l pred[0] = 1, it states that eyes are open, if value of l pred[0] = 0 then, it states that eyes are closed.

### Step 5 – Calculate Score to Check whether Person is Drowsy ###

The score is basically a value we will use to determine how long the person has closed his eyes. So if both eyes are closed, we will keep on increasing score and when eyes are open, we decrease the score. We are drawing the result on the screen using cv2.putText() function which will display real time status of the person.

cv2.putText(frame, “Open”, (10, height-20), font, 1, (255,255,255), 1, cv2.LINE_AA )

A threshold is defined for example if score becomes greater than 15 that means the person’s eyes are closed for a long period of time. This is when we beep the alarm using sound.play().

## **Results** ##
![image](https://user-images.githubusercontent.com/88433888/199168559-610d43eb-53e7-4e72-bf26-f51e50f87637.png)

![image](https://user-images.githubusercontent.com/88433888/199168644-1424a620-2119-413d-b1d5-fd7ffbc1e1de.png)

![image](https://user-images.githubusercontent.com/88433888/199168674-296ad479-d56b-433a-9c5c-eab978badf0a.png)

A model has been implemented using live image input to detect facial features. Then the eyes were extracted from the face as our region of interest. This was used to determine whether a drivers eyes were open or closed. If they were closed for more than 15 seconds an alarm was rung. In our discussions, we found that this is a project with high a impact on various industries of the world. It can be used not only in the vehicle industry but also for other purposes like, being used in mobile cameras as a detector for closed eyes.

## **Conclusion** ##

To conclude, we were able to successfully implement a model which will play a sound to wake the driver up in case he/she ends up being drowsy and close their eyes during driving. We were able to implement it using several libraries like Keras, tensorflow, etc. and use CNN for real-time image capture for face detection. We are of the opinion that this technology will be able to reduce the accidents caused by drowsiness of the driver, significantly.

## **References** ##

[1] https://sci-hub.si/10.1109/itsc.2002.1041208

[2] https://www.ijedr.org/papers/IJEDR1303017.pdf

[3]https://www.safetylit.org/citations/index.php?fuseaction=citations.viewdetails&citationIds[]=citjournalarticle_245681_38

[4] https://sci-hub.si/10.1109/VNIS.1994.396873

[5] https://sci-hub.si/10.1109/ICSIPA.2011.6144162

[6] https://journals.sagepub.com/doi/10.1243/0954407011528536

[7] https://www.ias.ac.in/article/fulltext/sadh/042/11/1835-1849

[8] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4970150/

[9] https://ieeexplore.ieee.org/abstract/document/739878

[10] https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1365-2869.1995.tb00220.x









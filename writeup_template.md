# **Behavioral Cloning** 

### Project Description

In this project, I built a Convolutional Neural Network (CNN) based on NVIDIA's [paper](https://arxiv.org/pdf/1604.07316v1.pdf) to clone car driving behavior. It predicts the steering angle by the image taken from the three cameras on the car (center, left and right) to provide different angles. The images are taken from the [simulator](https://github.com/udacity/self-driving-car-sim) and are augmented to account for the imbalanced data and various real world conditions. 
 
 [Lake track](https://www.youtube.com/watch?v=927dKS0QAiA)
 
 [Jungle track](https://youtu.be/ke_9bVAiG50)
 
### Files and how to run

My project includes the following files:
* utils.py includes the methods to preprocess and augment the image data
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 

### Requirements

1. Anaconda Python 3.5+
2. Udacity [Carnd-term1 starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) GPU environment 
3. Nvidia CUDA 8.0
4. Nvidia CuDNN 6
5. Tensorflow-gpu 1.4.1 (overrides Udacity's setting)
```python
# Use TensorFlow without GPU
conda env create -f environments.yml

# Use TensorFlow with GPU
conda env create -f environments-gpu.yml

# If using tensorflow 1.4.0 or up, please download CUDA 8.0 and cuDNN 6 and follow NVIDIA's instruction.
```

Or you can manually install the required libraries (see the contents of the environemnt*.yml files) using pip.


### Quick Start

1. Control of the car is by using a PS4 controller.

2. Two driving modes:
- Training: For user to take control over the car
- Autonomous: For car to drive by itself

3. Collecting data:
I drove on track 1 and track 2, and collected data by recording the driving experience using the built-in recorder. Data was saved as frame images and a driving log which shows the location of the images, steering angle, throttle, speed, etc. 

#### How to simulate

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene and press the Autonomous Mode button.  Then, run the model as follows:
`python drive.py model.h5 track_1`

#### To train the model

```python
python model.py -n 100 -s 20000 -b 256 -o True -l 0.0001
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.

### Appropriate training data

I used a PS4 controller to drove the car on both track 1 and track 2. Track 2 is much more difficult than track 1 as it has more curvy and up-down roads. I tried to keep the car on the surface and in the center. 
Data collection is very important. Too much bad data can severly worsen the result...

### Understanding Data

There are 3 cameras on the car which shows left, center and right images for each steering angle. 

#### Unbalanced data

The left/right skew is due to driving the car around the track in one direction only and can be eliminated by flipping each recorded image and its corresponding steering angle. 

Another problem is that the steering angle is very close to zero most of the time under normal conditions. One possible solution would be to sample more data of the car drifting to the edge of the road and recovering before a crash occurs. I tried this, but it turned out performing very badly and the car would easily drive off the road. This is due to overfitting as this case is rare. Inspired by this [post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9), I mimic the car wandering off and recovery. I added a small angle 0.25 to the left camera and subtract a small angle of 0.25 from the right camera. 

### Data augmentation

1. Brightness augmentation
Changing brightness to simulate day and night conditions. I generated images with different brightness by first converting images to HSV, randomly scaling up or down the V channel and converting back to the RGB channel.

2. Using left and right camera images
As already mentioned, I added 0.25 to the left camera and subtract 0.25 from the right camera to simulate car drifting to the edge and recovering from wandering off the surface. The main idea being the left camera has to move right to get to center, and right camera has to move left. This is simulating the **recovery event**.

3. Horizontal and vertical shifts
I shifted the camera images horizontally to simulate the effect of car being at different positions on the road, and add an offset corresponding to the shift to the steering angle. I added 0.004 steering angle units per pixel shift to the right, and subtracted 0.004 steering angle units per pixel shift to the left. I also shifted the images vertically by a random number to simulate the effect of driving up or down the slope.

4. Converting to YUV color space
After performing step 1-3, each image is converted to YUV color space.
 
### Preprocessing

1. I cropped the sky and car deck parts out of the image to avoid unnecessary noise. 
2. Images are then resized to 66x200 per Nvidia's paper. 

### Generators
The model is trained using Keras with Tensorflow backend. Since there are lots of images which take disk and memory spaces, I use Keras generator to generate image on the fly instead of preprocessing images beforehead at once. After the images being shuffled and going through data augmentation and preprocessing, the generator spits out images by a user-defined batch size.

### Model Architecture and Training Strategy

My model is built based on [the NVIDIA model](https://arxiv.org/pdf/1604.07316v1.pdf) which has been adopted in self driving car test. The architecture has been proven so that it is well suited for this project.
I used ELU activation function for every layer except for the output layer to introduce non-linearity. And the data is normalized in the model using a Keras lambda layer.

The model architecture is the following:
- YUV image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Fully connected: neurons: 1164, activation: ELU
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1 (output)

Below is the Keras's illustration of the entire architecture.
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          1342092     flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]
====================================================================================================
Total params: 1,595,511
Trainable params: 1,595,511
Non-trainable params: 0
__________________________________
```

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The initial learning rate is chosen to be 0.0001 which gives reasonable good result. 

#### Training log
```
20224/20224 [==============================] - 98s - loss: 0.3264 - val_loss: 0.2548
Epoch 2/100
20224/20224 [==============================] - 97s - loss: 0.2504 - val_loss: 0.2130
Epoch 3/100
20224/20224 [==============================] - 96s - loss: 0.2218 - val_loss: 0.1548
Epoch 4/100
20224/20224 [==============================] - 97s - loss: 0.2020 - val_loss: 0.1457
Epoch 5/100
20224/20224 [==============================] - 97s - loss: 0.1875 - val_loss: 0.1414
Epoch 6/100
20224/20224 [==============================] - 96s - loss: 0.1810 - val_loss: 0.2068
Epoch 7/100
20224/20224 [==============================] - 95s - loss: 0.1708 - val_loss: 0.1707
Epoch 8/100
20224/20224 [==============================] - 95s - loss: 0.1659 - val_loss: 0.1376
Epoch 9/100
20224/20224 [==============================] - 96s - loss: 0.1626 - val_loss: 0.1167
Epoch 10/100
20224/20224 [==============================] - 95s - loss: 0.1513 - val_loss: 0.1543
Epoch 11/100
20224/20224 [==============================] - 95s - loss: 0.1478 - val_loss: 0.1748
Epoch 12/100
20224/20224 [==============================] - 96s - loss: 0.1496 - val_loss: 0.1645
Epoch 13/100
20224/20224 [==============================] - 95s - loss: 0.1462 - val_loss: 0.1634
Epoch 14/100
20224/20224 [==============================] - 95s - loss: 0.1425 - val_loss: 0.1278
Epoch 15/100
20224/20224 [==============================] - 96s - loss: 0.1390 - val_loss: 0.1962
Epoch 16/100
20224/20224 [==============================] - 95s - loss: 0.1386 - val_loss: 0.1285
Epoch 17/100
20224/20224 [==============================] - 95s - loss: 0.1322 - val_loss: 0.1365
Epoch 18/100
20224/20224 [==============================] - 96s - loss: 0.1346 - val_loss: 0.1064
Epoch 19/100
20224/20224 [==============================] - 95s - loss: 0.1285 - val_loss: 0.1198
Epoch 20/100
20224/20224 [==============================] - 95s - loss: 0.1306 - val_loss: 0.1068
Epoch 21/100
20224/20224 [==============================] - 95s - loss: 0.1301 - val_loss: 0.1170
Epoch 22/100
20224/20224 [==============================] - 95s - loss: 0.1265 - val_loss: 0.1417
Epoch 23/100
20224/20224 [==============================] - 95s - loss: 0.1267 - val_loss: 0.1676
Epoch 24/100
20224/20224 [==============================] - 95s - loss: 0.1291 - val_loss: 0.1534
Epoch 25/100
20224/20224 [==============================] - 97s - loss: 0.1276 - val_loss: 0.1056
Epoch 26/100
20224/20224 [==============================] - 98s - loss: 0.1226 - val_loss: 0.1221
Epoch 27/100
20224/20224 [==============================] - 97s - loss: 0.1251 - val_loss: 0.1334
Epoch 28/100
20224/20224 [==============================] - 95s - loss: 0.1239 - val_loss: 0.1072
Epoch 29/100
20224/20224 [==============================] - 96s - loss: 0.1226 - val_loss: 0.1392
Epoch 30/100
20224/20224 [==============================] - 99s - loss: 0.1241 - val_loss: 0.1036
Epoch 31/100
20224/20224 [==============================] - 99s - loss: 0.1239 - val_loss: 0.1428
Epoch 32/100
20224/20224 [==============================] - 98s - loss: 0.1205 - val_loss: 0.1229
Epoch 33/100
20224/20224 [==============================] - 95s - loss: 0.1200 - val_loss: 0.1460
Epoch 34/100
20224/20224 [==============================] - 95s - loss: 0.1170 - val_loss: 0.1097
Epoch 35/100
20224/20224 [==============================] - 95s - loss: 0.1174 - val_loss: 0.1307
Epoch 36/100
20224/20224 [==============================] - 95s - loss: 0.1212 - val_loss: 0.1145
Epoch 37/100
20224/20224 [==============================] - 95s - loss: 0.1181 - val_loss: 0.1025
Epoch 38/100
20224/20224 [==============================] - 95s - loss: 0.1198 - val_loss: 0.1305
Epoch 39/100
20224/20224 [==============================] - 96s - loss: 0.1176 - val_loss: 0.1309
Epoch 40/100
20224/20224 [==============================] - 95s - loss: 0.1182 - val_loss: 0.0961
Epoch 41/100
20224/20224 [==============================] - 95s - loss: 0.1156 - val_loss: 0.1043
Epoch 42/100
20224/20224 [==============================] - 95s - loss: 0.1167 - val_loss: 0.1259
Epoch 43/100
20224/20224 [==============================] - 96s - loss: 0.1158 - val_loss: 0.1182
Epoch 44/100
20224/20224 [==============================] - 97s - loss: 0.1168 - val_loss: 0.1766
Epoch 45/100
20224/20224 [==============================] - 96s - loss: 0.1147 - val_loss: 0.0958
Epoch 46/100
20224/20224 [==============================] - 97s - loss: 0.1165 - val_loss: 0.0892
Epoch 47/100
20224/20224 [==============================] - 96s - loss: 0.1134 - val_loss: 0.1322
Epoch 48/100
20224/20224 [==============================] - 96s - loss: 0.1136 - val_loss: 0.0943
Epoch 49/100
20224/20224 [==============================] - 95s - loss: 0.1111 - val_loss: 0.0957
Epoch 50/100
20224/20224 [==============================] - 96s - loss: 0.1120 - val_loss: 0.1182
Epoch 51/100
20224/20224 [==============================] - 96s - loss: 0.1116 - val_loss: 0.1098
Epoch 52/100
20224/20224 [==============================] - 95s - loss: 0.1083 - val_loss: 0.0810
Epoch 53/100
20224/20224 [==============================] - 96s - loss: 0.1133 - val_loss: 0.1309
Epoch 54/100
20224/20224 [==============================] - 95s - loss: 0.1103 - val_loss: 0.1172
Epoch 55/100
20224/20224 [==============================] - 95s - loss: 0.1140 - val_loss: 0.1180
Epoch 56/100
20224/20224 [==============================] - 95s - loss: 0.1089 - val_loss: 0.1087
Epoch 57/100
20224/20224 [==============================] - 95s - loss: 0.1094 - val_loss: 0.1850
Epoch 58/100
20224/20224 [==============================] - 95s - loss: 0.1091 - val_loss: 0.1025
Epoch 59/100
20224/20224 [==============================] - 95s - loss: 0.1098 - val_loss: 0.1397
Epoch 60/100
20224/20224 [==============================] - 95s - loss: 0.1098 - val_loss: 0.1223
Epoch 61/100
20224/20224 [==============================] - 96s - loss: 0.1103 - val_loss: 0.1335
Epoch 62/100
20224/20224 [==============================] - 95s - loss: 0.1099 - val_loss: 0.1319
Epoch 63/100
20224/20224 [==============================] - 95s - loss: 0.1095 - val_loss: 0.0895
Epoch 64/100
20224/20224 [==============================] - 95s - loss: 0.1092 - val_loss: 0.1503
Epoch 65/100
20224/20224 [==============================] - 95s - loss: 0.1088 - val_loss: 0.1530
Epoch 66/100
20224/20224 [==============================] - 96s - loss: 0.1099 - val_loss: 0.1386
Epoch 67/100
20224/20224 [==============================] - 95s - loss: 0.1066 - val_loss: 0.1302
Epoch 68/100
20224/20224 [==============================] - 96s - loss: 0.1079 - val_loss: 0.0918
Epoch 69/100
20224/20224 [==============================] - 95s - loss: 0.1079 - val_loss: 0.1187
Epoch 70/100
20224/20224 [==============================] - 95s - loss: 0.1067 - val_loss: 0.1039
Epoch 71/100
20224/20224 [==============================] - 95s - loss: 0.1081 - val_loss: 0.1347
Epoch 72/100
20224/20224 [==============================] - 95s - loss: 0.1074 - val_loss: 0.0963
Epoch 73/100
20224/20224 [==============================] - 95s - loss: 0.1039 - val_loss: 0.0905
Epoch 74/100
20224/20224 [==============================] - 95s - loss: 0.1056 - val_loss: 0.0924
Epoch 75/100
20224/20224 [==============================] - 95s - loss: 0.1034 - val_loss: 0.1315
Epoch 76/100
20224/20224 [==============================] - 95s - loss: 0.1062 - val_loss: 0.0984
Epoch 77/100
20224/20224 [==============================] - 96s - loss: 0.1039 - val_loss: 0.0765
Epoch 78/100
20224/20224 [==============================] - 95s - loss: 0.1051 - val_loss: 0.1103
Epoch 79/100
20224/20224 [==============================] - 95s - loss: 0.1055 - val_loss: 0.1188
Epoch 80/100
20224/20224 [==============================] - 95s - loss: 0.1078 - val_loss: 0.0543
Epoch 81/100
20224/20224 [==============================] - 95s - loss: 0.1063 - val_loss: 0.1250
Epoch 82/100
20224/20224 [==============================] - 95s - loss: 0.1057 - val_loss: 0.0899
Epoch 83/100
20224/20224 [==============================] - 95s - loss: 0.1062 - val_loss: 0.1200
Epoch 84/100
20224/20224 [==============================] - 95s - loss: 0.1058 - val_loss: 0.1176
Epoch 85/100
20224/20224 [==============================] - 95s - loss: 0.1050 - val_loss: 0.1128
Epoch 86/100
20224/20224 [==============================] - 95s - loss: 0.1043 - val_loss: 0.0904
Epoch 87/100
20224/20224 [==============================] - 95s - loss: 0.1055 - val_loss: 0.0764
Epoch 88/100
20224/20224 [==============================] - 95s - loss: 0.1031 - val_loss: 0.1293
Epoch 89/100
20224/20224 [==============================] - 95s - loss: 0.1038 - val_loss: 0.0979
Epoch 90/100
20224/20224 [==============================] - 95s - loss: 0.1039 - val_loss: 0.1064
Epoch 91/100
20224/20224 [==============================] - 95s - loss: 0.1080 - val_loss: 0.1233
Epoch 92/100
20224/20224 [==============================] - 95s - loss: 0.1033 - val_loss: 0.1457
Epoch 93/100
20224/20224 [==============================] - 94s - loss: 0.1030 - val_loss: 0.0997
Epoch 94/100
20224/20224 [==============================] - 94s - loss: 0.1038 - val_loss: 0.0955
Epoch 95/100
20224/20224 [==============================] - 95s - loss: 0.1040 - val_loss: 0.1081
Epoch 96/100
20224/20224 [==============================] - 94s - loss: 0.1029 - val_loss: 0.0832
Epoch 97/100
20224/20224 [==============================] - 94s - loss: 0.1026 - val_loss: 0.1222
Epoch 98/100
20224/20224 [==============================] - 94s - loss: 0.1025 - val_loss: 0.0982
Epoch 99/100
20224/20224 [==============================] - 94s - loss: 0.0993 - val_loss: 0.1049
Epoch 100/100
20224/20224 [==============================] - 94s - loss: 0.1028 - val_loss: 0.0872
```

### Future work
Try out comma.ai, VGG16 and ResNet model, as other students were successfully using those model.

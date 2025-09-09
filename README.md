# Videophone

Videophone with AI detection of people

## Data

### Type of data (format)

The Data is images of people that you want to recognize, there can be several people on a same picture (family pictures for example). These images can be of both formats : **JPG** or **PNG**. If some of your images have the recent format **HEIC** you can convert them to JPG using the python file **ConversionHEICtoJPG.py**. 

### Labeling the data

Once you've got your images in the right format (PNG or JPG) you have to organise them. You will classify them according to the labels you want to give them. To do so you create a folder **Labeled_data** in the **Videophone** folder (main repository). Then in this folder you put your data in a folder named **RawData**. And you create as many folders as needed in a folder named **Data** that you create in the folder **Labeled_data** for as many classes as you need (number of different people you want to recognize so 5 people = 5 folders). Name each of these folders with the name (or function) of the person you want to recognize. In the folder **Labeled_data** create a folder **CroppedFaces**. In this folder will be all the faces extracted from your raw data (not labeled) that you have put in the folder **RawData**. You can now launch the script **SelectFaces** which is in the main folder **Videophone** (main repository). This script will use the model **MTCNN** to detect faces in your images and select them in the pictures to focus on the faces and not the rest of the images as well as to get individual faces in the case of multiple faces in a single picture. They will all be of the same size in the ouput : **224x224 (RGB)**. They will be put in the folder **CroppedFaces**. Now it is your turn to do the job : you have to go in this folder **CroppedFaces** and do the labeling by hand, you put each picture of a person in the corresponding folder of her name (or function) that is in the folder **Labeled_data/Data**. Once you have done with all the pictures in the folder **CroppedFaces** then your labeling by hand is finished : GOOD WORK ! To have something a bit more organised and clean you can use the file **OrderImages.py** on the folders of the people Labeled to have the names of the pictures labeles clean and organised : **NameOfThePerson_NumberOfTheImage.jpg**.

### Training the model

#### Processing the data

The data is first separated between **test data** and **training data** (20% for testing and 80% for training). Then the training data is processed through data augmentation : 
  - random rotations
  - random horizontal flips
  - color changes (using color jitter : brightness, contrast, saturation etc...)

The architecture of the model is coded with Pytorch. The architecture of the model to classify people by their faces is the following one (very simple architecture that can be imroved but as simple as it is, has a high performance even on quite small datasets) :

  - a convolutional layer with 3 input channels (RGB) and 32 output channels and a kernel size of 3x3 with striding of 1 and padding of 1
  - a max pooling layer with a kernel size of 2x2 and a stride of 2 to divide the number of parameters by 2
  - a convolutional layer with 32 input channels (outputs of the max pooling layer) and 64 output channels and a kernel size of 3x3 with striding of 1 and padding of 1
  - a fully connected layer of input size 64 * 56 * 56 (output channels of convolutional layer * output dimension of convolutional layer) and output size 128
  - a fully connected layer of input size 128 (output size of the previous fully connected layer) and output size num_classes (number of different people to detect)
    
  ```bash
  self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
  self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
  self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
  self.fc1 = nn.Linear(64 * 56 * 56, 128)
  self.fc2 = nn.Linear(128, num_classes)
```

### Using the model

Once trained the model is saved (the weights of the model) in a file named : **person_recognition_1.pth**. You can then use these learned weights with the known architecture of the model to use the trained model to identify people if your classes. To do so you can use the file **ModelUsage.py** in the main folder **Videophone**. You can play with it by giving to it images to classify.

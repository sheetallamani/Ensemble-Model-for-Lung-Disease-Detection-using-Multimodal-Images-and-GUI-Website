School of
Electronics and Communication Engineering
Minor Project Report
on
ENSEMBLE MODEL FOR LUNG
DISEASE DETECTION USING
MULTIMODAL IMAGES
By:
1. Sheetal Lamani USN: 01FE21BEC401
2. Pavan Netrakar USN: 01FE21BEC402
3. Samyakth Malgatti USN: 01FE20BEC317
4. Swaroop Jartarghar USN: 01FE20BEC313
Semester: VI, 2022-2023
Under the Guidance of
Dr. S. R. Nirmala
1
K.L.E SOCIETY’S
KLE Technological University,
HUBBALLI-580031
2022-2023
SCHOOL OF ELECTRONICS AND COMMUNICATION
ENGINEERING
CERTIFICATE
This is to certify that project entitled “ Ensemble model for lung disease de-
tection using multimodal images” is a bonafide work carried out by the student
team of ”Sheetal Lamani -01FE21BEC401, Pavan G Netrakar -01FE21BEC402,
Samyakth Malgatti -01FE20BEC317, Swaroop Jartarghar -01FE20BEC313”.
The project report has been approved as it satisfies the requirements with respect to the
minor project work prescribed by the university curriculum for BE (V Semester) in School
of Electronics and Communication Engineering of KLE Technological University for the
academic year 2022-2023.
Dr. S. R. Nirmala Nalini C. Iyer Prof. Basavaraj A
Guide Head of School Registrar
External Viva:
Name of Examiners Signature with date
1.
2.
2
ACKNOWLEDGMENT
The sense of accomplishment that comes with having completed the
Implementation and Analysis of Ensemble model for lung disease detec-
tion, Would be incomplete if we didn’t mention the names of the people
who helped us complete it because of their clear guidance, support, and
motivation.We are grateful to our revered institute, KLE Technological
University, Hubballi, for providing us with the opportunity to realise a
long-held dream of reaching the top.
We express the deep sense of appreciation and Obeisance towards our
Head of School of Electronics and Communication, Dr. Nalini C. Iyer
for giving the motivation and direction required for taking this industrial
project to its completion. We sincerely thank our guide Prof. Bhagyashree
Kinnal for consistent support and suggestions.
We also thank the complete ISHA team for their guidance and support.
Finally, we would like to thank all those who either specifically or in a
indirect way made a difference in this project. We too offer profound
appreciation to our guardians who have acknowledged, encouraged and
helped in our endeavor.
-Sheetal Lamani, Pavan G N, Samyakth M, Swaroop J
3
ABSTRACT
The precise and prompt identification of lung disorders is essential for ef-
fective treatment, making them a serious problem in the field of healthcare.
In this effort, we suggest an ensemble model that makes use of multimodal
pictures to detect lung disease. Our model attempts to increase the preci-
sion and reliability of lung disease diagnosis by integrating different imaging
modalities, such as X-ray and CT scans. The ensemble model effectively
analyses and interprets multimodal pictures by integrating several machine
learning algorithms and approaches.
With this strategy, we want to improve the lung disease detection sys-
tem’s overall functionality and robustness. A carefully curated dataset is
used to test the proposed model, and the findings show that it performs
better than previous methods. This study makes a contribution to the field
of medical diagnostics by providing a thorough method for identifying lung
disease on the basis of many imaging modalities.
4
Contents
1 Introduction 7
1.1 Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
1.2 Objectives . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
1.3 Literature survey . . . . . . . . . . . . . . . . . . . . . . . 9
1.4 Problem statement . . . . . . . . . . . . . . . . . . . . . . 10
1.5 Application in Societal Context . . . . . . . . . . . . . . . 10
1.6 Organization of the report . . . . . . . . . . . . . . . . . . 11
2 Lung diseases 12
2.0.1 Details: . . . . . . . . . . . . . . . . . . . . . . . . 12
3 Datasets 15
4 System design 17
4.1 Design alternatives . . . . . . . . . . . . . . . . . . . . . . 17
4.1.1 Design 1: . . . . . . . . . . . . . . . . . . . . . . . 17
4.1.2 Design 2: . . . . . . . . . . . . . . . . . . . . . . . 19
5 Implementation details 21
5.1 System specifications . . . . . . . . . . . . . . . . . . . . . 21
5.2 Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
5.3 Flowchart . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
6 Results and discussions 25
6.1 Result Analysis . . . . . . . . . . . . . . . . . . . . . . . . 25
6.2 Experiment 1 . . . . . . . . . . . . . . . . . . . . . . . . . 25
6.3 Experiment 2 . . . . . . . . . . . . . . . . . . . . . . . . . 28
6.4 Experiment 3 . . . . . . . . . . . . . . . . . . . . . . . . . 31
7 Conclusions and future scope 32
7.1 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
7.2 Future scope . . . . . . . . . . . . . . . . . . . . . . . . . . 32
5
Chapter 1
Introduction
Globally, lung disorders Pneumonia,Adenocarcinoma,Large cell carci-
noma,Squamous cell carcinoma and Covid-19 disease pose a serious threat
to public health . For successful treatment and better patient outcomes,
these disorders must be identified early and correctly diagnosed. Tradi-
tional diagnostic techniques, such X-rays and CT scans, require human
interpretation, can be expensive, and can take a lot of time. Deep learn-
ing models have recently demonstrated encouraging results in automat-
ing the diagnosis of lung disorders, providing a quicker and more effective
substitute. In this study, we used a sizable dataset of chest X-ray pic-
tures and ct scan to construct a deep learning model for the categorization
of lung illnesses. We assessed the model’s effectiveness before releasing
it as a web-based tool to help medical practitioners identify and treat
lung ailments. Deep learning models are used in the proposed lung illness
classification system to divide chest X-ray images into four groups: nor-
mal,Pneumonia,Adenocarcinoma,Large cell carcinoma,Squamous cell car-
cinoma and Covid-19. A fully connected neural network is used to classify
the images after the features from the images have been extracted using
convolutional neural networks (CNNs). Metrics like accuracy, precision,
recall, and F1-score were used to assess the deep learning models’ perfor-
mance.
The system was implemented on a web application so that users could
upload their chest X-ray photos and promptly get a categorization result,
thus enhancing the system’s usefulness. This makes the system simple to
use for both users and medical professionals, offering a quick and precise
diagnosis for the early detection and treatment of lung disorders.no
6
1.1 Motivation
The goal of our research to classify lung disorders using a deep learning
model and to make it available for Doctors as web application to provide
quick response and accurate diagnosis of lung diseases.
Lung conditions like pneumonia , lung cancer and other disease are
common around the world and can have serious effects if not identified and
treated in a timely manner. However, getting an accurate diagnosis of these
disorders can take some time and requires specialised knowledge, making it
challenging for doctors in many regions to offer prompt diagnoses.Our re-
search intends to help doctors and healthcare professionals rapidly and
effectively diagnose and treat lung disorders by creating a deep learning
model that can reliably categorise lung diseases based on medical imaging
data. The model’s accessibility being made it available web application,
potentially helping those healthcare professional’s locations who might not
have access to specialised medical treatments.
1.2 Objectives
1. The goal of this research is to create a deep learning model that can
correctly categorise lung disorders based on images from chest
X-rays and CT scan. To be more precise, we want to divide photos
into 5 groups of normal ,Pneumonia ,Adenocarcinoma ,Large cell
carcinoma ,Squamous cell carcinoma and Covid-19.
2. Our ultimate goal is to provide a trustworthy and effective tool for
lung disease to available as a web application so that medical
practitioners may quickly and accurately diagnose patients.
7
1.3 Literature survey
[1]COVID-19 Classification using CT Scan Images with Resize-MobileNet
In this reaserch paper 61 COVID-19 patient images and 4326 chest CT
images of 43 healthy individuals were used. The dataset was divided into
a verification set (20%), training set (80%), and test set (1000 images).
A combination of MobileNet and Resize formed Resizing-MobileNet for
COVID-19 CT image identification. The Resize part was trained along-
side MobileNet, using N of 16 convolution kernels (3x3) for intermediate
layers and 7x7 kernels for the first and last layers. BatchNormalization and
LeakyReLU activation (slope coefficient of 0.2) were also utilized. Resizing-
MobileNet achieved an accuracy of 96.9%, sensitivity rate of 98.3%, and
specificity of 95.3%. It outperformed MobileNet, VGG19, Inception-V3,
and Densenet169 on Imagenet
[2]Detection and Classification of Lung Cancer Using VGG-16
This paper utilizes the Lunal6 dataset, consisting of subdirectories named
after patients’ ids. Each subfolder contains 180 2D picture cuts of 3D lung
images. The VGG Net is a convolutional neural network (CNN) architec-
ture trained to extract features and identify objects, including those that
are not clearly visible. It employs VGG 16 and VGG 19 models, with
16 and 19 weight layers, respectively, for object recognition. RGB images
of size 224x224 pixels serve as input. The network applies convolutional
layers with a 33% channel size and stride of 1, followed by maxpooling
layers for downsampling. Three fully connected layers with 4096, 4096,
and 1000 channels respectively, are used. The final layer employs a Soft-
max activation function for object classification. In summary, VGG Net is
a CNN architecture for object recognition, capable of handling highlights
and unclear objects. The VGG-16 model used achieves a training accuracy
of 99.84% and a validation accuracy of 88%.
[3] Diagnosing Covid-19 and Pneumonia from Chest CT-Scan and X-Ray
Images Using Deep Learning Technique
In this research study, data was collected from the Kaggle library. The
dataset includes CT scan images for COVID-19, pneumonia, and healthy
cases, as well as chest X-ray (CXR) images for the same categories. The
dataset contains a total of 19,820 CXR images and 2,481 CT scan images.
8
The models used are ResNet-50, DenseNet-121, Inception V3, MobileNet,
and VGG16. ResNet-50 has 50 layers and was pre-trained on the ImageNet
database. DenseNet-121 has 121 layers and also underwent pre-training on
ImageNet. Inception V3 incorporates various improvements and belongs
to the Inception family. MobileNet is an efficient architecture designed for
practical applications, utilizing depth-wise separable convolutions. VGG16
is known for its simplicity, with stacked convolutional layers and fully con-
nected layers. For CXR images, ResNet-50 achieved 86% accuracy, VGG16
achieved 99%, MobileNet achieved 86%, DenseNet-121 achieved 94%, and
Inception V3 achieved 88%. For CT scan images, ResNet-50 achieved 72%
accuracy, VGG16 achieved 82%, MobileNet achieved 95%, DenseNet-121
achieved 97%, and Inception V3 achieved 81%. In conclusion, VGG-16
demonstrated superior performance in detecting and classifying COVID-
19, pneumonia, and healthy patients using CXR images, while DenseNet-
121 showed accurate results specifically for COVID-19 and healthy patients
using CT scan data.
1.4 Problem statement
ENSEMBLE MODEL FOR LUNG DISEASE DETECTION
USING MULTIMODAL IMAGES.
1.5 Application in Societal Context
Deep learning models’ use in the area of medical image processing
has important societal repercussions. It can significantly improve patient
outcomes and perhaps save lives when lung diseases like lung cancer and
tuberculosis are detected early and correctly. These diagnostics can take a
while and are subject to human error, which can cause delays and incorrect
diagnosis.
Our study intends to increase the precision and efficacy of lung illness
diagnostics by creating a deep learning model for lung disease classification.
Global healthcare systems may be significantly impacted by this, especially
in places where access to medical professionals is constrained.
Additionally, we want to make the diagnosis process more approachable
and user-friendly by implementing the model on a web-based platform, so
empowering both patients and healthcare professionals.
9
1.6 Organization of the report
The chapters are organised as follows:
Chapter 1: It includes the report’s opening, which details the first actions
taken to comprehend the title and problem statement. It also includes a
literature review that examines and comprehends concepts like deep learn-
ing and neural networks in relation to social environments.
Chapter 2:This chapter presents an overview of lung diseases, discussing
their types and relevance to the project.
Chapter 3: This chapter presents an overview of Dataset , discussing num-
ber images (Xray/CT scan images)are required to the project.
Chapter 4: provides a summary of the deep learning model employed for
classification purposes.
Chapter 5:It summarises the system specification,algorithm and Flowchart
of project.
Chapter 6: It summarises the outcomes and result of the ensemble model
for lung disease detection.
Chapter 7: It includes the project’s conclusion section as well as the future
strategy for this project.
10
Chapter 2
Lung diseases
2.0.1 Details:
Adenocarcinoma : Adenocarcinoma is a type of lung cancer that
Figure 2.1: This is an image from a text that uses color to teach music.[4]
begins in the cells that line the alveoli, the air sacs in the lungs. Adeno-
carcinoma tends to grow and spread more slowly than other types of lung
cancer. Symptoms may include coughing, shortness of breath, chest pain,
and weight loss. Treatment options may include surgery, radiation therapy,
chemotherapy, targeted therapy, or a combination of these approaches.
Large cell carcinoma : Large cell carcinoma is a type of lung can-
Figure 2.2: This is an image from a text that uses color to teach music.[4]
cer that usually grows and spreads quickly. It is characterized by large,
abnormal cells and can occur in any part of the lung. Large cell carci-
noma accounts for approximately 10-15 of all lung cancer cases. Symptoms
may include coughing, shortness of breath, chest pain, and weight loss.
11
Treatment options may include surgery,radiation therapy,chemotherapy,
targeted therapy,or a combination of these approaches.
Squamous cell carcinoma : squamous cell carcinoma is a type of
Figure 2.3: This is an image from a text that uses color to teach music.[4]
lung cancer that typically starts in the bronchi, the tubes that carry air to
the lungs. It is strongly associated with smoking and accounts for approx-
imately 25-30 of all lung cancer cases. Symptoms may include coughing,
shortness of breath, chest pain, and weight loss. Treatment options may
include surgery, radiation therapy, chemotherapy, targeted therapy, or a
combination of these approaches.
Covid-19 disease : Covid-19 is an infectious respiratory illness
Figure 2.4: This is an image from a text that uses color to teach music.[4]
caused by the SARS-CoV-2 virus. It primarily affects the respiratory sys-
tem and can cause a range of symptoms from mild to severe, including
cough, fever, and shortness of breath. In severe cases, Covid-19 can lead
to pneumonia, acute respiratory distress syndrome (ARDS), and death.
Covid-19 has caused a global pandemic and has had a significant impact
on public health, healthcare systems, and the economy. Prevention mea-
sures include vaccination, social distancing, wearing masks, and practicing
good hygiene.
Pneumonia : Pneumonia is an inflammatory condition of the lung
affecting primarily the tiny air sacs known as alveoli. It can be caused
by various factors, including bacterial, viral, or fungal infections, as well
as exposure to pollutants, irritants, or allergens. Symptoms may include
coughing, fever, chest pain, and shortness of breath. Pneumonia can range
12
Figure 2.5: This is an image from a text that uses color to teach music.[4]
in severity from mild to life-threatening and can affect people of all ages,
but is particularly dangerous for young children, the elderly, and indi-
viduals with weakened immune systems. Treatment options may include
antibiotics, antiviral medications, or supportive care to manage symptoms.
13
Chapter 3
Datasets
This project’s dataset was obtained from Kaggle, a popular website for
sharing and discovering datasets[5].It consists of a varied selection of CT
scans and X-ray pictures pertinent to the identification of lung diseases.
To guarantee that the dataset would be appropriate for both training and
assessing the ensemble model, it was carefully chosen.1,764 CT scan images
and 6,076 X-ray images make up the dataset for this project, assuring a siz-
able amount of data for the ensemble model’s training and evaluation.The
CT scan and X-ray datasets are split into test, train, and validation sets us-
ing appropriate percentages. Both ResNet50 and VGG16 models are used
for both datasets, which consist of 5 classes for CT scans and 2 classes for
X-rays.
Figure 3.1
14
Disease Train Test Validation
Large cell carcinoma 115 51 21
Squamous carcinoma 155 51 21
Adenocarcinoma 195 120 23
Normal 298 250 64
Covid-19 200 150 50
Table 3.1
Above table shows the how CT scan data set being divided for train,test
and validation of 5 classes(Large cell carcinoma, Squamous carcinoma,
Adenocarcinoma, Normal and Covid-19).
Disease Train Test Validation
Pneumonia 1349 234 100
Normal 3883 390 120
Table 3.2
Above table shows the how X ray scan data set being divided for
train,test and validation of binary classes(Normal and Pneumonia) .
Since we utilize larger layered DL models ( as shown in fig4.1 and 4.2)
and have a smaller dataset of both X-ray and CT scan images, we prefer
data augmentation. Data augmentation is used to avoid overfitting of the
model. The augmented data is directly processed in the ML architecture
without saving the augmented images. Left shift, right shift, flipping of
images, and other techniques are employed in the data augmentation pro-
cess.
15
Chapter 4
System design
In this chapter, Interfaces are listed. The functions or techniques used to
achieve the required output, as well as the methods used in obtaining these
outputs, are mentioned below. Our project is divided into sections that
involve the use of different neural network models. As a result, a functional
block can be used to illustrate how this system functions.
4.1 Design alternatives
To accomplish semantic segmentation, PSP-Net and UNet are two com-
parable models that are employed.
4.1.1 Design 1:
Resnet-50 for Image Classification :
ResNet-50 is a convolutional neural network architecture designed for im-
age classification tasks. It is composed of a series of convolutional blocks,
with each block having multiple convolutional layers and skip connections.
Here is an overview of the steps involved in ResNet-50:
Figure 4.1: This is an image from a text that uses color to teach music.
• Input: The input to the network is an RGB image with dimensions of
224x224x3.
16
• Convolutional Layer: The image is passed through a convolutional
layer with 64 filters of size 7x7 and a stride of 2. Batch normalization
and ReLU activation are applied to the output.
• Max Pooling: The output of the convolutional layer is passed through
a max pooling layer with a pool size of 3x3 and a stride of 2.
• Convolutional Blocks: There are 4 convolutional blocks in the net-
work, each containing multiple convolutional layers and skip connec-
tions. The blocks are numbered from 1 to 4, and the number of filters
in each block increases as the network becomes deeper.
• Global Average Pooling: After the last convolutional block, the output
is passed through a global average pooling layer. This layer takes the
average of each feature map over its spatial dimensions.
• Fully Connected Layer: The output of the global average pooling
layer is flattened and passed through a fully connected layer with
1000 units. A softmax activation function is applied to produce the
final class probabilities.
• Output: The final output of the network is a probability distribution
over the 1000 possible classes in the ImageNet dataset.
To summarize, ResNet-50 comprises convolutional blocks with skip con-
nections, followed by global average pooling and a fully connected layer.
This architecture has shown impressive results in image classification tasks,
owing to the effective use of skip connections to address the vanishing gra-
dient problem.
17
4.1.2 Design 2:
VGG-16:
VGG16 (Visual Geometry Group 16) is a convolutional neural network
architecture designed for image classification tasks. It was developed by
the Visual Geometry Group at the University of Oxford and consists of 16
layers, including 13 convolutional layers and 3 fully connected layers.
Here is an overview of the VGG16 architecture:
Figure 4.2: This is an image from a text that uses color to teach music.
• Input: The input to the network is an RGB image with dimensions of
224x224x3.
• Convolutional Layers: There are 13 convolutional layers in the net-
work, all with a 3x3 filter size and a stride of 1. The number of filters
in each layer varies, starting from 64 filters in the first layer and dou-
bling in number for every subsequent layer, until the last two layers,
which each have 512 filters.
• Max Pooling Layers: After every two convolutional layers, a max
pooling layer with a pool size of 2x2 and a stride of 2 is added. This
reduces the spatial dimensions of the feature maps by half.
• Fully Connected Layers: After the last convolutional layer, the output
is flattened and passed through three fully connected layers with 4096,
4096, and 1000 units, respectively. The first two fully connected layers
also have a dropout rate of 0.5 to prevent overfitting. A softmax
activation function is applied to the output of the last fully connected
layer to produce the final class probabilities.
• Output: The final output of the network is a probability distribution
over the 1000 possible classes in the ImageNet dataset.
In summary, the VGG16 architecture consists of 13 convolutional layers,
followed by max pooling layers, three fully connected layers, and a softmax
18
activation function at the output layer. It is a popular model in computer
vision due to its simplicity and effectiveness, achieving high accuracy on
various image classification tasks.
19
Chapter 5
Implementation details
5.1 System specifications
Deep Learning
The deep learning field in machine learning teaches computers how to learn
from experiences, just like people do. Machine learning algorithms use
computer methods to ”learn” information directly from data, rather than
using a predefined equation as a model.
Initialization:
1. Required libraries -
Import required python libraries and the classification from Tensor
flow framework.
2. Load Dataset -
The dataset constuction was manually, with each image being thor-
oughly examined by a group of medical experts before being assigned
to the appropriate class. When training a deep learning model to at-
tain high accuracy, the annotation procedure made sure the dataset
was appropriately labelled.
3. Augmentaion and Pre-Processing -
To perform Data visualization on dataset and a deep learning model
is trained on the labelled dataset using a variety of methods, including
data augmentation, transfer learning, and hyperparameter tweaking.
The model is then assessed on a different set of test photos in order
to gauge its performance and correctness.
4. Creating Network -
For classifying lung diseases in our project, we used both the VGG16
20
and ResNet50 architectures. In comparison to training a model from
scratch, you can get higher accuracy and quicker convergence by util-
ising the pre-trained weights of these models.
5. Train Network -
The pre-trained models were loaded first, and fully linked layers were
then built on top of the models. The stochastic gradient descent
(SGD) optimizer and backpropagation technique were then used to
train the models on the training data in order to reduce the categorical
cross-entropy loss function.
To artificially expand the dataset’s size and avoid overfitting, we used
data augmentation techniques like rotation, flipping, and zooming
during training.
6. Test Network:
Loading the best check point from saved model weights to perform
prediction on input image.To get a test image from the test dataset
and passing it through the trained model will generate an output im-
age (masked image).The quality of the output obtained depends on
the accuracy of the our trained model.
7. Evaluate Performance:
Performance evaluation plays very important role when it comes to
predictability and efficiency of the model to produce the desired out-
put. So by using Dice Loss and IOU metrics for the analysis of train
and validation. Then eventually Plotting the graph of Train vs Valu-
ation.
21
5.2 Algorithm
To implement the whole model on Kaggle notebook by creating ot
to access GPU memory. The following steps show the algorithm of Our
project figure 5.1 .
Training Phase:
Step 1.Set the deep learning model’s initial parameters, such as the num-
ber of classes, the architecture (such as VGG16 or ResNet50), and any
hyperparameters (such as learning rate and batch size).
Step 2.Load the training data, which should be made up of chest X-ray
images and the labels that go with them (denoting whether or not there is
lung disease present).
Step 3.Preprocessing the data entails applying any required transforma-
tions, such as normalisation, scaling, and data augmentation methods like
flipping, rotating, or cropping.
Step 4.Using a predetermined ratio and divide the preprocessed data into
training ,testing and validation sets.
Step 5.Utilise backpropagation and stochastic gradient descent to train the
deep learning model on the training data. The model should calculate the
loss (for instance, using categorical cross-entropy) and modify the network
weights to minimise the loss throughout each training iteration.
Step 6.During training, keep an eye on how the model performs on the
validation set to look for overfitting and make the
Testing Phase:
Step 1. To get unbiased estimates of the model’s accuracy, precision, recall,
F1-score, and other important metrics, assess the final model’s performance
on a different test set.
Step 2.For later use, like model deployment in a web application, save the
trained model parameters in model itself.
22
5.3 Flowchart
Figure 5.1
The above figure 5.1 shows the Flow chart of Ensemble model for lung
disease detection using multimodal image. The flowchart shows three parts.
Firstly, the Data part is responsible for data construction. The Model
Training part handles model selection, training, comparison, and model
saving. Lastly, the Classified Output part is used for deploying the model
and testing the images.
23
Chapter 6
Results and discussions
6.1 Result Analysis
In the previous chapter, we applied classification algorithms to classify
diseases. We are creating two classification models for two different types
of images: CT scans and X-rays. We will use CNN models (VGG16 and
ResNet50) to train and test both models separately on CT scans and X-ray
images. The performance of both models will be compared, and the highly
accurate models will be deployed on our web application.
6.2 Experiment 1
Experiment 1 involves demonstrating DL classification models on CT scan
images with 5 classes using the ResNet-50 and VGG16 networks.
Resnet-50 for CT scan Images:
Figure 6.1
24
Figure 6.2
We utilized a dataset of 1764 images labeled into 5 classes.With ResNet-
50, we obtained a test accuracy of 94.24. However, to address the issue
of overfitting, we implemented data augmentation with epochs 56 and a
batch size of 20. This approach significantly improved the test accuracy to
95.27, as evidenced by the F1 score graph.The F1 score graph illustrates
the relationship between accuracy and epochs, figure 6.1 demonstrating
that accuracy increases with each epoch while figure 6.2 shows the loss
function gradually decreases.
VGG16 for CT scan Images:
Figure 6.3
We have obtained the 81.91 test accuracy for VGG16 as mentioned
in the F1 score graph for same of 5 classes.With provided below with
epoch size of 40 and batch size of 40.The F1 score graph illustrates the
relationship between accuracy and epochs,figure 6.3 demonstrating that
25
Figure 6.4
accuracy increases with each epoch while figure 6.4 shows the loss function
gradually decreases.
Comparison of Model:
Models Train Test Validation
Resnet50 99.98 94.22 95.63
VGG16 96.27 81.91 86.97
Table 6.1: Comparison table
In the conclusion, it was observed that ResNet50 demonstrated higher
accuracy compared to VGG16. Based on the outcomes of Experiment 1,
ResNet is deemed more suitable for deployment in the web application
due to its superior test accuracy.
26
6.3 Experiment 2
Experiment 2 involves demonstrating DL classification models on CT scan
images with 2(binary classification) classes using the ResNet-50 and VGG16
networks.
Resnet-50 for Xray Images:
Figure 6.5
Figure 6.6
We utilized a dataset of 6076 images labeled into 2 classes . With
ResNet-50, we obtained a test accuracy of 82.54 .The number of training
iterations or epochs is often represented by an F1 score or loss graph.With
provided below with epoch size of 70 and batch size of 32 . The F1 score
27
graph illustrates the relationship between accuracy and epochs figure 6.5
demonstrating that accuracy increases with each epoch while figure 6.6 the
loss function gradually decreases.
VGG 16 for Xray Images:
We have obtained the 94.23 test accuracy for VGG16 as mentioned in
the F1 score graph for same of 2 classes . With provided below with
epoch size of 70 and batch size of 32 . The F1 score graph illustrates the
relationship between accuracy and epochs,figure 6.7 demonstrating that
accuracy increases with each epoch while figure 6.8 shows the loss function
gradually decreases.
Figure 6.7
Figure 6.8
28
Comparison of Model:
Models Train Test Validation
Resnet50 94.28 82.54 88.87
VGG16 96.29 93.40 94.23
Table 6.2: Comparison table
Table 6.3
In the conclusion, it was observed that VGG16 demonstrated higher
accuracy compared to Rasnet50. Based on the outcomes of Experiment 2,
VGG16 is deemed more suitable for deployment in the web application
due to its superior test accuracy.
29
6.4 Experiment 3
User interface (GUI):
A well-liked Python web framework for creating web apps is called Flask.
It is a good option for implementing machine learning models because of
its simplicity, flexibility, and usability. We will go over Flask’s benefits as
the web framework for deploying the ML model in this part.
Figure 6.9
The models are saved in the file named model.h5. The path of the
model is defined within the Flask framework, which connects both the
backend and frontend. The backend contains the model.h5 file, while the
user interface (HTML) page are part of frontend. The accuracy of disease
detection is displayed within the user page. All coding for the framework
is implemented in Python.
In conclusion, healthcare professionals can access our GUI. This GUI is
user-friendly, allowing users to simply upload X-ray and CT scan images
through their operating system files and obtain the results on the same
page.The above figure 6.9 illustrates a small view of our GUI using the
Flask web framework.
30
Chapter 7
Conclusions and future scope
7.1 Conclusion
In this project, we used deep learning methods to classify lung diseases
using the VGG16 and ResNet50 models. In contrast to the ResNet50
model, which was trained on a multiclass classification issue with five
classes, the VGG16 model was trained on a binary classification problem
with two classes. Our experimental findings demonstrated the utility of
both models in categorising lung illnesses, with the ResNet50 and VGG 16
model performing better than the VGG16 model. Additionally, we noted
that the ResNet50 model outperformed the VGG16 model in terms of F1-
score, precision, and recall.
Additionally, we implemented the ResNet50 model in Flask, enabling
us to create a web application for classifying lung diseases. Medical practi-
tioners can submit chest X-ray and CT scan images into this user-friendly
web tool to get instantaneous predictions on the existence of lung disease.
7.2 Future scope
Overall, the creation and implementation of an ML multi-modal lung can-
cer detection model for the web can significantly influence lung cancer diag-
nosis and treatment, especially in regions with limited access to healthcare
resources. Patients in remote locations can get prompt and precise diag-
noses without having to travel . Major part of feature scope of multi-modal
can deployed in to mobile application.Patients in remote locations can get
prompt and precise diagnoses . Multi-modal methods, including merging
3D CT scans and MRI scans, can improve accuracy even more by offering
complementing data.
31
References
[1] Xupeng Han, Yuehui Chen,COVID-19 Classification using CT Scan
Images with Resize-MobileNet, Technology Journal of Advanced Research,
Ideas, and Innovations 10 Nov 2009.
[2] Dr. K. Ramanjaneyulu ,K. Hemanth Kumar , K. Snehith,G.Jyothirmai
,K. Venkata Krishna , Detection and Classification of Lung Cancer Using
VGG-16 , International Journal of Engineering Research and General Sci-
ence, 15-March-2023.
[3] Puneet Jain,S. Santhanalakshma, Diagnosing Covid-19 and Pneumonia
from Chest CT-Scan and X-Ray Images Using Deep Learning Technique,
15-March-2023.
[4] Website Link https://www.webmd.com/lung/lung-diseases-overview
[5] https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pn eu-
monia
https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
32

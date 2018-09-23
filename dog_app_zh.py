
# coding: utf-8

# ## 卷积神经网络（Convolutional Neural Network, CNN）
# 
# ## 项目：实现一个狗品种识别算法App
# 
# 在这个notebook文件中，有些模板代码已经提供给你，但你还需要实现更多的功能来完成这个项目。除非有明确要求，你无须修改任何已给出的代码。以**'(练习)'**开始的标题表示接下来的代码部分中有你需要实现的功能。这些部分都配有详细的指导，需要实现的部分也会在注释中以'TODO'标出。请仔细阅读所有的提示。
# 
# 除了实现代码外，你还**需要**回答一些与项目及代码相关的问题。每个需要回答的问题都会以 **'问题 X'** 标记。请仔细阅读每个问题，并且在问题后的 **'回答'** 部分写出完整的答案。我们将根据 你对问题的回答 和 撰写代码实现的功能 来对你提交的项目进行评分。
# 
# >**提示：**Code 和 Markdown 区域可通过 **Shift + Enter** 快捷键运行。此外，Markdown可以通过双击进入编辑模式。
# 
# 项目中显示为_选做_的部分可以帮助你的项目脱颖而出，而不是仅仅达到通过的最低要求。如果你决定追求更高的挑战，请在此 notebook 中完成_选做_部分的代码。
# 
# ---
# 
# ### 让我们开始吧
# 在这个notebook中，你将迈出第一步，来开发可以作为移动端或 Web应用程序一部分的算法。在这个项目的最后，你的程序将能够把用户提供的任何一个图像作为输入。如果可以从图像中检测到一只狗，它会输出对狗品种的预测。如果图像中是一个人脸，它会预测一个与其最相似的狗的种类。下面这张图展示了完成项目后可能的输出结果。（……实际上我们希望每个学生的输出结果不相同！）
# 
# ![Sample Dog Output](images/sample_dog_output.png)
# 
# 在现实世界中，你需要拼凑一系列的模型来完成不同的任务；举个例子，用来预测狗种类的算法会与预测人类的算法不同。在做项目的过程中，你可能会遇到不少失败的预测，因为并不存在完美的算法和模型。你最终提交的不完美的解决方案也一定会给你带来一个有趣的学习经验！
# 
# ### 项目内容
# 
# 我们将这个notebook分为不同的步骤，你可以使用下面的链接来浏览此notebook。
# 
# * [Step 0](#step0): 导入数据集
# * [Step 1](#step1): 检测人脸
# * [Step 2](#step2): 检测狗狗
# * [Step 3](#step3): 从头创建一个CNN来分类狗品种
# * [Step 4](#step4): 使用一个CNN来区分狗的品种(使用迁移学习)
# * [Step 5](#step5): 建立一个CNN来分类狗的品种（使用迁移学习）
# * [Step 6](#step6): 完成你的算法
# * [Step 7](#step7): 测试你的算法
# 
# 在该项目中包含了如下的问题：
# 
# * [问题 1](#question1)
# * [问题 2](#question2)
# * [问题 3](#question3)
# * [问题 4](#question4)
# * [问题 5](#question5)
# * [问题 6](#question6)
# * [问题 7](#question7)
# * [问题 8](#question8)
# * [问题 9](#question9)
# * [问题 10](#question10)
# * [问题 11](#question11)
# 
# 
# ---
# <a id='step0'></a>
# ## 步骤 0: 导入数据集
# 
# ### 导入狗数据集
# 在下方的代码单元（cell）中，我们导入了一个狗图像的数据集。我们使用 scikit-learn 库中的 `load_files` 函数来获取一些变量：
# - `train_files`, `valid_files`, `test_files` - 包含图像的文件路径的numpy数组
# - `train_targets`, `valid_targets`, `test_targets` - 包含独热编码分类标签的numpy数组
# - `dog_names` - 由字符串构成的与标签相对应的狗的种类

# # 提问1 为什么earlystop在我设置了patience等于10后，val_loss 已经开始上升了，但是还是没有自动停止？ ----Step3

# [sklearn.datasets](http://scikit-learn.org/stable/datasets/index.html) scikit-learn’s datasets.load_files for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
# ``` python
# from sklearn import datasets
# rawData = datasets.load_files("data_folder")
# 
# rawData
# Out[10]: 
# {'DESCR': None,
#  'data': ['5 start, \r\ni like this book.',
#   '4 start, \r\nthis book is good,\r\ni like it.',
#   "1 start, \r\npretty bad, don't like it at all.",
#   "2 start, \r\nwe don't like so much."],
#  'filenames': array(['data_folder\\positive_folder\\1.txt',
#         'data_folder\\positive_folder\\2.txt',
#         'data_folder\\negative_folder\\4.txt',
#         'data_folder\\negative_folder\\3.txt'], 
#        dtype='|S33'),
#  'target': array([1, 1, 0, 0]),
#  'target_names': ['negative_folder', 'positive_folder']}
# 
# rawData.data
# Out[11]: 
# ['5 start, \r\ni like this book.',
#  '4 start, \r\nthis book is good,\r\ni like it.',
#  "1 start, \r\npretty bad, don't like it at all.",
#  "2 start, \r\nwe don't like so much."]
# 
# rawData.target
# Out[12]: array([1, 1, 0, 0])
# 
# rawData.filenames[rawData.target[0]]
# Out[13]: 'data_folder\\positive_folder\\2.txt
# ```
# [glob](https://blog.csdn.net/sunhuaqiang1/article/details/70244497)
# 
# 
# -1 是去掉最后的`/`
# ``` python
# dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
# for item in glob("dogImages/train/*/"):
#     print (item[20:-1])
# ```    

# In[3]:


from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('/data/dog_images/train')
valid_files, valid_targets = load_dataset('/data/dog_images/valid')
test_files, test_targets = load_dataset('/data/dog_images/test')

# load list of dog names
dog_names = [item[27:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


# #### 修改一 
# 改正 从 [item[20:-1] 到 [item[27:-1]
# >``` python
# dog_names = [item[27:-1] for item in sorted(glob("/data/dog_images/train/*/"))]
# ```

# In[2]:


dog_names


# ### 导入人脸数据集
# 
# 在下方的代码单元中，我们导入人脸图像数据集，文件所在路径存储在名为 `human_files` 的 numpy 数组。

# 以下展示了使用 shuffle() 方法的实例：
# 
# 
# ```python
# import random
# 
# list = [20, 16, 10, 5];
# random.shuffle(list)
# print "随机排序列表 : ",  list
# 
# random.shuffle(list)
# print "随机排序列表 : ",  list
# 以上实例运行后输出结果为：
# 
# 随机排序列表 :  [16, 5, 10, 20]
# 随机排序列表 :  [16, 5, 20, 10]
# ```

# In[3]:


import random
random.seed(8675309)

# 加载打乱后的人脸数据集的文件名
human_files = np.array(glob("/data/lfw/*/*"))
random.shuffle(human_files)

# 打印数据集的数据量
print('There are %d total human images.' % len(human_files))


# In[4]:


human_files


# ---
# <a id='step1'></a>
# ## 步骤1：检测人脸
#  
# 我们将使用 OpenCV 中的 [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) 来检测图像中的人脸。OpenCV 提供了很多预训练的人脸检测模型，它们以XML文件保存在 [github](https://github.com/opencv/opencv/tree/master/data/haarcascades)。我们已经下载了其中一个检测模型，并且把它存储在 `haarcascades` 的目录中。
# 
# 在如下代码单元中，我们将演示如何使用这个检测模型在样本图像中找到人脸。
# 
# openCV cv2.imread() loads images as BGR while numpy.imread() loads them as RGB.

# In[5]:


import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')

# 提取预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# 加载彩色（通道顺序为BGR）图像
img = cv2.imread(human_files[3])

# 将BGR图像进行灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 在图像中找出脸
faces = face_cascade.detectMultiScale(gray)

# 打印图像中检测到的脸的个数
print('Number of faces detected:', len(faces))

# 获取每一个所检测到的脸的识别框
for (x,y,w,h) in faces:
    # 在人脸图像中绘制出识别框
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# 将BGR图像转变为RGB图像以打印
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 展示含有识别框的图像
plt.imshow(cv_rgb)
plt.show()


# 在使用任何一个检测模型之前，将图像转换为灰度图是常用过程。`detectMultiScale` 函数使用储存在 `face_cascade` 中的的数据，对输入的灰度图像进行分类。
# 
# 在上方的代码中，`faces` 以 numpy 数组的形式，保存了识别到的面部信息。它其中每一行表示一个被检测到的脸，该数据包括如下四个信息：前两个元素  `x`、`y` 代表识别框左上角的 x 和 y 坐标（参照上图，注意 y 坐标的方向和我们默认的方向不同）；后两个元素代表识别框在 x 和 y 轴两个方向延伸的长度 `w` 和 `d`。 
# 
# ### 写一个人脸识别器
# 
# 我们可以将这个程序封装为一个函数。该函数的输入为人脸图像的**路径**，当图像中包含人脸时，该函数返回 `True`，反之返回 `False`。该函数定义如下所示。

# In[6]:


# 如果img_path路径表示的图像检测到了脸，返回"True" 
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# ### **【练习】** 评估人脸检测模型

# 
# ---
# 
# <a id='question1'></a>
# ### __问题 1:__ 
# 
# 在下方的代码块中，使用 `face_detector` 函数，计算：
# 
# - `human_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？
# - `dog_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？
# 
# 理想情况下，人图像中检测到人脸的概率应当为100%，而狗图像中检测到人脸的概率应该为0%。你会发现我们的算法并非完美，但结果仍然是可以接受的。我们从每个数据集中提取前100个图像的文件路径，并将它们存储在`human_files_short`和`dog_files_short`中。

# #### 修改二
# 直接使用np.mean 来计算平均值,使用
# ``` python
# hface_in_human = [face_detector(item) for item in human_files_short]
# hface_in_human.count(True)/len(hface_in_human)*100
# 
# np.mean([face_detector(human) for human in human_files_short])
# ```

# In[7]:


human_files_short = human_files[:100]
dog_files_short = train_files[:100]
## 请不要修改上方代码


## TODO: 基于human_files_short和dog_files_short
## 中的图像测试face_detector的表现
#hface_in_human = [face_detector(item) for item in human_files_short]
#hface_in_dog = [face_detector(item) for item in dog_files_short]

faces_in_human = np.mean([face_detector(human) for human in human_files_short])
faces_in_dog = np.mean([face_detector(dog) for dog in dog_files_short])
# 打印数据集的数据量
print('There are {0:.0f}% huamn faces in human_files,'.format(faces_in_human*100),
     '{0:.0f}%  human faces in dog_files.'.format(faces_in_dog*100))


# ---
# 
# <a id='question2'></a>
# 
# ### __问题 2:__ 
# 
# 就算法而言，该算法成功与否的关键在于，用户能否提供含有清晰面部特征的人脸图像。
# 那么你认为，这样的要求在实际使用中对用户合理吗？如果你觉得不合理，你能否想到一个方法，即使图像中并没有清晰的面部特征，也能够检测到人脸？
# 
# __回答:__ 
# 
# 
# 我认为这样的要求不合理，因为在实际操作中，有很大的几率摄像头不会捕捉到清晰的人脸。
# 
# 
# 1.可以直接在opencv后面使用mlp或cnn进行训练-先用opencv处理图像，再用mlp和cnn对图像进行学习，并输出。
# 
# 2.如果使用CNN或者mlp，可以旋转，拉伸，模糊一下图像，并把这些生成的图像加入数据库，然后再次训练，这样训练好的识别器就可以学习到图片在不同情况下的信息。
# 
# 使用Haar Cascades技术来检测人脸的原理请参阅：[Face Detection using Haar Cascades] (https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html)

# ---
# 
# <a id='Selection1'></a>
# ### 选做：
# 
# 我们建议在你的算法中使用opencv的人脸检测模型去检测人类图像，不过你可以自由地探索其他的方法，尤其是尝试使用深度学习来解决它:)。请用下方的代码单元来设计和测试你的面部监测算法。如果你决定完成这个_选做_任务，你需要报告算法在每一个数据集上的表现。

# In[ ]:


features = [cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2GRAY) for item in human_files]
target = np.ones(human_files.shape[0])


# In[ ]:


features = np.array(features).reshape(np.array(features).shape[0], 1, 250, 250)


# In[ ]:


## (选做) TODO: 报告另一个面部检测算法在LFW数据集上的表现
### 你可以随意使用所需的代码单元数
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42) 


# In[ ]:


X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[ ]:


# break training set into training and validation sets
(X_train, X_valid) = X_train[5000:], X_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]


# print shape of training set
print('x_train shape:', X_train.shape)

# print number of training, validation, and test images
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_valid.shape[0], 'validation samples')


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
# create and configure augmented image generator
datagen_train = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True) # randomly flip images horizontally

# fit augmented image generator on data
datagen_train.fit(X_train)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# define the model
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# summarize the model
model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])


# In[ ]:


# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# In[ ]:


from keras.callbacks import ModelCheckpoint   

batch_size = 32
epochs = 10

# train the model
checkpointer = ModelCheckpoint(filepath='aug_model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)
model.fit_generator(datagen_train.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=epochs, verbose=2, callbacks=[checkpointer],
                    validation_data=(X_valid, y_valid),
                    validation_steps=X_valid.shape[0] // batch_size)


# verbose：日志显示
# 
# verbose = 0 为不在标准输出流输出日志信息
# 
# verbose = 1 为输出进度条记录
# 
# verbose = 2 为每个epoch输出一行记录
# 
# 注意： 默认为 1

# In[ ]:


#model.load_weights('dog.model.best.hdf5')
# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# ---
# <a id='step2'></a>
# 
# ## 步骤 2: 检测狗狗
# 
# 在这个部分中，我们使用预训练的 [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) 模型去检测图像中的狗。下方的第一行代码就是下载了 ResNet-50 模型的网络结构参数，以及基于 [ImageNet](http://www.image-net.org/) 数据集的预训练权重。
# 
# ImageNet 这目前一个非常流行的数据集，常被用来测试图像分类等计算机视觉任务相关的算法。它包含超过一千万个 URL，每一个都链接到 [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 中所对应的一个物体的图像。任给输入一个图像，该 ResNet-50 模型会返回一个对图像中物体的预测结果。

# In[8]:


from keras.applications.resnet50 import ResNet50

# 定义ResNet50模型
ResNet50_model = ResNet50(weights='imagenet')


# ### 数据预处理
# 
# - 在使用 TensorFlow 作为后端的时候，在 Keras 中，CNN 的输入是一个4维数组（也被称作4维张量），它的各维度尺寸为 `(nb_samples, rows, columns, channels)`。其中 `nb_samples` 表示图像（或者样本）的总数，`rows`, `columns`, 和 `channels` 分别表示图像的行数、列数和通道数。
# 
# 
# - 下方的 `path_to_tensor` 函数实现如下将彩色图像的字符串型的文件路径作为输入，返回一个4维张量，作为 Keras CNN 输入。因为我们的输入图像是彩色图像，因此它们具有三个通道（ `channels` 为 `3`）。
#     1. 该函数首先读取一张图像，然后将其缩放为 224×224 的图像。
#     2. 随后，该图像被调整为具有4个维度的张量。
#     3. 对于任一输入图像，最后返回的张量的维度是：`(1, 224, 224, 3)`。
# 
# 
# - `paths_to_tensor` 函数将图像路径的字符串组成的 numpy 数组作为输入，并返回一个4维张量，各维度尺寸为 `(nb_samples, 224, 224, 3)`。 在这里，`nb_samples`是提供的图像路径的数据中的样本数量或图像数量。你也可以将 `nb_samples` 理解为数据集中3维张量的个数（每个3维张量表示一个不同的图像。

# [image](https://tensorflow.google.cn/api_docs/python/tf/keras/preprocessing/image/load_img?hl=zh-cn)
# 
# Arguments:
# 
# path: Path to image file
# 
# grayscale: Boolean, whether to load the image as grayscale.
# 
# target_size: Either None (default to original size) or tuple of ints (img_height, img_width).
# 
# interpolation: Interpolation method used to resample the image if the target size is different from that of the loaded image. Supported methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3 or newer is installed, "lanczos" is also supported. If PIL version 3.4.0 or newer is installed, "box" and "hamming" are also supported. By default, "nearest" is used.
# 
# Returns:
# A PIL Image instance.
# 
# np.expand_dim-(https://blog.csdn.net/qq_16949707/article/details/53418912)
# 
# tqdm - (https://pypi.org/project/tqdm/)

# In[9]:


from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)#Converts a PIL Image instance to a Numpy array.
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# ### 基于 ResNet-50 架构进行预测
# 
# 对于通过上述步骤得到的四维张量，在把它们输入到 ResNet-50 网络、或 Keras 中其他类似的预训练模型之前，还需要进行一些额外的处理：
# 1. 首先，这些图像的通道顺序为 RGB，我们需要重排他们的通道顺序为 BGR。
# 2. 其次，预训练模型的输入都进行了额外的归一化过程。因此我们在这里也要对这些张量进行归一化，即对所有图像所有像素都减去像素均值 `[103.939, 116.779, 123.68]`（以 RGB 模式表示，根据所有的 ImageNet 图像算出）。
# 
# 导入的 `preprocess_input` 函数实现了这些功能。如果你对此很感兴趣，可以在 [这里](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py) 查看 `preprocess_input`的代码。
# 
# 
# 在实现了图像处理的部分之后，我们就可以使用模型来进行预测。这一步通过 `predict` 方法来实现，它返回一个向量，向量的第 i 个元素表示该图像属于第 i 个 ImageNet 类别的概率。这通过如下的 `ResNet50_predict_labels` 函数实现。
# 
# 通过对预测出的向量取用 argmax 函数（找到有最大概率值的下标序号），我们可以得到一个整数，即模型预测到的物体的类别。进而根据这个 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)，我们能够知道这具体是哪个品种的狗狗。
# 

# In[10]:


from keras.applications.resnet50 import preprocess_input, decode_predictions
def ResNet50_predict_labels(img_path):
    # 返回img_path路径的图像的预测向量
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


# ### 完成狗检测模型
# 
# 
# 在研究该 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 的时候，你会注意到，狗类别对应的序号为151-268。因此，在检查预训练模型判断图像是否包含狗的时候，我们只需要检查如上的 `ResNet50_predict_labels` 函数是否返回一个介于151和268之间（包含区间端点）的值。
# 
# 我们通过这些想法来完成下方的 `dog_detector` 函数，如果从图像中检测到狗就返回 `True`，否则返回 `False`。

# In[11]:


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


# ### 【作业】评估狗狗检测模型
# 
# ---
# 
# <a id='question3'></a>
# ### __问题 3:__ 
# 
# 在下方的代码块中，使用 `dog_detector` 函数，计算：
# 
# - `human_files_short`中图像检测到狗狗的百分比？
# - `dog_files_short`中图像检测到狗狗的百分比？

# #### 修改三
# 直接使用np.mean 来计算平均值,使用
# ``` python
# hface_in_human = [face_detector(item) for item in human_files_short]
# hface_in_human.count(True)/len(dog_in_human)*100
# 
# np.mean([dog_detector(human) for human in human_files_short])
# ```

# In[12]:


### TODO: 测试dog_detector函数在human_files_short和dog_files_short的表现

faces_in_human = np.mean([dog_detector(human) for human in human_files_short])
faces_in_dog = np.mean([dog_detector(dog) for dog in dog_files_short])
# 打印数据集的数据量
print('There are {0:.0f}% huamn faces in human_files,'.format(faces_in_human*100),
     '{0:.0f}%  human faces in dog_files.'.format(faces_in_dog*100))


# ---
# 
# <a id='step3'></a>
# 
# ## 步骤 3: 从头开始创建一个CNN来分类狗品种
# 
# 
# 现在我们已经实现了一个函数，能够在图像中识别人类及狗狗。但我们需要更进一步的方法，来对狗的类别进行识别。在这一步中，你需要实现一个卷积神经网络来对狗的品种进行分类。你需要__从头实现__你的卷积神经网络（在这一阶段，你还不能使用迁移学习），并且你需要达到超过1%的测试集准确率。在本项目的步骤五种，你还有机会使用迁移学习来实现一个准确率大大提高的模型。
# 
# 在添加卷积层的时候，注意不要加上太多的（可训练的）层。更多的参数意味着更长的训练时间，也就是说你更可能需要一个 GPU 来加速训练过程。万幸的是，Keras 提供了能够轻松预测每次迭代（epoch）花费时间所需的函数。你可以据此推断你算法所需的训练时间。
# 
# 值得注意的是，对狗的图像进行分类是一项极具挑战性的任务。因为即便是一个正常人，也很难区分布列塔尼犬和威尔士史宾格犬。
# 
# 
# 布列塔尼犬（Brittany） | 威尔士史宾格犬（Welsh Springer Spaniel）
# - | - 
# <img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">
# 
# 不难发现其他的狗品种会有很小的类间差别（比如金毛寻回犬和美国水猎犬）。
# 
# 
# 金毛寻回犬（Curly-Coated Retriever） | 美国水猎犬（American Water Spaniel）
# - | -
# <img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">
# 
# 同样，拉布拉多犬（labradors）有黄色、棕色和黑色这三种。那么你设计的基于视觉的算法将不得不克服这种较高的类间差别，以达到能够将这些不同颜色的同类狗分到同一个品种中。
# 
# 黄色拉布拉多犬（Yellow Labrador） | 棕色拉布拉多犬（Chocolate Labrador） | 黑色拉布拉多犬（Black Labrador）
# - | -
# <img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">
# 
# 我们也提到了随机分类将得到一个非常低的结果：不考虑品种略有失衡的影响，随机猜测到正确品种的概率是1/133，相对应的准确率是低于1%的。
# 
# 请记住，在深度学习领域，实践远远高于理论。大量尝试不同的框架吧，相信你的直觉！当然，玩得开心！
# 
# 
# ### 数据预处理
# 
# 
# 通过对每张图像的像素值除以255，我们对图像实现了归一化处理。

# In[13]:


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True #为什么？              

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# ### 【练习】模型架构
# 
# 
# 创建一个卷积神经网络来对狗品种进行分类。在你代码块的最后，执行 `model.summary()` 来输出你模型的总结信息。
#     
# 我们已经帮你导入了一些所需的 Python 库，如有需要你可以自行导入。如果你在过程中遇到了困难，如下是给你的一点小提示——该模型能够在5个 epoch 内取得超过1%的测试准确率，并且能在CPU上很快地训练。
# 
# ![Sample CNN](images/sample_cnn.png)

# ---
# 
# <a id='question4'></a>  
# 
# ### __问题 4:__ 
# 
# 在下方的代码块中尝试使用 Keras 搭建卷积网络的架构，并回答相关的问题。
# 
# 1. 你可以尝试自己搭建一个卷积网络的模型，那么你需要回答你搭建卷积网络的具体步骤（用了哪些层）以及为什么这样搭建。
# 2. 你也可以根据上图提示的步骤搭建卷积网络，那么请说明为何如上的架构能够在该问题上取得很好的表现。
# 
# __回答:__ 我是使用上图的提示搭建的网络，上图可以缓慢的缩减维度，而又在同时慢慢的增加深度。对比之前自己写的CNN，如果增加深度或者减少维度太快的话，会损失很多的信息。

# #### 修改四
# 1.不同基本CNN结构原理资料-http://cs231n.github.io/convolutional-networks/#architectures
# 
# >The input layer (that contains the image) should be divisible by 2 many times. Common numbers include 32 (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224 (e.g. common ImageNet ConvNets), 384, and 512.
# 
# >The conv layers should be using small filters (e.g. 3x3 or at most 5x5), using a stride of S=1, and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. That is, when F=3, then using P=1 will retain the original size of the input. When F=5, P=2. For a general F, it can be seen that P=(F−1)/2 preserves the input size. If you must use bigger filter sizes (such as 7x7 or so), it is only common to see this on the very first conv layer that is looking at the input image.
# 
# >The pool layers are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive fields (i.e. F=2), and with a stride of 2 (i.e. S=2). Note that this discards exactly 75% of the activations in an input volume (due to downsampling by 2 in both width and height). Another slightly less common setting is to use 3x3 receptive fields with a stride of 2, but this makes. It is very uncommon to see receptive field sizes for max pooling that are larger than 3 because the pooling is then too lossy and aggressive. This usually leads to worse performance.
# 
# 2.加入：
# 
# >Batch normalization layer用来解决Covariate Shift的问题 参考-https://www.cnblogs.com/guoyaohua/p/8724433.html
# 
# >atchNorm为什么NB呢，关键还是效果好。①不仅仅极大提升了训练速度，收敛过程大大加快；②还能增加分类效果，一种解释是这是类似于Dropout的一种防止过拟合的正则化表达方式，所以不用Dropout也能达到相当的效果；③另外调参过程也简单多了，对于初始化要求没那么高，而且可以使用大的学习率等。总而言之，经过这么简单的变换，带来的好处多得很，这也是为何现在BN这么快流行起来的原因。
# 
# >Dropout layer用来降低模型复杂度，增强模型的泛化能力，防止过拟合，顺带降低了运算量

# In[14]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.models import Sequential

model = Sequential()


### TODO: 定义你的网络架构
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu', input_shape=(224, 224, 3)))   
model.add(BatchNormalization(axis = 1 ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization(axis = 1 ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization(axis = 1 ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())
model.add(Dense(133, activation='softmax'))
model.summary()


# In[15]:


## 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ---

# ## 【练习】训练模型
# 
# 
# ---
# 
# <a id='question5'></a>  
# 
# ### __问题 5:__ 
# 
# 在下方代码单元训练模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。
# 
# 可选题：你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)，来优化模型的表现。
# 
# 

# In[16]:


'''
from keras.preprocessing.image import ImageDataGenerator

# create and configure augmented image generator
datagen_train = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True) # randomly flip images horizontally

# fit augmented image generator on data
datagen_train.fit(train_tensors)

from keras.callbacks import ModelCheckpoint   

batch_size = 32
epochs = 100

# train the model
checkpointer = ModelCheckpoint(filepath='augtest/aug_model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)
model.fit_generator(datagen_train.flow(train_tensors, train_targets, batch_size=batch_size),
                    steps_per_epoch=train_tensors.shape[0] // batch_size,
                    epochs=epochs, verbose=2, callbacks=[checkpointer],
                    validation_data=(valid_tensors, valid_targets),
                    validation_steps=valid_tensors.shape[0] // batch_size)
                    '''


# #### 修改五
# 可以使用keras里的回调函数，就是当你的validation loss开始上升的时候，就马上停止训练，是为了防止过拟合的，参考代码如下：
# ```python
#   keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# 
# keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10, verbose = 1)
#   ```
# 或者你也可以把epoch & model accuracy和epoch & model loss的关系图打印出来，然后找一个比较满意的epoch，参考代码如下：
# 
# ```python
#     # Fit the model
#     history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
#     # list all data in history
#     print(history.history.keys())
#     # summarize history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
# ```

# In[17]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

### TODO: 设置训练模型的epochs的数量

epochs = 150

### 不要修改下方代码

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',  monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystopper = EarlyStopping(monitor='val_loss', patience = 10, verbose = 1)
model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer, earlystopper], verbose=1)


# In[18]:


## 加载具有最好验证loss的模型
model.load_weights('saved_models/weights.best.from_scratch.hdf5')


# ### 测试模型
# 
# 在狗图像的测试数据集上试用你的模型。确保测试准确率大于1%。

# In[19]:


# 获取测试数据集中每一个图像所预测的狗品种的index
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# Test accuracy: 4.1866% 提升-Test accuracy: 19.0191%
# 
# 提高准确率有很多小技巧～
# 
# >1.你可以使劲往上加层，直到它在测试集上过拟合，然后再加正则化和数据增强
# 
# >2.如果不过拟合了，再接着往上加层
# 
# 通常模型的大小取决于数据的量和复杂度，但是如果你使用max-pooling，你需要增加向上的每一层的神经元（比如你可以double一下）。通常在dense layer之前有2-5层，kernel size 3-5就差不多。你也可以用grid search找一组比较满意的参数～
# 
# 常用的正则化方法：
# 
# >batch normalization. 防止梯度消失～你可以参阅这篇文章：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v3.pdf)
# 
# >Max-Norm regularization & Dropout. 你可以参阅这篇文章: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
# 
# >L1 / L2 weight regularization
# Sparsity regularization (e.g. [Sparse deep belief net model for visual area V2] (http://web.eecs.umich.edu/~honglak/nips07-sparseDBN.pdf))
# 
# >Gradient clipping (在成本领域进行更彻底的搜索)
# 
# >Data augmentation. Data augmentation可以增加你的数据集，从而防止过度拟合。而且max-out units在最近的图像分类竞赛中很成功: [Galaxy Zoo challenge on Kaggle](https://benanne.github.io/2014/04/05/galaxy-zoo.html) 和 [Classifying plankton with deep neural networks](https://benanne.github.io/2015/03/17/plankton.html)
# 
# （出自： [some advices about how to improve the performance of Convolutional Neural Networks）](https://www.researchgate.net/post/Could_you_give_me_some_advices_about_how_to_improve_the_performance_of_Convolutional_Neural_Networks)
# 
# 更多的阅读资料：
# 
# >[What is maxout in neural network?](https://stats.stackexchange.com/questions/129698/what-is-maxout-in-neural-network)
# 
# >[What is the difference between max pooling and max out?](https://www.quora.com/What-is-the-difference-between-max-pooling-and-max-out)
# 
# >[Maxout Networks](https://arxiv.org/pdf/1302.4389.pdf)

# ---
# <a id='step4'></a>
# ## 步骤 4: 使用一个CNN来区分狗的品种
# 
# 
# 使用 迁移学习（Transfer Learning）的方法，能帮助我们在不损失准确率的情况下大大减少训练时间。在以下步骤中，你可以尝试使用迁移学习来训练你自己的CNN。
# 

# ### 得到从图像中提取的特征向量（Bottleneck Features）

# In[64]:


bottleneck_features = np.load('/data/bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']


# ### 模型架构
# 
# 该模型使用预训练的 VGG-16 模型作为固定的图像特征提取器，其中 VGG-16 最后一层卷积层的输出被直接输入到我们的模型。我们只需要添加一个全局平均池化层以及一个全连接层，其中全连接层使用 softmax 激活函数，对每一个狗的种类都包含一个节点。

# In[65]:


VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()


# In[66]:


## 编译模型

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[67]:


## 训练模型

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)



# In[68]:


## 加载具有最好验证loss的模型

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')


# ### 测试模型
# 现在，我们可以测试此CNN在狗图像测试数据集中识别品种的效果如何。我们在下方打印出测试准确率。

# In[69]:


# 获取测试数据集中每一个图像所预测的狗品种的index
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ### 使用模型预测狗的品种

# In[37]:


from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]


# ---
# <a id='step5'></a>
# ## 步骤 5: 建立一个CNN来分类狗的品种（使用迁移学习）
# 
# 现在你将使用迁移学习来建立一个CNN，从而可以从图像中识别狗的品种。你的 CNN 在测试集上的准确率必须至少达到60%。
# 
# 在步骤4中，我们使用了迁移学习来创建一个使用基于 VGG-16 提取的特征向量来搭建一个 CNN。在本部分内容中，你必须使用另一个预训练模型来搭建一个 CNN。为了让这个任务更易实现，我们已经预先对目前 keras 中可用的几种网络进行了预训练：
# 
# - [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
# - [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
# - [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
# - [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features
# 
# 这些文件被命名为为：
# 
#     Dog{network}Data.npz
# 
# 其中 `{network}` 可以是 `VGG19`、`Resnet50`、`InceptionV3` 或 `Xception` 中的一个。选择上方网络架构中的一个，他们已经保存在目录 `/data/bottleneck_features/` 中。
# 
# 
# ### 【练习】获取模型的特征向量
# 
# 在下方代码块中，通过运行下方代码提取训练、测试与验证集相对应的bottleneck特征。
# 
#     bottleneck_features = np.load('/data/bottleneck_features/Dog{network}Data.npz')
#     train_{network} = bottleneck_features['train']
#     valid_{network} = bottleneck_features['valid']
#     test_{network} = bottleneck_features['test']

# 关于四个架构的区别，请参考这篇文章：ImageNet: [VGGNet, ResNet, Inception, and Xception with Keras](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)

# In[28]:


### TODO: 从另一个预训练的CNN获取bottleneck特征
bottleneck_features_VGG19 = np.load('/data/bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features_VGG19['train']
valid_VGG19 = bottleneck_features_VGG19['valid']
test_VGG19 = bottleneck_features_VGG19['test']


# In[33]:


### TODO: 从另一个预训练的CNN获取bottleneck特征
bottleneck_features_Resnet50 = np.load('/data/bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features_Resnet50['train']
valid_Resnet50 = bottleneck_features_Resnet50['valid']
test_Resnet50 = bottleneck_features_Resnet50['test']


# In[16]:


### TODO: 从另一个预训练的CNN获取bottleneck特征
bottleneck_features_Inception = np.load('/data/bottleneck_features/DogInceptionV3Data.npz')
train_Inception = bottleneck_features_Inception['train']
valid_Inception = bottleneck_features_Inception['valid']
test_Inception = bottleneck_features_Inception['test']


# In[17]:


### TODO: 从另一个预训练的CNN获取bottleneck特征
bottleneck_features_Xception = np.load('/data/bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features_Xception['train']
valid_Xception = bottleneck_features_Xception['valid']
test_Xception = bottleneck_features_Xception['test']


# ### 【练习】模型架构
# 
# 建立一个CNN来分类狗品种。在你的代码单元块的最后，通过运行如下代码输出网络的结构：
#     
#         <your model's name>.summary()
#    
# ---
# 
# <a id='question6'></a>  
# 
# ### __问题 6:__ 
# 
# 
# 在下方的代码块中尝试使用 Keras 搭建最终的网络架构，并回答你实现最终 CNN 架构的步骤与每一步的作用，并描述你在迁移学习过程中，使用该网络架构的原因。
# 
# 
# __回答:__ 
# 
# ### 1.
# 对于每个模型我都训练了20个epochs，batch_size=20
# 
# >VGG -72.7273%
# 
# >ResNet-50 - 82.6555%
# 
# >Inception - 80.6220%
# 
# >Xception - 80.6220%
# 
# 从准确率上来看，ResNet50有最高的准确率，对于内存的要求来说Xception最小，在准确率上，我认为通过微调，Xception可以达到和Resnet相当的水平，所以在这里我选择Xception作为架构。
# 
# 我的网络架构如下，首先是GAP层-提取和压缩经过Xception Net学习的特征，然后送入有500个节点的FC全连接层，然后设定Dropout为0.2，最后是133的全连接层，使用softmax分类。
# 
# 从imagenet获取的inception训练模型，已经包含了足够的图像信息，不用再添加额外的Conv层来学习数据，所以在这里我选择了使用GAP层去压缩和总结信息。
# 
# 之后，使用一个500个节点的FC全连接层。使用这个全连接层的原因是，把inception最后的全连接层除以2得来的。
# 
# 之后，使用relu激活函数，0.5的dropout率是给予经验选择的，其实在最后的验证率上0.5， 0.4， 0.3， 0.2都差不多。
# 
# ### 2.
# 
# 为什么这一架构会在这一分类任务中成功？
# >这四个架构都是经过反复多次实验确定的，非常有效果的架构。以Inception net为例，inception net是多层特征提取器，通过分别多次同时提取特征，然后叠加，就可以学到不同层次的特征，所以效果非常好。
# 
# 为什么早期（第三步 ）的尝试不成功？
# >第三步中，第一，使用的网络在架构上，非常浅，学到的特征非常少，其次学习库非常小，上面四个网络是在Imagenet上经过大量训练在不同种类的训练集上得来的，这是这个小库无法比拟的。

# In[58]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.models import Sequential

test_model = Sequential()
### TODO: 定义你的框架
test_model = Sequential()
test_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
test_model.add(Dense(500))
test_model.add(Activation("relu"))
test_model.add(Dropout(0.5))
test_model.add(Dense(133, activation='softmax'))

test_model.summary()


# In[59]:


test_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# 推荐尝试一下Adam优化器的，时下比较流行，相比于AdaGrad, RMSProp, SGDNesterov 和 AdaDelta来说效率更高～可以参考一下这篇文章：[Gentle Introduction to the Adam Optimization Algorithm for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
# 就像你在第三步时用的那样～

# ---
# 
# ### 【练习】训练模型
# 
# <a id='question7'></a>  
# 
# ### __问题 7:__ 
# 
# 在下方代码单元中训练你的模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。
# 
# 当然，你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 以优化模型的表现，不过这不是必须的步骤。
# 

# In[37]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
### TODO: 训练模型
checkpointer_VGG19 = ModelCheckpoint(filepath='saved_models/weights.best.VGG19_model.hdf5', 
                               verbose=1, save_best_only=True)

test_model.fit(train_VGG19, train_targets, 
          validation_data=(valid_VGG19, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer_VGG19], verbose=1)


# In[50]:


checkpointer_Resnet = ModelCheckpoint(filepath='saved_models/weights.best.Resnet_model.hdf5', 
                               verbose=1, save_best_only=True)

test_model.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer_Resnet], verbose=1)


# In[55]:


checkpointer_Inception = ModelCheckpoint(filepath='saved_models/weights.best.Inc_model.hdf5', 
                               verbose=1, save_best_only=True)

test_model.fit(train_Inception, train_targets, 
          validation_data=(valid_Inception, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer_Inception], verbose=1)


# In[60]:


checkpointer_X = ModelCheckpoint(filepath='saved_models/weights.best.X_model.hdf5', 
                               verbose=1, save_best_only=True)

test_model.fit(train_Xception, train_targets, 
          validation_data=(valid_Xception, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer_X], verbose=1)


# In[41]:


test_model.load_weights('saved_models/weights.best.VGG19_model.hdf5')
VGG19_predictions = [np.argmax(test_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]


# In[51]:


test_model.load_weights('saved_models/weights.best.Resnet_model.hdf5')
Resnet_predictions = [np.argmax(test_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]


# In[56]:


test_model.load_weights('saved_models/weights.best.Inc_model.hdf5')
Inception_predictions = [np.argmax(test_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Inception]


# In[61]:


test_model.load_weights('saved_models/weights.best.X_model.hdf5')
Xception_predictions = [np.argmax(test_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]


# ---
# 
# ### 【练习】测试模型
# 
# <a id='question8'></a>  
# 
# ### __问题 8:__ 
# 
# 在狗图像的测试数据集上试用你的模型。确保测试准确率大于60%。

# In[43]:


test_accuracy_VGG19 = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
print('Test accuracy: %.4f%%' % test_accuracy_VGG19)


# In[52]:


test_accuracy_Resnet = 100*np.sum(np.array(Resnet_predictions)==np.argmax(test_targets, axis=1))/len(Resnet_predictions)
print('Test accuracy: %.4f%%' % test_accuracy_Resnet)


# In[57]:


### TODO: 在测试集上计算分类准确率
test_accuracy = 100*np.sum(np.array(Inception_predictions)==np.argmax(test_targets, axis=1))/len(Inception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# In[62]:


test_accuracy_Xception = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ---
# 
# ### 【练习】使用模型测试狗的品种
# 
# 
# 实现一个函数，它的输入为图像路径，功能为预测对应图像的类别，输出为你模型预测出的狗类别（`Affenpinscher`, `Afghan_hound` 等）。
# 
# 与步骤5中的模拟函数类似，你的函数应当包含如下三个步骤：
# 
# 1. 根据选定的模型载入图像特征（bottleneck features）
# 2. 将图像特征输输入到你的模型中，并返回预测向量。注意，在该向量上使用 argmax 函数可以返回狗种类的序号。
# 3. 使用在步骤0中定义的 `dog_names` 数组来返回对应的狗种类名称。
# 
# 提取图像特征过程中使用到的函数可以在 `extract_bottleneck_features.py` 中找到。同时，他们应已在之前的代码块中被导入。根据你选定的 CNN 网络，你可以使用 `extract_{network}` 函数来获得对应的图像特征，其中 `{network}` 代表 `VGG19`, `Resnet50`, `InceptionV3`, 或 `Xception` 中的一个。
#  
# ---
# 
# <a id='question9'></a>  
# 
# ### __问题 9:__

# In[45]:


### TODO: 写一个函数，该函数将图像的路径作为输入
def dog_names_guess(img_path):
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    index = np.argmax(Inception_model.predict(bottleneck_feature))
    return dog_names[index]
### 然后返回此模型所预测的狗的品种


# ---
# 
# <a id='step6'></a>
# ## 步骤 6: 完成你的算法
# 
# 
# 
# 实现一个算法，它的输入为图像的路径，它能够区分图像是否包含一个人、狗或两者都不包含，然后：
# 
# - 如果从图像中检测到一只__狗__，返回被预测的品种。
# - 如果从图像中检测到__人__，返回最相像的狗品种。
# - 如果两者都不能在图像中检测到，输出错误提示。
# 
# 我们非常欢迎你来自己编写检测图像中人类与狗的函数，你可以随意地使用上方完成的 `face_detector` 和 `dog_detector` 函数。你__需要__在步骤5使用你的CNN来预测狗品种。
# 
# 下面提供了算法的示例输出，但你可以自由地设计自己的模型！
# 
# ![Sample Human Output](images/sample_human_output.png)
# 
# 
# 
# 
# <a id='question10'></a>  
# 
# ### __问题 10:__
# 
# 在下方代码块中完成你的代码。
# 
# ---
# 

# In[48]:


### TODO: 设计你的算法
def dog_human_guess(img_path):
    if dog_detector(img_path) == True:
        message = 'The breed of this dog is'
        value = dog_names_guess(img_path)
    elif face_detector(img_path) == True:
        message = 'You look like a...'
        value = dog_names_guess(img_path)
    else:
        return 'No human or dog'
        
    return "{}:{}".format(message, value)
### 自由地使用所需的代码单元数吧


# ---
# <a id='step7'></a>
# ## 步骤 7: 测试你的算法
# 
# 在这个部分中，你将尝试一下你的新算法！算法认为__你__看起来像什么类型的狗？如果你有一只狗，它可以准确地预测你的狗的品种吗？如果你有一只猫，它会将你的猫误判为一只狗吗？
# 
# 
# <a id='question11'></a>  
# 
# ### __问题 11:__
# 
# 在下方编写代码，用至少6张现实中的图片来测试你的算法。你可以使用任意照片，不过请至少使用两张人类图片（要征得当事人同意哦）和两张狗的图片。
# 同时请回答如下问题：
# 
# 1. 输出结果比你预想的要好吗 :) ？或者更糟 :( ？
# >输出的结果，我是比较满意的。
# 2. 提出至少三点改进你的模型的想法。
# 
# >a. 提高训练的epoch数，这里只训练了20个比较少
# 
# >b. 使用数据增强
# 
# >c. 把人和狗的图片做相似度分析，把相似度作为变量加入模型。

# In[7]:


import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread('/data/dog_images/test/001.Affenpinscher/Affenpinscher_00003.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('on') # 不显示坐标轴
plt.show()

lena = mpimg.imread('/data/dog_images/test/002.Afghan_hound/Afghan_hound_00116.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('on') # 不显示坐标轴
plt.show()

lena = mpimg.imread('/data/dog_images/test/003.Airedale_terrier/Airedale_terrier_00175.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('on') # 不显示坐标轴
plt.show()

lena = mpimg.imread('/data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('on') # 不显示坐标轴
plt.show()

lena = mpimg.imread('/data/lfw/Aaron_Guiel/Aaron_Guiel_0001.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('on') # 不显示坐标轴
plt.show()

lena = mpimg.imread('/data/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('on') # 不显示坐标轴
plt.show()


# In[49]:


## TODO: 在你的电脑上，在步骤6中，至少在6张图片上运行你的算法。

print(dog_human_guess("/data/dog_images/test/001.Affenpinscher/Affenpinscher_00003.jpg"))
print(dog_human_guess("/data/dog_images/test/002.Afghan_hound/Afghan_hound_00116.jpg"))
print(dog_human_guess("/data/dog_images/test/003.Airedale_terrier/Airedale_terrier_00175.jpg"))
print(dog_human_guess("/data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"))
print(dog_human_guess("/data/lfw/Aaron_Guiel/Aaron_Guiel_0001.jpg"))
print(dog_human_guess("/data/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg"))
## 自由地使用所需的代码单元数吧


# **注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出File -> Download as -> HTML (.html)把这个 HTML 和这个 iPython notebook 一起做为你的作业提交。**

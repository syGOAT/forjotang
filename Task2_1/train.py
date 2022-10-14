import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/test'

'''
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, 
featurewise_std_normalization = False, samplewise_std_normalization = False, 
zca_whitening = False, rotation_range = 0., width_shift_range = 0., height_shift_range = 0., 
shear_range = 0., zoom_range = 0., channel_shift_range = 0., fill_mode = 'nearest', cval = 0.0, 
horizontal_flip = False, vertical_flip = False, rescale = None, preprocessing_function = None, 
data_format = K.image_data_format(), )
每个参数的解释参考 https://blog.csdn.net/jacke121/article/details/79245732，官网中文https://keras.io/zh/preprocessing/image/
对rescale的理解: 对图片的每个像素值乘上rescale，把像素值放缩到0和1之间有利于模型的收敛，避免神经元“死亡”
'''
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

'''https://blog.csdn.net/baixue0729/article/details/96168979
生成一批增强数据
def flow_from_directory(self,
                        directory: Any, # 目标目录路径
                        target_size: Tuple[int, int] = (256, 256), # 目标大小
                        color_mode: str = 'rgb', # 颜色模式, grayscale, rgb, rgba
                        classes: Any = None, # 类列表
                        class_mode: str = 'categorical', # categorical, binary, sparse, input, None
                        batch_size: int = 32,
                        shuffle: bool = True,
                        seed: Any = None,
                        save_to_dir: Any = None,
                        save_prefix: str = '',
                        save_format: str = 'png',
                        follow_links: bool = False,
                        subset: Any = None,
                        interpolation: str = 'nearest') -> DirectoryIterator
'''
batch_size = 64
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",  # color_mode="gray_framescale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",  # color_mode="gray_framescale",
        class_mode='categorical')

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
# 卷积层，输出通道数目32，卷积核3*3，激活函数relu，输入尺寸48*48*1
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
# 三个卷积块 ↑ + 全连接 ↓
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('之前的w/emotion_model0.h5')

cv2.ocl.setUseOpenCL(False)  # 是否使用opencl，据说可以加速opencv

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4, decay=1e-6),metrics=['accuracy'])
emotion_model_info = emotion_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=28709 // batch_size,   # 每次循环过多少批，样本数 // batch_size
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // batch_size)
emotion_model.save_weights('emotion_model.h5')  # 保存emotion_model的权重

def video_pre():
    # start the webcam feed
    cap = cv2.VideoCapture(0)  # 0表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        # read()按帧读取视频。ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵
        if not ret:
            break
        bounding_box = cv2.CascadeClassifier('D:/anaconda3/envs/py37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
                        # 级联分类器。大概原理是判别某个物体是否属于某个分类。以人脸为例，我们可以把眼睛、鼻子、眉毛、嘴巴等属性定义成一个分类器，如果检测到一个模型符合定义人脸的所有属性，那么就认为它是一个人脸
                        # 分类器是.xml文件，这里不想随意移动文件夹，就给出了绝对路径
        '''haar特征
        Haar特征是一种反映图像的灰度变化的，像素分模块求差值的一种特征。
        它分为三类：边缘特征、线性特征、中心特征和对角线特征。
        用黑白两种矩形框组合成特征模板，在特征模板内用黑色矩形像素和减去白色矩形像素和来表示这个模版的特征值。
        https://senitco.github.io/2017/06/15/image-feature-haar/#:~:text=Haar%E7%89%B9%E5%BE%81%E6%98%AF%E4%B8%80,%E8%BE%83%E4%B8%BA%E7%BB%8F%E5%85%B8%E7%9A%84%E7%AE%97%E6%B3%95%E3%80%82
        '''
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # cv2.COLOR_BGR2gray_frame
                            # frame上面说过。       将rgb变为灰度图
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
            # 将上面生成的灰度图传入，scaleFactor表示在前后两次相继的扫描中，搜索窗口的比例系数,minNeighbos表示构成检测目标的相邻矩形的最小个数
            # 返回：用vector表示各个人脸特征（4维）的集合
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            # 在图像上绘制矩形。图像，起点，终点，边框颜色（上述是蓝色），边框粗细（上述是2px）
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                                                    # cv2.resize(输入mat数据, 输出图像尺寸)
                    # np.expand_dims() 拓展维度，上面是将最后加了一个维度，最前加了一个维度
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            '''cv2.putText()
            image: 图片，这里传入每一帧的rgb图
            text: 要添加的文字
            org: 文字添加在图片的位置
            fontFace: 字体类型
            fontScale: 字体大小
            color: 字体颜色
            thickness: 字体粗细
            '''
        cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
        # cv2.imshow(window_name, image)，在窗口中显示图像      # cv2.resize()还有一个参数是interpolation，插值方式，用于缩放图像后重新计算像素
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''
        cv2.waitKey(1)在有按键按下的时候返回按键的ASCII值，否则返回-1
        & 0xFF的按位与操作只取cv2.waitKey(1)返回值最后八位，因为有些系统cv2.waitKey(1)的返回值不止八位
        ord('q')表示q的ASCII值
        '''

    cap.release()  # 停止捕获视频
    cv2.destroyAllWindows()  # 关闭所有显示窗口

video_pre()
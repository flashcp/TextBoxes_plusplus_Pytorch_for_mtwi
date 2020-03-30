# TextBoxes_plusplus_for_mtwi
  TextBoxes++在天池mtwi比赛上的应用
### 一、简介  
  TextBoxes++ 在pytorch上的复现，所有方法基于ssd.pytorch进行修改。  <br>
### 二、预训练模型  
  [vgg16_reducedfc.pth](https://pan.baidu.com/s/1JAaKKiQ6laR0MwdgWKhpgg)    <br>
  百度云提取码:rtcz
### 三、环境
 python 3.7  <br>
 torch                1.4.0+cu92  <br>
 torchvision          0.5.0+cu92  <br>
 Google colab默认环境可正常运行  <br>
### 四、使用方法
 1.train:  <br>
 ``!python train.py --dataset mtwi384 --dataset_root /content/gdrive/mtwi_2018_train --batch_size 8 --lr 1e-4``  <br>
 2.test:   <br>
 ``!python test_mtwi.py --trained_model weights/ssd768_mtwi_90000.pth --save_folder test/sample_task2 0.18 --mtwi_root /content/gdrive/mtwi_2018_task2_test``<br>
### 五、结果
 MTWI 2018 挑战赛中的结果Precision:0.629，Recall:0.365  <br>

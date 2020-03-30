import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append('E:\coding\deeplearning\TextBoxes_plusplus_tianchi')

import torch
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from data import MTWIDetectionTest
from ssd import build_ssd
from data import MTWI_CLASSES as labels
from matplotlib.path import Path
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
import time
import warnings
warnings.filterwarnings("ignore")


def run(img_id, Rectangle=False):
    image, image_name = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # View the sampled input image before transform
    plt.figure(figsize=(10,10))
    # plt.imshow(rgb_image)
    # plt.show()

    x = cv2.resize(rgb_image, (size, size)).astype(np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(6)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.3:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:].clamp(max=1, min=0) * scale).cpu().numpy()
            path_data = [
                (Path.MOVETO, (pt[4], pt[5])),
                (Path.CURVE4, (pt[6], pt[7])),
                (Path.CURVE4, (pt[8], pt[9])),
                (Path.CURVE4, (pt[10], pt[11])),
                (Path.MOVETO, (pt[4], pt[5])),
            ]
            codes, verts = zip(*path_data)
            path = Path(verts, codes)
            x, y = zip(*path.vertices)
            plt.plot(x, y, '-', linewidth=1, color='r')

            if Rectangle:
                coords_1 = (pt[0] - pt[2] / 2, pt[1] - pt[3] / 2), pt[2], pt[3]
                color = colors[i]
                # Rectangle(*coords):coords包含矩形左上角坐标和长宽
                currentAxis.add_patch(plt.Rectangle(*coords_1, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(pt[0]-pt[2]/2+1, pt[1]-pt[3]/2+1, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
            j += 1

    plt.savefig(image_name[:-3]+'.png')
    plt.show()
    end_t = time.time()
    print('time consuming', end_t-start_t)


if __name__ == '__main__':
    size = 384
    weight_path = 'E:\coding\deeplearning\TextBoxes_plusplus_tianchi\weights\ssd384_mtwi_60000.pth'
    mtwi_path = 'F:\chromedown\mtwi_2018_task2_test\icpr_mtwi_task2'
    start_t = time.time()
    net = build_ssd('test', size, 2)  # initialize SSD
    net.load_weights(weight_path)
    testset = MTWIDetectionTest(mtwi_path, None)
    rand_list = np.random.randint(0, 10000, 5)
    for i in rand_list:
        run(i)

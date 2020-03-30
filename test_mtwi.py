from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
# from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import MTWI_CLASSES as labelmap
from PIL import Image
# from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import MTWIAnnotationTransform, MTWIDetection, BaseTransform, MTWI_CLASSES
from data import MTWIDetectionTest
import torch.utils.data as data
from ssd import build_ssd
import cv2
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd384_mtwi_60000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='F:/chromedown/mtwi_2018_task2_test/icpr_mtwi_task2/output/sample_task2/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.2, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--mtwi_root', default='F:\chromedown\mtwi_2018_task2_test\icpr_mtwi_task2', help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img, image_name = testset.pull_image(i)
        filename = save_folder + image_name[:-3] + 'txt'
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(transform(rgb_image)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        # height, width, channels = img.shape
        scale = torch.Tensor(img.shape[1::-1]).repeat(6)
        for i in range(detections.size(1)):
            with open(filename, mode='a') as f:
                j = 0
                while detections[0, i, j, 0] >= thresh:
                    pt = (detections[0, i, j, 1:].clamp(max=1, min=0)*scale).cpu().numpy()
                    coords = pt[4], pt[5], pt[6], pt[7], pt[8], pt[9], pt[10], pt[11]
                    f.write(','.join(str(c) for c in coords) + '\n')
                    j += 1


def test_mtwi():
    # load net
    num_classes = len(MTWI_CLASSES) + 1 # +1 background
    net = build_ssd('test', 384, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = MTWIDetectionTest(args.mtwi_root, None)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size),
             thresh=args.visual_threshold)


if __name__ == '__main__':
    test_mtwi()

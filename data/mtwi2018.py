import os.path as osp
import os
import torch
import torch.utils.data as data
import cv2
import numpy as np


MTWI_CLASSES = (  # always index 0
    'text',)


class MTWIAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(MTWI_CLASSES, range(len(MTWI_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target:
            obj_list = obj.strip().split(',')
            if obj_list[8] == '###':
                continue
            name = 'text'
            bndbox = []
            for i, pt in enumerate(obj_list[:8]):
                cur_pt = float(pt)
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            p12 = (bndbox[0] - bndbox[2], bndbox[1] - bndbox[3])
            p23 = (bndbox[2] - bndbox[4], bndbox[3] - bndbox[5])
            if p12[0]*p23[1]-p12[1]*p23[0] < 0:
                bndbox[0:7:2] = bndbox[6::-2]
                bndbox[1:8:2] = bndbox[7::-2]
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, x3, y3, x4, y4, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class MTWIDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to mtwi_2018_train folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 transform=None, target_transform=MTWIAnnotationTransform(),
                 dataset_name='mtwi2018'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join(self.root, 'txt_train', '{}.txt')
        self._imgpath = osp.join(self.root, 'image_train', '{}.jpg')
        self.ids = list()
        for files in os.listdir(osp.join(self.root, 'txt_train')):
            self.ids.append(files.replace('.txt', ''))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        with open(self._annopath.format(img_id), 'r', encoding='utf-8') as files:
            target = files.readlines()
        # cv2读取jpg图片是BGR格式，需要转换通道顺序变为RGB
        img = cv2.imread(self._imgpath.format(img_id))
        try:
            height, width, channels = img.shape
        except:
            print(img_id)
            height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            # target in point-form
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :8], target[:, 8])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath.format(img_id), cv2.IMREAD_COLOR)

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        with open(self._annopath.format(img_id), 'r', encoding='utf-8') as files:
            target = files.readlines()
        gt = self.target_transform(target, 1, 1)
        return img_id, gt

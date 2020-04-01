# -*- coding: utf-8 -*-
import torch
from shapely.geometry import Polygon
import numpy as np

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax
def point_form_2(boxes):
    """ 将 prior_boxes 转换为按顺时针从左上到左下的坐标(x1,y1,x2,y2,x3,y3,x4,y4)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted x1,y1,x2,y2,x3,y3,x4,y4 form of boxes.
    """
    reboxes = torch.zeros((boxes.shape[0], 8))
    reboxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    reboxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    reboxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    reboxes[:, 3] = boxes[:, 1] - boxes[:, 3] / 2
    reboxes[:, 4] = boxes[:, 0] + boxes[:, 2] / 2
    reboxes[:, 5] = boxes[:, 1] + boxes[:, 3] / 2
    reboxes[:, 6] = boxes[:, 0] - boxes[:, 2] / 2
    reboxes[:, 7] = boxes[:, 1] + boxes[:, 3] / 2
    return reboxes

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    box_a_4 = torch.zeros(box_a.shape[0], 4)
    box_a_4[:, 0] = box_a[:, 0:7:2].min(dim=1)[0]
    box_a_4[:, 1] = box_a[:, 1:8:2].min(dim=1)[0]
    box_a_4[:, 2] = box_a[:, 0:7:2].max(dim=1)[0]
    box_a_4[:, 3] = box_a[:, 1:8:2].max(dim=1)[0]

    inter = intersect(box_a_4, box_b)
    area_a = ((box_a_4[:, 2]-box_a_4[:, 0]) *
              (box_a_4[:, 3]-box_a_4[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def intersect_ploy(box_a, box_b):
    """
    任意两个图形的相交面积的计算
    :param data1: 当前物体[(x1,y1), (x2, y2), .., ..]
    :param data2: 待比较的物体
    :return: 当前物体与待比较的物体的面积交集, box_a的面积
    """
    box_a = box_a.reshape(4, 2)
    box_b = box_b.reshape(4, 2)

    poly1 = Polygon(box_a).convex_hull      # Polygon：多边形对象
    poly2 = Polygon(box_b).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area, poly1.union(poly2).area


def jaccard_2(box_a, box_b, overlaps):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4], [num_objects,8] point-offset
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4], [num_objects,8] point-offset
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    # inter = intersect(box_a, box_b)
    A = box_a.size(0)
    B = box_b.size(0)

    box_a_index,  box_b_index = torch.where(overlaps>0)
    inter = torch.zeros(A, B)
    union = torch.zeros(A, B)
    inter_union = np.array([intersect_ploy(box_a[box_a_index[i]], box_b[box_b_index[i]]) for i in range(len(box_a_index))])
    for j in range(len(box_a_index)):
        inter[box_a_index[j], box_b_index[j]] = inter_union[j, 0]
        union[box_a_index[j], box_b_index[j]] = inter_union[j, 1]
    return inter / union  # [A,B]

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors]. point-form
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4]. center-form
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(truths, point_form(priors))

    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True) # (A, 1) 每一行的最大值，即单一真实框和预选框IOU最大的一个下标
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True) # (1, B) 每一列的最大值，即单一预选框和真实框IOU最大的一个下标
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4] [num_priors,8], point-form
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    pos = torch.where(conf>0)
    loc = encode(matches, priors, pos, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def define_orient(matched, horizontal_rectangle):
    '''
    确定多边形四个角的顺序，采用欧几里得距离，求最小的排列
    :param matched: [num_priors, 8], point-form
    :param horizontal_rectangle: [num_priors, 8], point-form
    :return:
    '''
    # 1234-2341-3412-4123
    distance_min_1 = torch.sum((matched[:, 0:2] - horizontal_rectangle[:, 0:2]) ** 2, dim=1) + \
                   torch.sum((matched[:, 2:4] - horizontal_rectangle[:, 2:4]) ** 2, dim=1) + \
                   torch.sum((matched[:, 4:6] - horizontal_rectangle[:, 4:6]) ** 2, dim=1) + \
                   torch.sum((matched[:, 6:8] - horizontal_rectangle[:, 6:8]) ** 2, dim=1) # (num_priors)

    distance_min_2 = torch.sum((matched[:, 2:4] - horizontal_rectangle[:, 0:2]) ** 2, dim=1) + \
                     torch.sum((matched[:, 4:6] - horizontal_rectangle[:, 2:4]) ** 2, dim=1) + \
                     torch.sum((matched[:, 6:8] - horizontal_rectangle[:, 4:6]) ** 2, dim=1) + \
                     torch.sum((matched[:, 0:2] - horizontal_rectangle[:, 6:8]) ** 2, dim=1)  # (num_priors)

    distance_min_3 = torch.sum((matched[:, 4:6] - horizontal_rectangle[:, 0:2]) ** 2, dim=1) + \
                     torch.sum((matched[:, 6:8] - horizontal_rectangle[:, 2:4]) ** 2, dim=1) + \
                     torch.sum((matched[:, 0:2] - horizontal_rectangle[:, 4:6]) ** 2, dim=1) + \
                     torch.sum((matched[:, 2:4] - horizontal_rectangle[:, 6:8]) ** 2, dim=1)  # (num_priors)

    distance_min_4 = torch.sum((matched[:, 6:8] - horizontal_rectangle[:, 0:2]) ** 2, dim=1) + \
                     torch.sum((matched[:, 0:2] - horizontal_rectangle[:, 2:4]) ** 2, dim=1) + \
                     torch.sum((matched[:, 2:4] - horizontal_rectangle[:, 4:6]) ** 2, dim=1) + \
                     torch.sum((matched[:, 4:6] - horizontal_rectangle[:, 6:8]) ** 2, dim=1)  # (num_priors)

    origin = torch.cat((distance_min_1.unsqueeze(1),
                        distance_min_2.unsqueeze(1),
                        distance_min_3.unsqueeze(1),
                        distance_min_4.unsqueeze(1)), 1) # (num_priors, 4)

    origin_indices = origin.min(dim=1).indices  # (num_priors)
    for i, box in enumerate(matched):
        if origin_indices[i] == 1:
            index = [2, 3, 4, 5, 6, 7, 0, 1]
            matched[i] = matched[i][index]
        if origin_indices[i] == 2:
            index = [4, 5, 6, 7, 0, 1, 2, 3]
            matched[i] = matched[i][index]
        if origin_indices[i] == 3:
            index = [6, 7, 0, 1, 2, 3, 4, 5]
            matched[i] = matched[i][index]

    return matched


def encode(matched, priors, pos, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].[num_priors, 8], point-form
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    # 多边形文本框外接水平矩形
    x0 = matched[:, 0:7:2].min(-1, keepdim=True).values
    y0 = matched[:, 1:8:2].min(-1, keepdim=True).values
    x1 = matched[:, 0:7:2].max(-1, keepdim=True).values
    y1 = matched[:, 1:8:2].max(-1, keepdim=True).values
    horizontal_rectangle = torch.cat((x0, y0, x1, y0, x1, y1, x0, y1), 1)
    matched[pos] = define_orient(matched[pos], horizontal_rectangle[pos])

    w = x1 - x0
    h = y1 - y0

    delta_x = ((x1+x0)/2 - priors[:, 0].unsqueeze(1))/priors[:, 2].unsqueeze(1)
    delta_y = ((y1+y0)/2 - priors[:, 1].unsqueeze(1))/priors[:, 3].unsqueeze(1)
    delta_w = torch.log(w/priors[:, 2].unsqueeze(1))
    delta_h = torch.log(h/priors[:, 3].unsqueeze(1))

    # 预选框
    priors_8 = point_form_2(priors) # center-offset form to 8point-form
    matched[:, 0:7:2] = (matched[:, 0:7:2] - priors_8[:, 0:7:2]) / priors[:, 2].unsqueeze(1)
    matched[:, 1:8:2] = (matched[:, 1:8:2] - priors_8[:, 1:8:2]) / priors[:, 3].unsqueeze(1)

    target_delta = torch.cat((delta_x, delta_y, delta_w, delta_h), 1)
    target_delta = torch.cat((target_delta, matched), 1)

    # return target for smooth_l1_loss
    return target_delta  # [num_priors,12]

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions. center-offset form+point-form
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * priors[:, 2:],
        priors[:, 2:4] * torch.exp(loc[:, 2:4])), 1)

    priors_8 = point_form_2(priors)
    loc[:, 4:11:2] = priors_8[:, 0:7:2] + priors[:, 2].unsqueeze(1) * loc[:, 4:11:2]
    loc[:, 5:12:2] = priors_8[:, 1:8:2] + priors[:, 3].unsqueeze(1) * loc[:, 5:12:2]

    boxes = torch.cat((boxes, loc[:, 4:12]), 1)
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=(0.5, 0.2), top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.方框+四边形nms
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].point-form,[num_priors,12]
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    # First NMS with a relatively high IOU threshold (e.g. 0.5) on the minimum horizontal rectangles
    keep = scores.new(scores.size(0)).zero_().long()
    # 返回元素数目
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0:7:2].min(-1).values
    y1 = boxes[:, 1:8:2].min(-1).values
    x2 = boxes[:, 0:7:2].max(-1).values
    y2 = boxes[:, 1:8:2].max(-1).values

    area_f = torch.mul(x2 - x1, y2 - y1)
    # 预选框排名，升序
    v, idx_f = scores.sort(0)  # sort in ascending order
    idx_f = idx_f[-top_k:]
    # boxes.new()该Tensor的type和device都和原有Tensor一致，且无内容
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx_f.numel() > 0:
        i = idx_f[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx_f.size(0) == 1:
            break
        idx_f = idx_f[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx_f, out=xx1)
        torch.index_select(y1, 0, idx_f, out=yy1)
        torch.index_select(x2, 0, idx_f, out=xx2)
        torch.index_select(y2, 0, idx_f, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area_f, 0, idx_f)  # load remaining areas)
        union = (rem_areas - inter) + area_f[i]
        IoU_f = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        # input<=other，坐标位置为1，否则为0
        idx_f = idx_f[IoU_f.le(overlap[0])]

    # sencod NMS on quadrilaterals or rotated rectangles is applied to a few remaining
    # candidate boxes with a lower IOU threshold (e.g. 0.2).
    x1 = boxes[:, 4]
    y1 = boxes[:, 5]
    x2 = boxes[:, 6]
    y2 = boxes[:, 7]
    x3 = boxes[:, 8]
    y3 = boxes[:, 9]
    x4 = boxes[:, 10]
    y4 = boxes[:, 11]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    xx3 = boxes.new()
    yy3 = boxes.new()
    xx4 = boxes.new()
    yy4 = boxes.new()

    count_s = 0
    keep_s = scores.new(scores.size(0)).zero_().long()
    idx_s = keep[:count]
    while idx_s.numel() > 0:
        i = idx_s[0]
        keep_s[count_s] = i
        count_s += 1
        if idx_s.size(0) == 1:
            break
        max_box = boxes[i]
        idx_s = idx_s[1:]
        torch.index_select(x1, 0, idx_s, out=xx1)
        torch.index_select(y1, 0, idx_s, out=yy1)
        torch.index_select(x2, 0, idx_s, out=xx2)
        torch.index_select(y2, 0, idx_s, out=yy2)
        torch.index_select(x3, 0, idx_s, out=xx3)
        torch.index_select(y3, 0, idx_s, out=yy3)
        torch.index_select(x4, 0, idx_s, out=xx4)
        torch.index_select(y4, 0, idx_s, out=yy4)

        inter_union = np.array([intersect_ploy(max_box[4:12], np.array([xx1[i], yy1[i],
                                                         xx2[i], yy2[i],
                                                         xx3[i], yy3[i],
                                                         xx4[i], yy4[i]])) for i in range(xx1.shape[0])])
        inter, union = inter_union[:, 0], inter_union[:, 1]
        IoU_s = torch.tensor(inter / union)
        idx_s = idx_s[IoU_s.le(overlap[1])]
    return keep_s, count_s


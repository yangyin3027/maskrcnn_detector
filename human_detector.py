import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import random

import torch
import torchvision

from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

def show_bboxes(ax, boxes, labels, colors):
    for i in range(len(boxes)):
        # plot bboxes
        bbox = boxes[i]
        xy = (bbox[0], bbox[1])
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = mlp.patches.Rectangle(xy, w, h, lw=1, 
                            facecolor='none', edgecolor=colors[i])
        ax.add_patch(rect)

        text = labels[i]

        bbox_props = dict(boxstyle='square', facecolor=colors[i],
                          edgecolor='none', pad=0)
        ax.text(xy[0], xy[1], text,style='italic',
                              color='k',horizontalalignment='left',
                              verticalalignment='bottom',
                              bbox=bbox_props)

def show_masks(ax, masks, colors, contour=True):
    # make colors a customized cmap
    cmap = mlp.colors.LinearSegmentedColormap.from_list('mask_cmap', colors)

    for i in range(len(masks)):
        mask = masks[i]
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        masked = np.ma.masked_where(mask==False, mask)
        ax.imshow(masked, cmap, alpha=.3)
        # draw contour or mask
        if contour:
            h, w = mask.shape
            x, y = np.arange(w), np.arange(h)
            X,Y = np.meshgrid(x, y)
            ax.contour(X, Y, mask, 2, colors = colors[i])            

class MaskRCNN:
    def __init__(self):
        self.weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = maskrcnn_resnet50_fpn(weights = self.weights)

        # Set model to eval mode
        self.model.eval()

        self.preprocess = self.weights.transforms()
        self.cls_table = self.weights.meta['categories']
    
    def __call__(self, x, score_threshold=0.8, mask_threshold=.5):
        if not isinstance(x, torch.Tensor):
            x = to_tensor(x)
        assert x.ndim >= 2, 'At least 2-dimensional tensor needed'
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        
        x = self.preprocess(x)
        
        pred = self.model(x)[0]
        # detach from the computation graph
        pred = {k:v.data for k, v in pred.items()}
        valid_idx = pred['scores'] > score_threshold
        pred = {key: out[valid_idx] for key, out in pred.items()}

        # create colors for each class_name
        colors =['c', 'y', 'm', 'r', 'b', 'g']
        # shuffle colors
        random.shuffle(colors)
        # designate each class_name an unique color
        colors =[colors[i%len(colors)] for i in pred['labels']]

        pred['labels'] = [self.cls_table[label] for label in pred['labels']]
        pred['masks'] = pred['masks'] > mask_threshold

        self.show_detection(x, pred, colors)

        return pred
  
    def show_detection(self, x, detection, colors):
        '''
        Plot bboxes, masks along with annotation of cls_name and prob
        Args:
            x (torch.Tensor): [batch, c, w, h]
            detection (dict): keys as 'boxes', 'labels', 'scores', and 'mask'
        Return:
            matplotlib.figure object
        '''
        # transform normalized img tensor to numpy
        img = np.asarray(to_pil_image(x.squeeze(0)))

        fig, ax = plt.subplots(frameon=False)
        
        
        fig.set(tight_layout=True)
        
        ax.imshow(img)
        ax.set_axis_off()
        ax.margins(0, 0)

        scores = [f'{round(x.numpy()*100)}%' for x in detection['scores']]
        labels = [f'{l}: {s}' for l, s in zip(detection['labels'], scores)]

        show_bboxes(ax, detection['boxes'], labels, colors)
        show_masks(ax, detection['masks'], colors)

        plt.show()

        return fig

def run_example(img_file, threshold):
    img = read_image(img_file)
    maskrcnn = MaskRCNN()
    pred = maskrcnn(img, threshold)
    return pred

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float,
                        default=.8)
    parser.add_argument('-i', '--img', default='./images/demo.jpg')

    args = parser.parse_args()

    run_example(args.img, args.threshold)
        

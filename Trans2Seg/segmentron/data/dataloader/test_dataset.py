import os
import sys
import cv2
import math
import random

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from .entity import params, ObjectCategory
from .anno_read import anno_read
# coco dataset (coco_train, mode, imgids)

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

class HOIDataset(Dataset):
    NUM_CLASS = 21
    def __init__(self, insize=512, mode='train', task='instance_segmentation', n_samples=None):
        myanno_read=anno_read()
        self.annos = myanno_read
        assert mode in ['train', 'val', 'eval'], 'Data loading mode is invalid.'
        self.mode = mode
        self.task = task
        self.train_list = []
        self.test_list = []

        # according to the task you choose
        # load image lists to train or test
        if task == 'instance_segmentation':
            if mode == 'train':
                f = open(params['coco_dir']+'/'+'train.txt', encoding='utf-8')
                for line in f:
                    self.train_list.append(line.strip())
                self.train_list = map(int, self.train_list)
                self.train_list = sorted(set(self.train_list))
                self.imgIds = self.train_list
                
                self.seg_imgIds = myanno_read.getSegmentationIds(self.imgIds)
                self.seg_imgIds = sorted(set(self.seg_imgIds))
                
            else:
                f = open(params['coco_dir']+'/'+'test.txt', encoding='utf-8')
                for line in f:
                    self.test_list.append(line.strip())
                self.test_list = map(int, self.test_list)
                self.test_list = sorted(set(self.test_list))
                self.imgIds = self.test_list
                # obtain ids from image with HOIs, filter image without HOIs
                self.seg_imgIds = myanno_read.getSegmentationIds(self.imgIds)
                self.seg_imgIds = sorted(set(self.seg_imgIds))


        # get path information of each frame
        self.frameIds, self.videoIds, self.date = myanno_read.getFrameIds(self.imgIds)

        if self.mode in ['val', 'eval'] and n_samples is not None:
            self.imgIds = random.sample(self.imgIds, n_samples)
        print('{} images: {}'.format(mode, len(self)))
        self.insize = insize

    def __len__(self):
        return len(self.imgIds)

    def get_hoi_annotation(self, ind=None, img_id=None):
        ''' get HOIs with their ID from imageID '''
        ''' get poses with their ID from imageID '''
        ''' get objects from imageID '''
        ''' load the image according to the directory information, if not return None'''

        if ind is not None:
            img_id = self.imgIds[ind]
            hoi_img_id = self.hoi_imgIds[ind]
            

        objects_for_img = self.annos.loadObjects(img_ids=[img_id])

        # if too few keypoints

        obj_anns = objects_for_img

        # get path information of each frame and load the image
        videoId = self.videoIds[ind]
        frameId = self.frameIds[ind]
        date = self.date[ind]
        if self.mode == 'train':
            # img_path = os.path.join(params['coco_dir'], '0802', videoId, '{:012d}.jpg'.format(frameId))
            img_path = params['coco_dir'] + '/' + 'img' + '/' + 'collaboration' + '/' + date + '/' + videoId + '/' + '{:08d}.jpg'.format(frameId)
            print(img_path)
        else:
            img_path = os.path.join(params['coco_dir'], 'val2017', '{:08d}.jpg'.format(img_id))
        img = cv2.imread(img_path)

        ignore_mask = np.zeros(img.shape[:2], 'bool')

        if self.mode == 'eval':
            return img, img_id, objects_for_img, ignore_mask
        return img, img_id, obj_anns, ignore_mask


    def resize_data(self, img, ignore_mask, poses, obj_bboxs, human_bbox, shape):
        """resize img, poses and bbox"""

        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8), shape).astype('bool')
        if len(poses)>0:
            poses[:, :, :2] = (poses[:, :, :2] * np.array(shape) / np.array((img_w, img_h)))
        objs = (obj_bboxs[:, :2] * np.array(shape) / np.array((img_w, img_h)))
        objs = np.hstack((objs, (obj_bboxs[:, 2:] * np.array(shape) / np.array((img_w, img_h)))))
        humans = (human_bbox[:, :2] * np.array(shape) / np.array((img_w, img_h)))
        humans = np.hstack((humans, (human_bbox[:, 2:] * np.array(shape) / np.array((img_w, img_h)))))
        return resized_img, ignore_mask, poses, objs, humans

    # return shape: (height, width)
    def generate_gaussian_heatmap(self, shape, joint, sigma):
        """ transform pose labels to heatmap """

        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        # generate the Gaussian distribution
        return gaussian_heatmap


    # return shape: (2, height, width)



    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data

    def parse_objects_annotation(self, obj_anns):
        """load bbox of objects and humans from annos"""

        count = 0

        for obj in obj_anns:
            obj_bbox = np.array(obj['object_bbox']).reshape(-1, 4)
            human_bbox = np.array(obj['human_bbox']).reshape(-1, 4)
            label = obj['object_category']
            count += 1
            if count == 1:
                objs = obj_bbox
                labels = label
                humans = human_bbox
            else:
                objs = np.vstack((objs, obj_bbox))
                humans = np.vstack((humans, human_bbox))
                labels.append(label)
        return objs, humans, labels



    def get_seg_annotation(self, ind=None, img_id=None, task = 'instance_segmentation'):
        ''' get segmentation from imageID '''
        ''' load the image according to the directory information '''

        if ind is not None:
            img_id = self.imgIds[ind]
            seg_img_id = self.seg_imgIds[ind]
        #对应ind的img_id和seg_img_id
        print(img_id)
        print(seg_img_id)
        segs_for_img = self.annos.loadSegs(ids=[seg_img_id])
        print(segs_for_img)
        segs_anns = segs_for_img
        
        # get path information of each frame and load the image
        videoId = self.videoIds[ind]
        frameId = self.frameIds[ind]
        date = self.date[ind]
        if self.mode == 'train':
            # img_path = os.path.join(params['coco_dir'], '0802', videoId, '{:012d}.jpg'.format(frameId))
            img_path = params['coco_dir'] + '/' + 'img' + '/' + task + '/' + date + '/' + videoId + '/' + '{:08d}.jpg'.format(frameId)
            #print(img_path)
        else:
            img_path = os.path.join(params['coco_dir'], 'val2017', '{:08d}.jpg'.format(img_id))
        img = cv2.imread(img_path)

        ignore_mask = np.zeros(img.shape[:2], 'bool')

        if self.mode == 'eval':
            return img, img_id, segs_anns
        return img, img_id, segs_anns

    def load_seg_annotation(self, segs_anns, task):
        ''' get object_polygon label from annos '''
        ''' get object_category label from annos '''
        ''' get solution_category label from annos '''

        for seg in segs_anns:
            obj_id= seg['object_id']
            label = seg['object_category']
            object_polygon = seg['object_polygon']

            solution_id = []
            label_id = []
            for i in label:
                id = ObjectCategory[i]
                label_id.append(id)


        return obj_id, label_id, object_polygon, solution_id

    def resize_data_seg(self, img, object_polygon, shape):
        """resize img, and polygon"""

        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)

        resized_polygon = []
        count = 0
        if len(object_polygon)>0:
            for obj in object_polygon:
                if count == 0:
                    np_obj = np.array(obj).reshape(-1, 2) * np.array(shape) / np.array((img_w, img_h))
                    np_objs = np_obj.flatten()
                    np_objs = [np_objs.tolist()]
                    count += 1
                else:
                    np_obj = np.array(obj).reshape(-1, 2) * np.array(shape) / np.array((img_w, img_h))
                    np_obj = np_obj.flatten()
                    np_obj = [np_obj.tolist()]
                    # np_objs = np.vstack((np_objs, np_obj))
                    np_objs = np_objs + np_obj
                    count += 1

            resized_polygon = np_objs

        return resized_img, resized_polygon


    def annToRLE(self, polygon, h, w):
        """
        transform polygon to feature map mask
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = [polygon]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = polygon
        return rle

    def annToMask(self, polygon, h, w):
        """
        transform polygon to feature map mask
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(polygon, h, w)
        m = maskUtils.decode(rle)
        return m

    def parse_seg_annotation(self, resized_polygon, img):
        """transform polygon to feature map mask"""

        img_h, img_w, _ = img.shape

        masks = [self.annToMask(polygon, img_h, img_w).reshape(-1) for polygon in resized_polygon]
        masks = np.vstack(masks)
        masks = masks.reshape(-1, img_h, img_w)

        return masks
        
    def __getitem__(self, i):
        if self.task == 'instance_segmentation':
            img, img_id, segs_anns = self.get_seg_annotation(ind=i, task=self.task)

            obj_ids, labels, object_polygon,_ = self.load_seg_annotation(segs_anns, task=self.task)
            resized_img, resized_polygon = self.resize_data_seg(img, object_polygon, shape=(self.insize, self.insize))
            marks = self.parse_seg_annotation(resized_polygon, resized_img)

            marks = torch.tensor(marks)
            labels = torch.tensor(labels)
            obj_ids = torch.tensor(obj_ids)

            return resized_img, marks, labels, obj_ids
    @property
    def classes(self):
        """Category names."""
        return ('pipette',
    'PCR_tube',
    'tube',
    'waste_box',
    'vial',
    'measuring_flask',
    'beaker',
    'wash_bottle',
    'water_bottle',
    'erlenmeyer_flask',
    'culture_plate',
    'spoon',
    'electronic_scale',
    'LB_solution',
    'stopwatch',
    'D_sorbitol',
    'solution_P1',
    'plastic_bottle',
    'agarose',
    'cell_spreader')
if __name__ == "__main__":#裁剪为512，512
    myanno_read=anno_read()
    train_dataset=HOIDataset(myanno_read,512)
   
    
    








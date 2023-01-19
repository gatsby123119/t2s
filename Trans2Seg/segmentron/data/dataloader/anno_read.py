import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
import os
from collections import defaultdict
import sys
from .entity import params

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class anno_read:
    def __init__(self, annotation_file='./dataset/coco2017/Instance_segmentation.json', mode='train'):
        self.dataset, self.imgs,self.objects, self.segs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()


    def createIndex(self):
        # create index of img, pose, hois, objects and segmentation
        print('creating index...')
        imgs, objects = {}, {}
        #catToImgs =  defaultdict(list)

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['image_id']] = img

        if 'objects' in self.dataset:
            for obj in self.dataset['objects']:
                objects[obj['image_id']] = obj

        if 'segmentation' in self.dataset:
            for seg in self.dataset['segmentation']:
                self.segs[seg['image_id']] = seg


        print('index created!')

        # create class members
        self.imgs = imgs
        self.objects = objects
        # self.catToImgs = catToImgs

    def getSegmentationIds(self, img_list=[], is_tracking=0):
        """
        obtian seg_ids from given train/test list
        """
        seg_id = []
        for seg in self.dataset['segmentation']:
            if seg['image_id'] in img_list:
                seg_id.append(seg['image_id'])

        return seg_id


    def getFrameIds(self, img_id):
        """
        the image location from imageid
        the directory is composed of date, frame_ids, video_ids
        """
        frame_ids, video_ids, date = [], [], []
        for img in self.dataset['images']:
            if img['image_id'] in img_id:
                frame_ids.append(img['frame_id'])
                video_ids.append(img['video_id'])
                date.append(img['date'])

        return frame_ids, video_ids, date


    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[0][catId])
                else:
                    ids &= set(self.catToImgs[0][catId])
        return list(ids)

    # def getHoiIds(self, hoi_imgIds=[]):
    #     """
    #     get HOIids from the given img_id
    #     """
    #     imgIds = hoi_imgIds if _isArrayLike(hoi_imgIds) else [hoi_imgIds]
    #
    #     if len(imgIds) == 0:
    #         hois = self.dataset['hoi']
    #     else:
    #         if not len(imgIds) == 0:
    #             lists = [self.hoiToImgs[imgId] for imgId in imgIds if imgId in self.hoiToImgs]
    #             hois = list(itertools.chain.from_iterable(lists))
    #         else:
    #             hois = self.dataset['hoi']
    #
    #     ids = [hoi['id'] for hoi in hois]
    #
    #     return ids


    def loadObjects(self, img_ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(img_ids):
            return [self.objects[id] for id in img_ids]
        elif type(img_ids) == int:
            return [self.objects[id] for id in img_ids]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]


    def loadSegs(self, ids=[]):
        """
        load segmentation annos from the given img_id
        """
        if _isArrayLike(ids):
            return [self.segs[id] for id in ids]
        elif type(ids) == int:
            return [self.segs[id] for id in ids]


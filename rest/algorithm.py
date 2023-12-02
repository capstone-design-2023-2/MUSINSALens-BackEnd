# -*- coding: utf-8 -*-

from __future__ import print_function

from skimage.filters import gabor_kernel
from skimage.feature import hog

from skimage import color
from scipy import ndimage as ndi

import configparser
import pymysql
import sys
import os

import numpy as np
import pandas as pd
import json

import multiprocessing
import itertools

import pickle
from scipy import spatial
import cv2
import math

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG

#cache
categories = ['long_sleeve_dress', 'long_sleeve_outwear', 'long_sleeve_top', 'short_sleeve_dress', 'short_sleeve_outwear',
             'short_sleeve_top', 'shorts', 'skirt', 'sling_dress', 'sling', 'trousers', 'vest_dress', 'vest']


os.makedirs('cache', exist_ok=True)

for category in categories:
    category_dir = category + ' cache'
    subfolder_path = os.path.join('cache', category_dir)
    os.makedirs(subfolder_path, exist_ok=True)

class Gabor(object):

    gabor_theta     = 4
    gabor_frequency = (0.1, 0.5, 0.8)
    gabor_sigma     = (1, 3, 5)
    gabor_bandwidth = (0.3, 0.7, 1)

    gabor_n_slice  = 2
    gabor_depth = 1

    def make_gabor_kernel(theta=4 , frequency=(0.1, 0.5, 0.8), sigma=(1, 3, 5), bandwidth=(0.3, 0.7, 1)):
        kernels = []
        for t in range(theta):
            t = t / float(theta) * np.pi
            for f in frequency:
                if sigma:
                    for s in sigma:
                        kernel = gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s)
                        kernels.append(kernel)
                if bandwidth:
                    for b in bandwidth:
                        kernel = gabor_kernel(f, theta=t, bandwidth=b)
                        kernels.append(kernel)
        return kernels
    
    gabor_kernels = make_gabor_kernel()
    
    if gabor_sigma and not gabor_bandwidth:
        assert len(gabor_kernels) == gabor_theta * len(gabor_frequency) * len(gabor_sigma), "kernel nums error in make_gabor_kernel()"
    elif not gabor_sigma and gabor_bandwidth:
        assert len(gabor_kernels) == gabor_theta * len(gabor_frequency) * len(gabor_bandwidth), "kernel nums error in make_gabor_kernel()"
    elif gabor_sigma and gabor_bandwidth:
        assert len(gabor_kernels) == gabor_theta * len(gabor_frequency) * (len(gabor_sigma) + len(gabor_bandwidth)), "kernel nums error in make_gabor_kernel()"
    elif not gabor_sigma and not gabor_bandwidth:
        assert len(gabor_kernels) == gabor_theta * len(gabor_frequency), "kernel nums error in make_gabor_kernel()"

    def gabor_histogram(self, img, normalize=True):
        height, width, channel = img.shape

        hist = self._gabor(img, kernels=self.gabor_kernels)
  
        if normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _feats(self, image, kernel):
        feats = np.zeros(2, dtype=np.double)
        filtered = ndi.convolve(image, np.real(kernel), mode='wrap')
        feats[0] = filtered.mean()
        feats[1] = filtered.var()
        return feats

    def _power(self, image, kernel):
        image = (image - image.mean()) / image.std()  # Normalize images for better comparison.
        f_img = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                        ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
        feats = np.zeros(2, dtype=np.double)
        feats[0] = f_img.mean()
        feats[1] = f_img.var()
        return feats


    def _gabor(self, image, kernels=make_gabor_kernel(), normalize=True):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        img = color.rgb2gray(image)

        results = []
        feat_fn = self._power
        for kernel in kernels:
            results.append(pool.apply_async(self._worker, (img, kernel, feat_fn)))
        pool.close()
        pool.join()

        hist = np.array([res.get() for res in results])

        if normalize:
            hist = hist / np.sum(hist, axis=0)

        return hist.T.flatten()
    
    def _worker(self, img, kernel, feat_fn):
        try:
            ret = feat_fn(img, kernel)
        except:
            print("return zero")
            ret = np.zeros(2)
        return ret
   
    def make_samples(self, data, category, verbose=True):
        sample_cache = 'gabor'

        temp = os.path.join(os.getcwd(), 'cache')
        cache_dir = category + ' cache'

        data_img = data['img'].tolist()
        db_img = data['img'][1:].tolist()

        try:
            samples = pickle.load(open(os.path.join(temp, cache_dir, sample_cache), "rb", True))
            samples_img = [sample['img'] for sample in samples]
        
        except Exception as e: #nothing in cache
            print(e)
            samples = []
            samples_img = []

        fin_sample = []
        
        for img in data_img:
            if img in samples_img:
                sample = next((sample for sample in samples if sample['img'] == img), None)
                sample['hist'] /= np.sum(sample['hist'])  # normalize
                fin_sample.append(sample)
            else:
                for item in data.itertuples():
                    if item.img == img:
                        d = item
                        break

                d_img, d_category = getattr(d, "img"), getattr(d, "category")
                img = cv2.imread(d_img)
                d_hist = self.gabor_histogram(img)

                fin_sample.append({
                            'img':  d_img, 
                            'hist': d_hist
                        })
        sample_ca = [item for item in fin_sample if item['img'] != data['img'][0]] #db data in cache
        pickle.dump(sample_ca, open(os.path.join(temp, cache_dir, sample_cache), "wb", True))
        return fin_sample

class Color(object):

    def color_histogram(self, img, normalize = True):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        if normalize:
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        return hist

    def make_samples(self, data, category, verbose=True):
        sample_cache = 'color'

        temp = os.path.join(os.getcwd(), 'cache')
        cache_dir = category + ' cache'

        data_img = data['img'].tolist()
        db_img = data['img'][1:].tolist()

        try:
            samples = pickle.load(open(os.path.join(temp, cache_dir, sample_cache), "rb", True))
            samples_img = [sample['img'] for sample in samples]

        except Exception as e: #nothing in cache
            print(e)
            samples = []
            samples_img = []

        fin_sample = []

        for img in data_img:
            if img in samples_img:
                sample = next((sample for sample in samples if sample['img'] == img), None)
                sample['hist'] /= np.sum(sample['hist'])  # normalize
                fin_sample.append(sample)
            else:
                for item in data.itertuples():
                    if item.img == img:
                        d = item
                        break
                d_img, d_category = getattr(d, "img"), getattr(d, "category")

                img = cv2.imread(d_img)
                d_hist = self.color_histogram(img)

                fin_sample.append({
                                    'img':  d_img, 
                                    'hist': d_hist
                                })
            
        sample_ca = [item for item in fin_sample if item['img'] != data['img'][0]] #db data in cache

        pickle.dump(sample_ca, open(os.path.join(temp, cache_dir, sample_cache), "wb", True))
                
        return fin_sample

class HistogramOfGradients(object):

    def hog_histogram(self, img, n_bin=10, n_slice=6, normalize=True):
        height, width, channel = img.shape
    
        hist = np.zeros((n_slice, n_slice, n_bin))
        h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
        w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
        
        for hs in range(len(h_silce)-1):
            for ws in range(len(w_slice)-1):
                img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
                hist[hs][ws] = self._HOG(img_r, n_bin)
        
        if normalize:
            hist /= np.sum(hist)
    
        return hist.flatten()

    def _HOG(self, img, n_bin, normalize=True):
        image = color.rgb2gray(img)
        fd = hog(image, orientations=8, pixels_per_cell=(2,2), cells_per_block=(1,1))
        bins = np.linspace(0, np.max(fd), n_bin+1, endpoint=True)
        hist, _ = np.histogram(fd, bins=bins)
    
        if normalize:
            hist = np.array(hist) / np.sum(hist)
    
        return hist

    def make_samples(self, data, category, verbose=True):
        print("==================hog==================")
        sample_cache = "hog"

        temp = os.path.join(os.getcwd(), 'cache')
        cache_dir = category + ' cache'

        data_img = data['img'].tolist()
        db_img = data['img'][1:].tolist()

        try:
            samples = pickle.load(open(os.path.join(temp, cache_dir, sample_cache), "rb", True))
            samples_img = [sample['img'] for sample in samples]

        except Exception as e: #nothing in cache
            print(e)
            samples = []
            samples_img = []
        fin_sample = []
        for img in data_img:
            if img in samples_img:
                sample = next((sample for sample in samples if sample['img'] == img), None)
                sample['hist'] /= np.sum(sample['hist'])  # normalize
                fin_sample.append(sample)
            else:
                for item in data.itertuples():
                    if item.img == img:
                        d = item
                        break

                d_img, d_category = getattr(d, "img"), getattr(d, "category")
                img = cv2.imread(d_img)
                d_hist = self.hog_histogram(img)
                fin_sample.append({
                                'img':  d_img, 
                                'hist': d_hist
                            })
        
        sample_ca = [item for item in fin_sample if item['img'] != data['img'][0]] #db data in cache

        pickle.dump(sample_ca, open(os.path.join(temp, cache_dir, sample_cache), "wb", True))

        return fin_sample
    
# configs for histogram
VGG_model  = 'vgg19'  # model type
pick_layer = 'avg'    # extract feature of this layer
gabor_d_type     = 'd1'     # distance type

gabor_depth      = 3        # retrieved gabor_depth, set to None will count the ap for whole database

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=False, remove_fc=False, show_params=True):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]
        self.fc_ranges = ((0, 2), (2, 5), (5, 7))

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        x = self.features(x)

        avg_pool = torch.nn.AvgPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
        avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
        avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
        output['avg'] = avg

        x = x.view(x.size(0), -1)  # flatten()
        dims = x.size(1)
        if dims >= 25088:
            x = x[:, :25088]
            for idx in range(len(self.fc_ranges)):
                for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
                    x = self.classifier[layer](x)
                output["fc%d"%(idx+1)] = x
        else:
            w = self.classifier[0].weight[:, :dims]
            b = self.classifier[0].bias
            x = torch.matmul(x, w.t()) + b
            x = self.classifier[1](x)
            output["fc1"] = x
            for idx in range(1, len(self.fc_ranges)):
                for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
                    x = self.classifier[layer](x)
                output["fc%d"%(idx+1)] = x

        return output

ranges = {
  'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
  'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
  'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
  'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

cfg = {
  'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGGNetFeat(object):

    def make_samples(self, data, category, verbose=True):
        sample_cache = "vgg"
    
        temp = os.path.join(os.getcwd(), 'cache')
        cache_dir = category + ' cache'

        data_img = data['img'].tolist()
        db_img = data['img'][1:].tolist()

        try:
            samples = pickle.load(open(os.path.join(temp, cache_dir, sample_cache), "rb", True))
            samples_img = [sample['img'] for sample in samples]

        except Exception as e:
            print(e)
            samples = []
            samples_img = []
        fin_sample = []
        for img in data_img:
            if img in samples_img:
                sample = next((sample for sample in samples if sample['img'] == img), None)
                sample['hist'] /= np.sum(sample['hist'])  # normalize
                fin_sample.append(sample)
            else:
                for item in data.itertuples():
                    if item.img == img:
                        d = item
                        break
                
                vgg_model = VGGNet(requires_grad=False, model=VGG_model)
                vgg_model.eval()

                if use_gpu:
                    vgg_model = vgg_model.cuda()
        
                d_img, d_category = getattr(d, "img"), getattr(d, "category")
                img = cv2.imread(d_img)
                img = img[:, :, ::-1]  # switch to BGR
                img = np.transpose(img, (2, 0, 1)) / 255.
                img[0] -= means[0]  # reduce B's mean
                img[1] -= means[1]  # reduce G's mean
                img[2] -= means[2]  # reduce R's mean
                img = np.expand_dims(img, axis=0)

                try:
                    if use_gpu:
                        inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
                    else:
                        inputs = torch.autograd.Variable(torch.from_numpy(img).float())
                    d_hist = vgg_model(inputs)[pick_layer]
                    d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
                    d_hist /= np.sum(d_hist)  # normalize
                    fin_sample.append({
                                'img':  d_img, 
                                'hist': d_hist
                            })
                except:
                    pass

        sample_ca = [item for item in fin_sample if item['img'] != data['img'][0]] #db data in cache
        pickle.dump(sample_ca, open(os.path.join(temp, cache_dir, sample_cache), "wb", True))

        return fin_sample
    
def read_db_config(filename='config.ini', section='Database'): #config
    parser = configparser.ConfigParser()
    parser.read(filename)

    db_config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db_config[param[0]] = param[1]
    else:
        raise Exception(f"Section {section} not found in the {filename} file")

    return db_config

def cbir(json_string):
    query = json.loads(json_string)
    query = pd.DataFrame([query])
    print(query)

    query_category = query['category'][0]
    query_sort = query['sort_criteria'][0]

    print("======cbir")

    try:
        connect = pymysql.connect(db='musinsaDB', user='root', password='MUSINSA_Lens_1', host='43.201.176.97', port=3306, charset='utf8', cursorclass=pymysql.cursors.DictCursor)
        cur = connect.cursor()
    except pymysql.Error as e:
        print(f"Database connection error: {e}")
    
    sql = "SELECT image_path AS img, category FROM product WHERE category = %s"
    try:
        cur.execute(sql, query_category)
    except pymysql.Error as e:
        print(f"Database query execution error: {e}")


    db_result = cur.fetchall()
    data_category = pd.DataFrame(db_result)
    del query['sort_criteria']

    data_category = pd.concat([query, data_category], ignore_index = True)

    feature_extract_methods = {
        "color": Color,
        "hog": HistogramOfGradients,
        "gabor": Gabor,
        "vgg": VGGNetFeat,
    }

    selected_mthd = feature_extract_methods[query_sort]()

    print(data_category)
    print(query_category)
    samples = selected_mthd.make_samples(data_category, query_category)

    hists = [sample['hist'] for sample in samples]
    hists = [hist.astype(np.float32) for hist in hists]

    fin_result = pd.DataFrame({'img':[sample['img'] for sample in samples],
                               'hist': hists})
            
    print("fin result")
    print(fin_result)
    ret_list = []

    for i, histogram in enumerate(fin_result['hist']):
        ret = cv2.compareHist(fin_result.iloc[0]['hist'], histogram, cv2.HISTCMP_BHATTACHARYYA)
        ret_list.append(ret)

    # print(ret_list)
    fin_result['ret'] = ret_list #유사도 값
    fin_result = fin_result.sort_values(by='ret')
    fin_result = fin_result[fin_result['ret'] != 0] #쿼리 제거

    sorted_image_path = fin_result['img']
    placeholders = ', '.join(['%s'] * len(sorted_image_path))


    # print(placeholders)
    sql = f"SELECT name, brand, price, image_url, image_path FROM product WHERE image_path IN ({placeholders})"
    print("=====")

    try:
        cur.execute(sql, tuple(sorted_image_path))
    except pymysql.Error as e:
        print(f"Second database query execution error: {e}")

    result = cur.fetchall()

    son_result = json.dumps(result, ensure_ascii=False)
    
    cur.close()

    return son_result

def main(request_json_dict):
    try:    
        # request_json_dict = {
        #     "img" : str("D:/dev/vscode/musinsa/MUSINSALens-Algorithm/test-image2.png"),
        #     "category" : "long_sleeve_top",
        #     # "sort_criteria" : "vgg"
        #     # "sort_criteria" : "hog"
        #     "sort_criteria" : "gabor"
        #     # "sort_criteria" : "color"
            
        # }
        request_json_dict['img'] = request_json_dict.pop('path')
        request_json_dict.pop('score')
        print(request_json_dict)
        request_json = json.dumps(request_json_dict, ensure_ascii=False)
        result = cbir(request_json)
        # print(result)
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
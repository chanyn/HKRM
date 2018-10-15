from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.misc import imread
import os
import pickle
import json
import numpy as np
import sys
sys.path.append("../../coco/PythonAPI/")
from pycocotools.coco import COCO
from collections import defaultdict
import time
import itertools
import gc

def _any_in(source, target):
    for entry in source:
        if entry in target:
            return True
    return False


def _like_array(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def _get_cat_id(synset, cats):
    for idx in cats.keys():
        if cats[idx] == synset:
            return idx

def _remote_to_local(url, pre='VG'):
    """
    convert remote image url to local file name

    :param url: remote image url in the server
    :param pre: prefix of the visual genome image folder

    :return: local file name
    """
    return os.path.join(pre, url.split('/')[-1])

def _visualize_bbxs(objects, data_dir, num_bbxs=-1):
    """
    visualize objects in an image

    :param objects: objects (including corresponding image) need to be drawn
    :param data_dir: directory where 'objects.json' is stored
    :param num_bbxs: how many bounding boxes to display
    """
    img_path = os.path.join(data_dir, _remote_to_local(objects['image_url']))
    img = imread(img_path)
    plt.imshow(img)
    img_bbxs = objects['objects']
    ax = plt.gca()
    if num_bbxs < 0:
        num_bbxs = len(img_bbxs)
    for bbx in img_bbxs[:num_bbxs]:
        if len(bbx['synsets']) > 0:
            color = np.random.rand(3)
            ax.add_patch(Rectangle((bbx['x'], bbx['y']),
                                   bbx['w'],
                                   bbx['h'],
                                   fill=False,
                                   edgecolor=color,
                                   linewidth=3))
            ax.text(bbx['x'], bbx['y'],
                    '/'.join(synset.split('.')[0] for synset in bbx['synsets']),
                    style='italic',
                    size='larger',
                    bbox={'facecolor':'white', 'alpha':.5})
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()


def _count_coco(data_dir, data_type, data_year):
    """
    calculate coco statistics per category

    :param data_dir: root directory of COCO
    :param data_type: train or val
    :param data_year: 2014 or 2017
    """

    anno_file = '{}/annotations/instances_{}{}.json'.\
        format(data_dir, data_type, data_year)
    coco = COCO(anno_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_stats = []
    for cnt, cat in enumerate(cats, 1):
        cat_name = cat['name']
        img_ids = coco.getImgIds(catIds=coco.getCatIds([cat_name]))
        ann_ids = coco.getAnnIds(catIds=coco.getCatIds([cat_name]))
        cat_stats.append((cat_name, len(img_ids), len(ann_ids)))
        print('[{}] {} counted...'.format(cnt, cat_name))
    plt.subplot(2, 1, 1)
    cat_names, cat_imgs, cat_anns = zip(*sorted(cat_stats, key=lambda x_y_z: -x_y_z[2]))
    plt.bar(range(len(cat_names)), cat_anns, tick_label=cat_names)
    plt.title('#Instances Per Category')

    plt.subplot(2, 1, 2)
    cat_names, cat_imgs, cat_anns = zip(*sorted(cat_stats, key=lambda x_y_z: -x_y_z[1]))
    plt.bar(range(len(cat_names)), cat_imgs, tick_label=cat_names)
    plt.title('#Images Per Category')
    plt.show()
'''
cats : synsets
imgs : imgs dir
anns : bbox/attribute 'object'
abn_anns : proposals of synsets length > 1 
rels : relationships
attrs : attributes
'''
class VG:
    def __init__(self, data_dir, annotation_file=None, relation_file=None, attr_file=None, num=-1, stats=False):
        self.data_dir = data_dir
        self.num = num
        self.dataset = dict()
        self.anns, self.cats, self.imgs = dict(), dict(), dict()
        self.ann_lens, self.img_lens = {}, {}
        self.img_to_anns, self.cat_to_imgs = defaultdict(list), defaultdict(list)

        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(os.path.join(self.data_dir,
                                                  annotation_file), 'r'))
            print('Done (t={:0.2f}s'.format(time.time() - tic))
            self.dataset = dataset
            self.create_index()

        del dataset, self.dataset
        gc.collect()

    def create_index(self):
        print('creating index...')
        if self.num < 0:
            self.num = len(self.dataset)
        for cnt, img in enumerate(self.dataset[:self.num], 1):
            self.imgs[img['image_id']] = img
            for ann in img['objects']:
                ann['image_id'] = img['image_id']
                synsets = ann['synsets']
                synset = synsets[0]
                if 'category_id' not in ann:
                    if synset not in self.cats.values():
                        self.cats[len(self.cats)] = synset
                        category_id = len(self.cats) - 1
                    else:
                        category_id = _get_cat_id(synset, self.cats)
                    ann['category_id'] = category_id
                else:
                    category_id = ann['category_id']
                    self.cats[category_id] = synset
                self.cat_to_imgs[category_id].append(img['image_id'])
                self.img_to_anns[img['image_id']].append(ann['object_id'])
                self.anns[ann['object_id']] = ann

            if cnt % 100 == 0:
                print('{} images indexed...'.format(cnt))
            elif cnt == self.num:
                print('{} images indexed...'.format(cnt))
        print('index created!')

    def get_ann_ids(self, cat_ids=[], img_ids=[]):
        cat_ids = cat_ids if _like_array(cat_ids) else [cat_ids]
        img_ids = img_ids if _like_array(img_ids) else [img_ids]

        if len(img_ids) > 0:
            lists = [self.img_to_anns[img_id] for img_id in img_ids
                     if img_id in self.img_to_anns]
            ids = list(itertools.chain.from_iterable(lists))
        else:
            ids = self.anns.keys()
        if len(cat_ids) > 0:
            ids = [idx for idx in ids if
                   self.anns[idx]['category_id'] in cat_ids]
        return ids

    def get_cat_ids(self, cat_ids=[]):
        cat_ids = cat_ids if _like_array(cat_ids) else [cat_ids]

        ids = self.cats.keys()
        if len(cat_ids) > 0:
            ids = [cat_id for cat_id in cat_ids if cat_id in ids]
        return sorted(ids)

    def get_img_ids(self, cat_ids=[], img_ids=[]):
        cat_ids = cat_ids if _like_array(cat_ids) else [cat_ids]
        img_ids = img_ids if _like_array(img_ids) else [img_ids]

        if len(img_ids) > 0:
            ids = set(img_ids) & set(self.imgs.keys())
        else:
            ids = set(self.imgs.keys())
        for i, cat_id in enumerate(cat_ids):
            if i == 0:
                ids_int = ids & set(self.cat_to_imgs[cat_id])
            else:
                ids_int |= ids & set(self.cat_to_imgs[cat_id])
        if len(cat_ids) > 0:
            return list(ids_int)
        else:
            return list(ids)

    def load_anns(self, ids=[]):
        if _like_array(ids):
            return [self.anns[idx] for idx in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def load_cats(self, ids=[]):
        if _like_array(ids):
            return [self.cats[idx] for idx in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def load_imgs(self, ids=[]):
        if _like_array(ids):
            return [self.imgs[idx] for idx in ids]
        elif type(ids) is int:
            return [self.imgs[ids]]

    def show_cat_anns(self, cat_id=None, img_in=None, ann_in=None):
    #according cat_id or cat_id&img_ids show picture with gt bbox

        if not img_in:
            img_ids = self.get_img_ids(cat_id)
        else:
            img_ids = [0]
        for img_id in img_ids:
            if not img_in:
                img_path = os.path.join(self.data_dir,
                                        _remote_to_local(self.imgs[img_id]['image_url']))
            else:
                img_path = os.path.join(self.data_dir,
                                        _remote_to_local(img_in['image_url']))
            img = imread(img_path)
            plt.imshow(img)

            if not ann_in:
                ann_ids = self.get_ann_ids(cat_id, img_id)
            else:
                ann_ids = [0]
            ax = plt.gca()
            for ann_id in ann_ids:
                color = np.random.rand(3)
                if not ann_in:
                    ann = self.anns[ann_id]
                else:
                    ann = ann_in
                ax.add_patch(Rectangle((ann['x'], ann['y']),
                                       ann['w'],
                                       ann['h'],
                                       fill=False,
                                       edgecolor=color,
                                       linewidth=3))
                ax.text(ann['x'], ann['y'],
                        'name: ' + ann['names'][0],
                        style='italic',
                        size='larger',
                        bbox={'facecolor': 'white', 'alpha': .5})
                ax.text(ann['x'], ann['y']+ann['h'],
                        'synsets: ' + ','.join(ann['synsets']),
                        style='italic',
                        size='larger',
                        bbox={'facecolor': 'white', 'alpha': .5})
            plt.show()

    def load_res(self, res_dir, res_file):
        return VG(res_dir, res_file)


if __name__ == '__main__':
    data_dir = '/data/VisualGenome/'

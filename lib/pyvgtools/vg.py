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
from collections import defaultdict, Counter
import time
import itertools
import gc

MAX_ATTR = 180
MAX_REL = 180
MAX_CLS = 1000

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


def _object_synsets(objects, num=-1):
    """
    count instances of object synsets in Visual Genome

    :param objects: images considered

    :return categories: dictionary of categories containing instance number
    """
    categories = {}
    if num < 0:
        num = len(objects)
    for cnt, image in enumerate(objects[:num], 1):
        for object in image['objects']:
            synsets = object['synsets']
            for synset in synsets:
                if synset in categories:
                    image_ids = categories[synset]['image_ids']
                    image_id = image['image_id']
                    if image_id not in image_ids:
                        image_ids.append(image_id)

                    object_ids = categories[synset]['object_ids']
                    object_id = object['object_id']
                    if object_id not in object_ids:
                        object_ids.append(object_id)
                else:
                    categories[synset] = {'image_ids': [image['image_id']],
                                          'object_ids': [object['object_id']]}
        if cnt % 100 == 0:
            print('%d images\' objects\' synsets processed...' % cnt)
        elif cnt == num:
            print('%d images\' objects\' synsets processed...' % cnt)
    return categories


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


def _construct_graph(synsets):
    """
    construct a graph for synsets using WordNet and NetworkX

    :param synsets: synsets need to be added

    :return graph: constructed graph
    """
    graph = nx.DiGraph()
    seen = set()

    def recurse(s):
        """
        recursively add synset and its hypernyms to the graph

        :param s: synset and whose hypernyms need to be added
        """
        if s not in seen:
            seen.add(s)
            # TODO: a synset may have >= 2 hypernyms
            for hn in s.hypernyms()[:1]:
                graph.add_edge(hn.name(), s.name())
                recurse(hn)

    for s in synsets:
        recurse(s)
    return graph


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
    def __init__(self, data_dir, annotation_file=None, num=-1, stats=False, align_dir=None):
        self.data_dir = data_dir
        self.num = num
        self.dataset = dict()
        self.anns, self.abn_anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.ann_lens, self.img_lens = {}, {}
        self.img_to_anns, self.cat_to_imgs = defaultdict(list), defaultdict(list)
        self.align_list = dict()

        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(os.path.join(self.data_dir,
                                                  annotation_file), 'r'))
            print('Done (t={:0.2f}s'.format(time.time() - tic))
            self.dataset = dataset
            if align_dir is not None:
                if align_dir == 'val':
                    self.align_list[954] = 'pigeon.n.01' # vg1000 val
                else:
                    align_path = os.path.join(self.data_dir, 'vg_' + align_dir + '_align.json')
                    self.align_list = json.load(open(align_path, 'r'))

            self.create_index()
            if stats:
                self.compute_cat_stats()

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
                # TODO: a box may have >= 2 or 0 synsets
                if len(synsets) != 1:
                    # self.show_cat_anns(img_in=img, ann_in=ann)
                    self.abn_anns[ann['object_id']] = ann
                # only consider those objects with exactly one synset
                else:
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
                    # self.cats[954] = 'pigeon.n.01' #vg1000 test
            if cnt % 100 == 0:
                print('{} images indexed...'.format(cnt))
            elif cnt == self.num:
                print('{} images indexed...'.format(cnt))
        if self.align_list:
            for a_i in self.align_list:
                self.cats[int(a_i)] = self.align_list[a_i]
            print("########### add lacking label done ##################")
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
        return sorted(ids)

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

    def compute_cat_stats(self, full=False):
        ann_lens, img_lens = {}, {}
        for cnt, cat_id in enumerate(self.cats, 1):
            ann_lens[cat_id] = len(self.get_ann_ids(cat_id))
            img_lens[cat_id] = len(self.get_img_ids(cat_id))
            if cnt % 10 == 0:
                print('{} categories computed...'.format(cnt))
            elif cnt == len(self.cats):
                print('{} categories computed...'.format(cnt))

        self.ann_lens = sorted(ann_lens.items(),
                               key=lambda k_v: -k_v[1])
        self.img_lens = sorted(img_lens.items(),
                               key=lambda k_v: -k_v[1])
        if full:
            with open(os.path.join(self.data_dir, 'ann_lens_1000.txt'), 'w') as f:
                f.write('{},{},{}\n'.format('synset', 'category_id', '#instances'))
                for cat in self.ann_lens:
                    f.write('{},{},{}\n'.format(self.cats[cat[0]], cat[0], cat[1]))

            # with open(os.path.join(self.data_dir, 'img_lens.txt'), 'w') as f:
            #     f.write('{},{},{}\n'.format('synset', 'category_id', '#images'))
            #     for cat in self.img_lens:
            #         f.write('{},{},{}\n'.format(self.cats[cat[0]], cat[0], cat[1]))

        # cat_ids, ann_lens = zip(*self.ann_lens)
        # cats = [self.cats[cat_id].split('.')[0] for cat_id in cat_ids]
        # plt.subplot(2, 1, 1)
        # plt.bar(range(cnt), ann_lens, tick_label=cats)
        # plt.title('#Instances Per Category')
        #
        # cat_ids, img_lens = zip(*self.img_lens)
        # cats = [self.cats[cat_id].split('.')[0] for cat_id in cat_ids]
        # plt.subplot(2, 1, 2)
        # plt.bar(range(cnt), img_lens, tick_label=cats)
        # plt.title('#Images Per Category')
        # plt.show()

    def draw_synset_graph(self, ann_ids):
        """
        draw synsets in an image

        :param objects: objects (synsets) need to be drawn
        """
        synsets = []
        for ann_id in ann_ids:
            object = self.anns[ann_id]
            if len(object['synsets']) > 0:
                synsets += [wn.synset(synset) for synset in object['synsets']]
        graph = _construct_graph(synsets)
        colors = []
        for node in graph:
            if node in map(lambda x: x.name(), synsets):
                colors.append('r')
            elif node in ['entity.n.01']:
                colors.append('g')
            else:
                colors.append('b')
        nx.draw_networkx(graph, pos=gl(graph), node_color=colors)
        plt.tick_params(labelbottom='off', labelleft='off')
        # plt.show()
        plt.savefig("cls_synset.png")
        gc.collect()

    def get_major_ids(self, list, num=1000):
        sorted_cat_ids = np.loadtxt(
            os.path.join(self.data_dir, list),
            dtype=np.int32, delimiter=',', skiprows=1, usecols=1)
        return sorted_cat_ids[:num].tolist()

    def dump_train_val(self, val_num=5000):
        cat_ids = self.get_major_ids('ann_lens.txt')
        img_ids = self.get_img_ids(cat_ids)
        print('{} out of {} images are left for train/val'.
            format(len(img_ids), len(self.imgs)))
        for img_id in img_ids:
            self.imgs[img_id]['objects'] =\
                [object for object in self.imgs[img_id]['objects'] if
                 'category_id' in object and object['category_id'] in cat_ids]
        img_ids = np.array(img_ids)
        val_ids = set(np.random.choice(img_ids, val_num, False).tolist())
        assert len(val_ids) == val_num
        img_ids = set(img_ids.tolist())
        train_ids = img_ids - val_ids
        assert len(train_ids) + len(val_ids) == len(img_ids)
        train_imgs = [self.imgs[img_id] for img_id in train_ids]
        val_imgs = [self.imgs[img_id] for img_id in val_ids]

        with open(os.path.join(data_dir, 'objects_train.json'), 'w') as ft:
            json.dump(train_imgs, ft)
        with open(os.path.join(data_dir, 'objects_val.json'), 'w') as fv:
            json.dump(val_imgs, fv)

    def load_res(self, res_dir, res_file):
        return VG(res_dir, res_file)


if __name__ == '__main__':
    data_dir = '/data/VisualGenome/'

    vg = VG(data_dir, 'objects.json', 'relationships.json','attributes.json',stats=True)
    # vg = VG(data_dir, annotation_file=None, relation_file=None, attr_file='attributes.json',stats=True)
    vg.dump_train_val()


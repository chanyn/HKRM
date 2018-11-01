from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os
import sys
import os.path as osp
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
import itertools
import pdb
# VG API
from pyvgtools.vg import VG
from pyvgtools.vgeval import VGeval

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

MAX_RELATION = 200
MAX_ATTRIBUTE = 200

class vg(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'vg_' + image_set)
        # VG specific config options
        self.config = {'use_salt': True,
                       'cleanup': True}
        # name, path
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'vg')
        # load VG API, classes, class <-> id mappings
        if image_set.find('val') == -1:
            self._VG = VG(self._data_path, self._get_ann_file())
        else:
            self._VG = VG(self._data_path, self._get_ann_file(), align_dir=image_set)

        cat_ids = self._VG.get_cat_ids()
        # cat_ids = pickle.load(open('/home/jiangchenhan/code/faster-rcnn.pytorch-master2/data/vg/val_ids.pkl', 'rb'))

        cats = self._VG.load_cats(cat_ids)
        self._class_to_vg_id = dict(zip(cats, cat_ids))

        self._classes = tuple(['__background__'] + cats)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        # Lookup table to map from VG category ids to internal class indices
        self._vg_id_to_class_ind = dict([(self._class_to_vg_id[cls], self._class_to_ind[cls])
                                         for cls in self._classes[1:]])
        self._image_index = self._VG.get_img_ids()
        # Default to roidb handler
        self.set_proposal_method('gt')
        self.competition_mode(False)

        self._data_name = image_set
        # Dataset splits that have ground-truth annotations
        self._gt_splits = ('train', 'val')

    def _get_ann_file(self):
        return osp.join(self._data_path, 'objects_' + self._image_set + '.json')

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def image_path_from_index(self, index, pre='VG/', suf='.jpg'):
        """
        Construct an image path from the image's "index" identifier.
        """
        file_name = pre + '{}'.format(index) + suf
        image_path = os.path.join(self._data_path, file_name)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_vg_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _load_vg_annotation(self, index):
        """
        Loads VG bounding-box instance annotations.
        """
        img = self._VG.load_imgs(index)[0]
        width = img['width']
        height = img['height']

        ann_ids = self._VG.get_ann_ids(img_ids=index)
        anns = self._VG.load_anns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_anns = []
        for ann in anns:
            x1 = np.max((0, ann['x']))
            y1 = np.max((0, ann['y']))
            x2 = np.min((width - 1, x1 + np.max((0, ann['w'] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, ann['h'] - 1))))
            if x2 >= x1 and y2 >= y1:
                ann['clean_bbox'] = [x1, y1, x2, y2]
                valid_anns.append(ann)
        anns = valid_anns
        num_anns = len(anns)

        boxes = np.zeros((num_anns, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_anns), dtype=np.int32)
        overlaps = np.zeros((num_anns, self.num_classes), dtype=np.float32)

        # # Lookup table to map from VG category ids to our internal class
        # # indices
        # vg_id_to_class_ind = dict([(self._class_to_vg_id[cls], self._class_to_ind[cls])
        #                            for cls in self._classes[1:]])

        for ix, ann in enumerate(anns):
            cls =self._vg_id_to_class_ind[ann['category_id']]
            boxes[ix, :] = ann['clean_bbox']
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        ds_utils.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'width': width,
                'height': height,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def _get_widths(self):
        return [r['width'] for r in self.roidb]

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'width': widths[i],
                     'height': self.roidb[i]['height'],
                     'boxes': boxes,
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'flipped': True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = osp.join(output_dir, ('detections_' +
                                         self._image_set +
                                         '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_vg_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._image_set.find('test') == -1:
            self._do_detection_eval(res_file.split('/')[-1], output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)

    def _print_detection_eval_metrics(self, vg_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(vg_eval, thr):
            ind = np.where((vg_eval.params.iouThrs > thr - 1e-5) &
                           (vg_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = vg_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(vg_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(vg_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            vg_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__' or cls == 'pigeon.n.01':
                continue
            # minus 1 because of __background__
            precision = vg_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{}: {:.1f}'.format(cls, 100 * ap))

        print('~~~~ Summary metrics ~~~~')
        vg_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        vg_dt = self._VG.load_res(output_dir, res_file)
        vg_eval = VGeval(self._VG, vg_dt)
        vg_eval.evaluate()
        vg_eval.accumulate()
        self._print_detection_eval_metrics(vg_eval)
        # eval_file = osp.join(output_dir, 'detection_results.pkl')
        # with open(eval_file, 'wb') as fid:
        #     pickle.dump(vg_eval, fid, pickle.HIGHEST_PROTOCOL)
        # print('Wrote VG eval results to: {}'.format(eval_file))

    def _vg_results_one_image(self, boxes, index, cnt):
        results = []
        for cls_ind, cls in enumerate(self.classes[1:]):
            dets = boxes[cls_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': self._class_to_vg_id[cls],
                  'x': xs[k],
                  'y': ys[k],
                  'w': ws[k],
                  'h': hs[k],
                  'object_id': cnt + k,
                  'synsets': [cls],
                  'score': scores[k]} for k in range(dets.shape[0])])
            cnt += dets.shape[0]
        return results, cnt

    def _write_vg_results_file(self, all_boxes, res_file):
        results = []
        cnt = 0
        for img_ind, index in enumerate(self.image_index):
            print('Collecting {} results ({:d}/{:d})'.format(index, img_ind+1,
                                                             len(self.image_index)))
            image = {'image_id': index}
            objects, cnt = self._vg_results_one_image([all_boxes[cls_ind][img_ind]
                                                       for cls_ind in range(1, len(self.classes))],
                                                      index, cnt)
            image['objects'] = objects
            results.append(image)
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = vg('val')
    res = d.roidb
    from IPython import embed; embed()

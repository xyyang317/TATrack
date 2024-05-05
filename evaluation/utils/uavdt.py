from __future__ import absolute_import
import os
import shutil

import numpy as np
import json

from evaluation.datasets.uavdtdataset import UAVDT
from evaluation.utils.metrics import rect_iou, center_error


class ExperimentUAVDT(object):
    r"""Experiment pipeline and evaluation toolkit for DTB70 dataset.

    Args:
        root_dir (string): Root directory of DTB70 dataset.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, result_dir='results', report_dir='reports'):
        self.dataset = UAVDT(root_dir)
        self.result_dir = os.path.join(result_dir, 'uavdt')
        self.report_dir = os.path.join(report_dir, 'uavdt')
        dirs = os.listdir(self.result_dir)
        if not os.path.isdir(os.path.join(self.result_dir, 'times')):
            os.mkdir(os.path.join(self.result_dir, 'times'), mode=0o777)
        for i in dirs:
            if i[-8:-4] == 'time':
                oldpath = os.path.join(self.result_dir, i)
                newpath = os.path.join(self.result_dir, 'times', i)
                shutil.move(oldpath, newpath)
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51


    def report(self, tracker_names,dataset,epoch):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir)
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, '{}.json'.format(tracker_names[0]))
        if not os.path.isfile(report_file):
            file = open(report_file,'w')
            file.write('{}')
            file.close()

        with open(report_file, 'r', encoding='utf8') as fp:
            performance = json.load(fp)

        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            speeds = np.zeros(seq_num)

            if dataset not in performance:
                performance.update({'{}'.format(dataset): {}})
            performance['{}'.format(dataset)].update({'epoch{}'.format(epoch): {}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(self.result_dir,'%s.txt' % seq_name)
                boxes = np.loadtxt(record_file, delimiter='\t')
                boxes[0] = anno[0]
                if not (len(boxes) == len(anno)):
                    print('warning: %s anno donnot match boxes' % seq_name)
                    len_min = min(len(boxes), len(anno))
                    boxes = boxes[:len_min]
                    anno = anno[:len_min]
                assert len(boxes) == len(anno)

                ious, center_errors = self._calc_metrics(boxes, anno)
                succ_curve[s], prec_curve[s] = self._calc_curves(
                    ious, center_errors)

                # calculate average tracking speed
                time_file = os.path.join(self.result_dir,'times/%s_time.txt' % seq_name)
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)
            succ_score = np.mean(succ_curve)
            prec_score = prec_curve[20]
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            performance['{}'.format(dataset)]['epoch{}'.format(epoch)].update({
                'success_score':
                    succ_score,
                'precision_score':
                    prec_score,
                'speed_fps':
                    avg_speed
            })

        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)




    def _calc_metrics(self, boxes, anno):
        # can be modified by children classes
        ious = rect_iou(boxes, anno)
        center_errors = center_error(boxes, anno)
        return ious, center_errors

    def _calc_curves(self, ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)

        return succ_curve, prec_curve


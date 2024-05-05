import json
import os
import shutil
from pathlib import Path

from lib.test.evaluation.local import local_env_settings

from evaluation.utils.dtb70 import ExperimentDTB70
from evaluation.utils.uav123_10fps import ExperimentUAV123_10fps
from evaluation.utils.uav123 import ExperimentUAV123
from evaluation.utils.uavdt import ExperimentUAVDT
from evaluation.utils.uavtrack import ExperimentUAVTRACK
from evaluation.utils.visdrone import ExperimentVISDRONE

datasets = ['dtb70', 'uavdt', 'visdrone', 'uav123_10fps', 'uav123', 'uavtrack']
datasets = datasets[4:]
settings = local_env_settings()
root_dirs = {'dtb70': settings.dtb70_path, 'uavdt': settings.uavdt_path, 'visdrone': settings.visdrone_path,
             'uav123_10fps': settings.uav123_10fps_path, 'uav123': settings.uav123_path, 'uavtrack': settings.uavtrack_path}
func_dict = {'dtb70': ExperimentDTB70, 'uavdt': ExperimentUAVDT, 'visdrone': ExperimentVISDRONE,
             'uav123_10fps': ExperimentUAV123_10fps, 'uav123': ExperimentUAV123, 'uavtrack': ExperimentUAVTRACK}

tracker_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）'
tracker_param = 'avit_tiny_patch16_224'
result_dir = os.path.join(tracker_path, 'output/test/tracking_results/ostrack', tracker_param)
report_dir = os.path.join(tracker_path, 'evaluation/output/report')
threads = 0
num_gpus = 1
#1.8'python /home/lsw/LSW/2023/ostrack-deit（MIM）/tracking/test.py ostrack avit_tiny_patch16_224 --dataset dtb70 --threads 0 --num_gpus 1 --test_epoch 100'
epochs = [300]
for epoch in epochs:
    if Path(result_dir).is_dir():
        dir_list = os.listdir(result_dir)
        if len(dir_list) >= 2:
            shutil.rmtree(result_dir)
    # shutil.rmtree(result_dir)
    for dataset in datasets:
        test_cmd = "python {}/tracking/test.py ostrack {} --dataset {} --threads {} --num_gpus {} --test_epoch {}".format(
            tracker_path, tracker_param, dataset, threads, num_gpus, epoch)
        os.system(test_cmd)

        root_dir = root_dirs[dataset]
        experiment = func_dict.get(dataset)(root_dir, result_dir, report_dir)
        report = experiment.report([tracker_param], dataset, epoch)

performance = {}
for data in datasets:
    path = os.path.join(report_dir, data, '{}.json'.format(tracker_param))
    with open(path, 'r') as f:
        json_data = json.load(f)
    performance.update(json_data)

report_file = os.path.join(report_dir, '{}_report.json'.format(tracker_param))
with open(report_file, 'w') as f:
    json.dump(performance, f, indent=4)

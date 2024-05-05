from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/got10k_lmdb'
    settings.got10k_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/itb'
    settings.lasot_extension_subset_path_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/lasot_lmdb'
    settings.lasot_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/lasot'
    settings.network_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/nfs'
    settings.otb_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/otb'
    settings.prj_dir = '/home/lsw/LSW/2023/ostrack-deit（MIM）'
    settings.result_plot_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/output/test/result_plots'
    settings.results_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/lsw/LSW/2023/ostrack-deit（MIM）/output'
    settings.segmentation_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/output/test/segmentation_results'
    settings.tc128_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/trackingnet'
    settings.uav_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/uav'
    settings.vot18_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/vot2018'
    settings.vot22_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/vot2022'
    settings.vot_path = '/home/lsw/LSW/2023/ostrack-deit（MIM）/data/VOT2019'
    settings.youtubevos_dir = ''

    settings.dtb70_path = '/home/lsw/data/DTB70'
    settings.uavdt_path = '/home/lsw/data/UAVDT'
    settings.visdrone_path = '/home/lsw/data/VisDrone2019-SOT-test-dev/VisDrone2018'
    settings.uav123_10fps_path = '/home/lsw/data/UAV123@10FPS'
    settings.uav123_path = '/home/lsw/data/UAV123'
    settings.uavtrack_path = '/home/lsw/data/UAVTrack112/V4RFlight112'

    return settings


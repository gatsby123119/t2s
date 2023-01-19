from enum import IntEnum

#from models.CocoPoseNet import CocoPoseNet

#from models.FaceNet import FaceNet
#from models.HandNet import HandNet
from pathlib import Path

ObjectCategory = {
    """ types of objects """
    
    'pipette':1,
    'PCR_tube':2,
    'tube':3,
    'waste_box':4,
    'vial':5,
    'measuring_flask':6,
    'beaker':7,
    'wash_bottle':8,
    'water_bottle':9,
    'erlenmeyer_flask':10,
    'culture_plate':11,
    'spoon':12,
    'electronic_scale':13,
    'LB_solution':14,
    'stopwatch':15,
    'D_sorbitol':16,
    'solution_P1':17,
    'plastic_bottle':18,
    'agarose':19,
    'cell_spreader':20,
}

params = {
    'coco_dir': './dataset/coco2017',
    'pretrained_path' : 'models/pretrained_vgg_base.pth',
    # training params
    'min_keypoints': 5,
    'min_area': 32 * 32,
    'insize': 368,
    'downscale': 8,
    'paf_sigma': 8,
    'heatmap_sigma': 7,
    'batch_size': 2,
    'lr': 1e-4,
    'num_workers': 0,
    'eva_num': 100,
    'board_loss_interval': 100,
    'eval_interval': 4,
    'board_pred_image_interval': 2,
    'save_interval': 2,
    'log_path': 'work_space/log',
    'work_space': Path('work_space'),
    
    'min_box_size': 64,
    'max_box_size': 512,
    'min_scale': 0.5,
    'max_scale': 2.0,
    'max_rotate_degree': 40,
    'center_perterb_max': 40,

    # inference params
    'inference_img_size': 368,
    'inference_scales': [0.5, 1, 1.5, 2],
    # 'inference_scales': [1.0],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,
    'n_integ_points_thresh': 8,
    'heatmap_peak_thresh': 0.05,
    'inner_product_thresh': 0.05,
    'limb_length_ratio': 1.0,
    'length_penalty_value': 1,
    'n_subset_limbs_thresh': 3,
    'subset_score_thresh': 0.2,
    
}

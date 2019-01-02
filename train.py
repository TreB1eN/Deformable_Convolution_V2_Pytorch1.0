import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Lighthead - RCNN")
    parser.add_argument("-r", "--resume", help="whether resume from the latest saved model",action="store_true")
    parser.add_argument("-save", "--from_save_folder", help="whether resume from the save path",action="store_true")
    args = parser.parse_args()
    return args

from config import cfg
cfg.merge_from_file('configs/e2e_deformconv_mask_rcnn_R_50_C5_1x.yaml')
cfg.freeze()

from DeConv_Learner import Learner

if __name__ == "__main__":
    args = parse_args()
    learner = Learner(cfg)
    learner.train(resume = args.resume, from_save_folder = args.from_save_folder)
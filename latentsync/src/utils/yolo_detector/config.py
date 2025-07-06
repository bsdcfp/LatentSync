# coding:utf-8
# author:tubb.tao
# define the confif file
import argparse

## 同步帧数
syncnet_T = 5
syncnet_mel_step_size = 16

## 定义全局step和epoch，写的太乱，后面有时间再优化
global_step = 0
global_epoch = 0

## 定义是否使用超分来处理
use_sr = False

## 定义是否使用disc来进行训练
use_disc = True

## 定义cache的路径
cache_dir = '/home/work/mel_feature'

class opts(object):
    """
    define config file 
    """
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')
        self.parser.add_argument("--data_root_list", nargs='+', help="frame root", default=['/home/work/align_hdtf/frame'], type=str)
        self.parser.add_argument("--txt_root_list", nargs='+', help="txt root", default=['/home/work/align_hdtf/txt'], type=str)
        self.parser.add_argument("--name_list", nargs='+', help="name list", default=['/home/work/align_hdtf/names.txt'], type=str)
        self.parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='/home/work/ywan_data_check/wav2lip_384/torchrun_ckpt_in_finetune', type=str)
        self.parser.add_argument('--syncnet_model_type', help='choice model type, [sync384, sync96]', default='sync96', type=str)
        self.parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', default='./weights/syncnet/lipsync_expert.pth', type=str)
        self.parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default='/home/work/ywan/models/wav2lip/model_384/gen_555000.pth', type=str)
        self.parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default=None, type=str)
    
    def get_config(self, kwargs=""):
        if kwargs:
            args = self.parser.parse_args(kwargs)
        else:
            args = self.parser.parse_args()
        return args

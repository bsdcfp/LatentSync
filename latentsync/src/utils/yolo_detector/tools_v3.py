# coding:utf-8
# author:tubb.tao
# modified
import os
import cv2
import math
import copy
import torch
import numpy as np
from .config import *
from PIL import Image
from os.path import join 
from ultralytics import YOLO
from hparams import hparams
# from models import SyncNet_96, SyncNet_384
from torchalign import FacialLandmarkDetector
from utils import decompose_tfm, img_warp, img_warp_back_inv_m, metrix_M

### do not use it to inference

def load_face_model(pretrained_model_dir, device):
    face_det = YOLO(f'{pretrained_model_dir}/yolov8n-face/yolov8n-face.pt')
    lmk_net = FacialLandmarkDetector(f'{pretrained_model_dir}/wflw/hrnet18_256x256_p1/')
    lmk_net.eval()
    face_det.to(device)
    lmk_net.to(device)
    return face_det, lmk_net

def landmark_to_keypoints(landmark):
    lefteye = np.mean(landmark[60:68, :], axis=0)
    righteye = np.mean(landmark[68:76, :], axis=0)
    nose = landmark[54, :]
    leftmouth = (landmark[76, :] + landmark[88, :]) / 2
    rightmouth = (landmark[82, :] + landmark[92, :]) / 2
    return (lefteye, righteye, nose, leftmouth, rightmouth)

@torch.no_grad()
def detect_face(face_det, face_img, device):
    boxes = face_det(face_img,
                            imgsz=640,
                            conf=0.01,
                            iou=0.5,
                            half=True,
                            augment=False,
                            device=device)[0].boxes
    bboxes = boxes.xyxy.cpu().numpy()
    return bboxes

@torch.no_grad()
def detect_lmk(lmk_net, image, device, bbox=None):
    if isinstance(bbox, list):
        bbox = np.array(bbox)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    bbox_tensor = torch.from_numpy(bbox[:, :4])
    landmark = lmk_net(img_pil, bbox=bbox_tensor, device=device).cpu().numpy()
    return landmark

@torch.no_grad()
def get_input_imginfo(face_det, lmk_net, frame, device, img_size=(256, 256), pads=(0, 0, 0, 0)):
    bbox = detect_face(face_det, frame.copy(), device)[0][:5]
    landmark = detect_lmk(lmk_net, frame.copy(), device, [bbox])[0]
    keypoints = landmark_to_keypoints(landmark)

    # keypoints = self.kpts_smoother.smooth(np.array(keypoints))

    m = metrix_M(face_size=200, expand_size=256, keypoints=keypoints)

    align_frame = img_warp(frame, m, 256, adjust=0)
    align_bbox = detect_face(face_det, align_frame.copy(), device)[0][:4]

    # align_bbox = self.abox_smoother.smooth(np.reshape(align_bbox, (-1, 2))).reshape(-1)

    # 重新warp 图片，保持scale 不变
    w, h = 256, 256
    rt, s = decompose_tfm(m)
    s_x, s_y = s[0][0], s[1][1]
    m = rt
    align_frame = cv2.warpAffine(frame, m, (math.ceil(w / s_x), math.ceil(h / s_y)), flags=cv2.INTER_CUBIC)
    inv_m = cv2.invertAffineTransform(m)

    face = copy.deepcopy(align_frame)
    h, w, c = align_frame.shape
    bbox = align_bbox
    bbox[0] *= (w - 1) / 255
    bbox[1] *= (h - 1) / 255
    bbox[2] *= (w - 1) / 255
    bbox[3] *= (h - 1) / 255

    rect = [round(f) for f in bbox[:4]]
    pady1, pady2, padx1, padx2 = pads
    y1 = max(0, rect[1] - pady1)
    y2 = min(h, rect[3] + pady2)
    x1 = max(0, rect[0] - padx1)
    x2 = min(w, rect[2] + padx2)

    coords = (y1, y2, x1, x2)
    face = face[y1:y2, x1:x2]
    print(face.shape)
    face = cv2.resize(face, img_size)
    cv2.imwrite('test_1.jpg', face)

    return {
        'img': face,
        'frame': frame,
        'coords': coords,
        'align_frame': align_frame,
        'm': m,
        'inv_m': inv_m,
    }

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    """
    加载模型
    """
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage, weights_only=True)
    print("Load checkpoint from: {} out".format(path))
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    return model

# def load_syncnet(model_path, device, syncnet_model_type='sync96'):
#     """
#     加载sync的模型
#     """
#     if syncnet_model_type == 'sync96':
#         syncnet = SyncNet_96().to(device)
#     else:
#         syncnet = SyncNet_384().to(device)
#     syncnet = load_checkpoint(model_path, syncnet, None, True, False)
#     for p in syncnet.parameters():
#         p.requires_grad = False
#     syncnet.eval()
#     return syncnet

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    """
    用于存储训练过程中的样本
    """
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    """
    保存模型的相关的信息
    """
    ### 记录全局信息
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    """
    test lmk model and preprocess
    """
    pretrained_model_dir = './weights'
    test_image = './test_data/cf/108.jpg'
    test_image = './test_data/rd/108.jpg'
    frame = cv2.imread(test_image)
    face_det, face_lmk = load_face_model(pretrained_model_dir, device)
    ## 
    import time
    t1 = time.time()
    get_input_imginfo(face_det, face_lmk, frame, device)
    print(time.time()-t1)

    """
    test load model
    """
    # sync_path = './weights/syncnet/ema_checkpoint_step000189000.pth'
    # syncnet = load_syncnet(sync_path, device)
    # input = torch.rand((1, 3, 256, 256))
    # output = syncnet(input)
    # print(ouptut.shape)
    # print(syncnet)
    """
    test load wave2lip model
    """
    # from models import Wav2Lip, Wav2Lip_disc_qual
    # checkpoint_path = './weights/wav2lip/ema_checkpoint_step000300000.pth'
    # model = Wav2Lip().to(device)
    # model = load_checkpoint(checkpoint_path, model, None, reset_optimizer=True)
    # print(model)


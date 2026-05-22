import argparse
import os
import sys
import os.path as osp
import time
import cv2
import torch
import statistics
### PTC implementation, trt => drive ~ trt_utils

from loguru import logger

# root_path=os.path.dirname(os.path.abspath(os.path.dirname(__file__))) 
# sys.path.append(root_path)  

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class Trk_vars(object):
    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10.0
        self.mot20 = False
        self.fp16 = True
        # inf vars
        self.exp_file = "exps/example/mot/yolox_nano_mix_det.py"
        self.ckpt_file = "pretrained/bytetrack_nano_mot17.pth.tar"
        self.source = "videos/mot16-04.mp4"
        self.save_result = False
        self.save_txt = False
        self.output_dir = "A_results"
        self.videos_f = "track_results"
        self.device = "gpu"
        self.demo = "video"

# input:
# demo image/video/webcam
# -f
# -c
# # default:
# --conf
# --nms
# --tsize
# --fps
# --fp16
# --fuse
# --trt

# --track_thresh
# --track_buffer
# --match_thresh
# --aspect_ratior_thresh
# --min_box_area
# --mot20

# # other:
# -expn
# -n
# --path
# --camid

# # parm
# --save_result
# --device



def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    return parser



def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def imageflow_demo(predictor, vis_folder, current_time, args):

    detection_time = []
    tracker_time = []
    # total_pc = 0

    cap = cv2.VideoCapture(args.source if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.source.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            t0 = time.time()  #############
            outputs, img_info = predictor.inference(frame, timer)
            t1 = time.time()  #############
            detection_time.append(t1-t0)
            if outputs[0] is not None:
                t2 = time.time()
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                t3 = time.time()
                tracker_time.append(t3-t2)
                # total_pc += p_c_total
                # print("Total pedestrains seen: ", total_pc)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                # online_im = plot_tracking(
                #     img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                # )
            # else:
            #     timer.toc()
            #     online_im = img_info['raw_img']
            # if args.save_result:
            #     vid_writer.write(online_im)
            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    return detection_time, tracker_time

def main(exp, args):

    vars_trk = Trk_vars()

    # YOLOX/nano/
    output_dir = osp.join(exp.output_dir, exp.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    if vars_trk.save_result:
        # track_vis/
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = ""

    device = torch.device("cuda" if vars_trk.device == "gpu" else "cpu")

    logger.info("Args: {}".format(vars_trk))

    model = exp.get_model().to(device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    logger.info("loading checkpoint")
    ckpt = torch.load(vars_trk.ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    # fuse + half precision
    model = fuse_model(model)
    model = model.half()  # to FP16

    predictor = Predictor(model, exp, device, vars_trk.fp16)
    current_time = time.localtime()

    det, trk = imageflow_demo(predictor, vis_folder, current_time, vars_trk)
    return det, trk


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp("exps/example/mot/yolox_nano_mix_det.py", None) # args.exp_file

    det, trk = main(exp, args)
    print("\nAverage detection time = ", statistics.mean(det))
    print("detection span = +/-", statistics.variance(det))
    print("\nAverage tracking time = ", statistics.mean(trk))
    print("detection span = +/-", statistics.variance(trk))

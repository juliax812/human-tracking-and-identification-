import glob
import cv2
import numpy as np
import math

#################### Utilities:
# IOU
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    boxAArea = abs((boxA[2]) * (boxA[3]))
    boxBArea = abs((boxB[2]) * (boxB[3]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# tlwhxywh convert
def tlwh_xywh(bb):
    out = [int(bb[0]+bb[2]/2), int(bb[1]+bb[3]/2), bb[2], bb[3]]
    return out
# obtain bb_width_height mean, given sample bbs
def w_h_mean(sample_bbs):
    # sample_bbs = [ [bb1], [bb2], ... ]; bb = [x1, y1, w, h]
    w_sum, w_c = 0,0
    h_sum, h_c = 0,0
    for i in sample_bbs:
        w_sum += i[2]
        w_c += 1
        h_sum += i[3]
        h_c += 1
    w = w_sum/w_c
    h = h_sum/h_c
    return w, h
#################### Read data:
def gt_count(path, gt=True):
    with open(path, 'r') as f:
        lines = f.readlines()
    print(len(lines))

    df = []

    if gt: # GT labels
        for line in lines:
            cur_line = line.rsplit(",")
            if cur_line[6] != '1' or cur_line[7] != '1':
                continue
            to_apd = cur_line[:-1]
            to_apd.append(cur_line[-1][:-1])
            df.append(to_apd)
    else: # Byte labels
        for line in lines:
            cur_line = line.rsplit(",")
            to_apd = cur_line[:-1]
            to_apd.append(cur_line[-1][:-1])
            df.append(to_apd)
            
    frame_wise = {} # {1:[ids], 2:[ids], ...}

    # line = [frame, id, tlwh, 1, 1, vis]
    for line in df:
        # print(line)
        if int(line[0]) not in frame_wise:
            frame_wise[int(line[0])] = [[int(float(line[1]))],[[int(float(line[2])),int(float(line[3])),int(float(line[4])),int(float(line[5]))]]]
        else:
            frame_wise[int(line[0])][0].append(int(float(line[1])))
            frame_wise[int(line[0])][1].append([int(float(line[2])),int(float(line[3])),int(float(line[4])),int(float(line[5]))])

    list_frames = []

    for i in frame_wise:
        list_frames.append([i, frame_wise[i]])

    hist_ids = []
    result = [] # [ [ids, bbs], [ids, bbs], ... ] # ONLY RECURSIVE (hist count ids + bbs)

    for i_f,frame in enumerate(list_frames):
        for index,ids in enumerate(frame[1][0]):
            if ids not in hist_ids:
                hist_ids.append(ids)
                # print(i_f, index, ids)
                try:
                    result[i_f][0].append(ids)
                    result[i_f][1].append(frame[1][1][index])
                except IndexError:
                    result.append([[ids],[frame[1][1][index]],[frame[0]]])
    
    out = []

    for idx,line in enumerate(result):
        if idx ==0:
            out.append(line)
            continue
        if line[2][0] != result[idx-1][2][0]:
            out.append(line)
            continue
        else:
            out[-1][0].append(line[0][0])
            out[-1][1].append(line[1][0])
    # out = [ [ids], [bbs], [frame_id] ]              - new dets
    # list_frames = [ frame_id, [ [ids], [bbs] ] ]    - all dets
    return out, list_frames
#####################
# Approach 1: Registration suppress
def reg_sup(frame_size, cur_frame, init_distribution, w_margin=0, h_margin=0):
    registration = 0
    adm_frame = [[],[],cur_frame[2]]
    for idx,bb in enumerate(cur_frame[1]):
        centbb = tlwh_xywh(bb)
        if centbb[0]<=(init_distribution[0]+w_margin) or \
            centbb[0]>=(frame_size[0]-init_distribution[0]-w_margin) or \
            centbb[1]<=(init_distribution[1]+h_margin) or \
            centbb[1]>=(frame_size[1]-init_distribution[1]-h_margin):
            # register valid birth.
            registration += 1
            adm_frame[0].append(cur_frame[0][idx])
            adm_frame[1].append(bb)
    if registration == 0:
        adm_frame = None
    # print(adm_frame)
    return registration, adm_frame
# Approach 2: IoU suppress
def iou_sup(bbs_cur, bbs_last, iou_threshold):
    list_comb = []
    # [idx, max_iou], ...
    for bb in bbs_cur[1]:
        comb = []
        for bb2 in bbs_last[1][1]:
            comb.append(bb_intersection_over_union(bb, bb2))
        if max(comb)<iou_threshold:
            list_comb.append([bbs_cur[1].index(bb), max(comb)])
            # print("####")
    # print(list_comb)
    return list_comb

############ temp visualization tools
def show_region(img_path, distribution, padding_wh, opa, w_h_marg):
    bground = cv2.imread(img_path)
    h, w, l = img.shape
    color = (255,0,255)
    thickness = -1

    bground = np.zeros((h+padding_wh[1]*2,w+padding_wh[0]*2,3), np.uint8)
    bground = cv2.rectangle(bground, (padding_wh[0],padding_wh[1]), (w+padding_wh[0],h+padding_wh[1]), (255,0,0), 2)
    o_lay = bground.copy()
    cv2.rectangle(o_lay, (padding_wh[0]+int(distribution[0]+w_h_marg[0]),padding_wh[1]+int(distribution[1]+w_h_marg[1])), (w+padding_wh[0]-int(distribution[0]+w_h_marg[0]),h+padding_wh[1]-int(distribution[1]+w_h_marg[1])), (255,0,255), -1)
    bground = cv2.addWeighted(o_lay, opa, bground, 1-opa, 0)
    out_text = out_text = "soft margin width: "+str(w_h_marg[0])+"; soft margin height: "+str(w_h_marg[1])
    bground = cv2.putText(bground, out_text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    y,x,_ = bground.shape
    print(w,h,x,y)

    cv2.imwrite("./img_vdo/1.jpg", bground)

if __name__ == '__main__':

    c_dir = "MOT16-04/"
    c_dir = "/content/drive/MyDrive/dataset/MOT16/train/train/" + c_dir + "img1/000001.jpg"

    # img_to_vdo(c_dir)
    
    gt_path = "MOT16-04/"
    gt_path = "/content/drive/MyDrive/dataset/MOT16/train/train/" + gt_path + "gt/gt.txt"

    byte_path = "/content/drive/MyDrive/P_Count/PTC_PC/YOLOX_outputs/yolox_nano_mix_det/track_vis/2022_12_11_09_19_16.txt"

    gt_c = False


    results, all_dets = gt_count(byte_path, gt=gt_c)

    total_count = 0
    dis_w,dis_h = w_h_mean(results[0][1])
    distribution = [dis_w, dis_h]

    # ellipse_demo(results, c_dir, distribution, gt=gt_c, w_h_marg=[0,-distribution[1]/2])

    ######### PC CODING
    # total_count = 0
    # dis_w,dis_h = w_h_mean(results[0][1])
    # distribution = [dis_w, dis_h]

    img = cv2.imread(c_dir)
    h, w, _ = img.shape
    frame_size = [w,h]

    print("###\n",all_dets[0],"\n###")

    print(distribution)
    print(frame_size)
    print(len(results[0][0]))

    soft_w = 0 # 0
    soft_h = -distribution[1]/2 # 0, 0 # -distribution[1]/2

    for idx,i in enumerate(results):
        if idx == 0:
            total_count += len(i[0])
        else:
            reg_sup_num, frame_a = reg_sup(frame_size, i, distribution, w_margin=soft_w, h_margin=soft_h)
            # total_count += reg_sup_num
            if frame_a is not None:
                x = iou_sup(frame_a, all_dets[idx-1], 0.33)
                total_count += len(x)
            
    
    print("Total Pedestrain seen in 1050 frames: ", total_count)

    print("writing regions: ")
    padding_wh_size = [400, 200]
    show_region(c_dir, distribution, padding_wh_size, 0.6, [soft_w, soft_h])




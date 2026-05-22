import glob
import cv2
import numpy as np
import math

def img_to_vdo(c_dir):

    images = sorted(glob.glob(c_dir))
    print(len(images))
    img_array = []
    for i in images:
        # print(i)
        img = cv2.imread(i)
        h, w, l = img.shape
        size = (w, h)
        img_array.append(img)
    
    print("start writing")

    out = cv2.VideoWriter('./videos/mot16-04.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

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
        
    # print(len(df))
    
    frame_wise = {} # {1:[ids], 2:[ids], ...}

    # line = [frame, id, tlwh, 1, 1, vis]
    for line in df:
        # print(line)
        if int(line[0]) not in frame_wise:
            frame_wise[int(line[0])] = [[int(float(line[1]))],[[int(float(line[2])),int(float(line[3])),int(float(line[4])),int(float(line[5]))]]]
        else:
            frame_wise[int(line[0])][0].append(int(line[1]))
            frame_wise[int(line[0])][1].append([int(float(line[2])),int(float(line[3])),int(float(line[4])),int(float(line[5]))])
    
    # count = 0
    # b_c = 0
    # for i in frame_wise:
    #     count += len(frame_wise[i][0])
    #     b_c += len(frame_wise[i][1])
    # print("id count = ", count)
    # print("b_c = ", b_c)

    list_frames = []

    for i in frame_wise:
        list_frames.append([i, frame_wise[i]])
    
    # for i in list_frames:
    #     print(i)

    # print(list_frames) # [ [frame, [[ids],[bbs]]], ~ ]

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

    print("#######################")
    # for i in out:
    #     print(i)

    return out, list_frames

def tlwh_xywh(bb):
    out = [int(bb[0]+bb[2]/2), int(bb[1]+bb[3]/2), bb[2], bb[3]]
    return out

def draw_single(img, cent, w_h, alpha):
    overlay = img.copy()
    cv2.ellipse(overlay, cent, w_h, 0, 0, 360, (255, 0, 255), -1)
    out = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    return out

def ellipse_demo(lists, sample_img_path, distribution, opa=0.6, gt=True, padding_wh=[400,200], w_h_marg=[0,0]):
    img = cv2.imread(sample_img_path)
    h, w, l = img.shape
    color = (255,0,255)
    thickness = -1

    img_array = []
    hist_c = 0

    for frame in lists:
        # print(frame[0])
        hist_c += len(frame[0])
        out_text = "Frame: "+str(frame[2][0])+"; Hist pedestrain count: "+str(hist_c)
        bground = np.zeros((h+padding_wh[1]*2,w+padding_wh[0]*2,3), np.uint8)
        bground = cv2.rectangle(bground, (padding_wh[0],padding_wh[1]), (w+padding_wh[0],h+padding_wh[1]), (255,0,0), 2)
        o_lay = bground.copy()
        cv2.rectangle(o_lay, (padding_wh[0]+int(distribution[0]+w_h_marg[0]),padding_wh[1]+int(distribution[1]+w_h_marg[1])), (w+padding_wh[0]-int(distribution[0]+w_h_marg[0]),h+padding_wh[1]-int(distribution[1]+w_h_marg[1])), (255,0,255), -1)
        bground = cv2.addWeighted(o_lay, opa, bground, 1-opa, 0)
        bground = cv2.putText(bground, out_text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        img = None
        for i,newbb in enumerate(frame[1]):
            bb_cent = tlwh_xywh(newbb)
            bb_cent = [bb_cent[0]+padding_wh[0], bb_cent[1]+padding_wh[1], bb_cent[2], bb_cent[3]]
            if i==0:
                img = draw_single(bground, (bb_cent[0], bb_cent[1]), (bb_cent[2], bb_cent[3]), opa)
            else:
                img = draw_single(img, (bb_cent[0], bb_cent[1]), (bb_cent[2], bb_cent[3]), opa)
        img_array.append(img)
    
    out_name = './byte_track_birth.mp4' if gt==False else './gt_track_birth.mp4'

    out = cv2.VideoWriter(out_name,cv2.VideoWriter_fourcc(*'mp4v'), 1, (w+padding_wh[0]*2, h+padding_wh[1]*2))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

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

def pedestrain_count(frame_size, cur_frame, init_distribution, w_margin=0, h_margin=0):
    # frame_size = [w,h]
    # init_distribution = [w,h]
    # cur_frame = [ [ids], [bbs], [cur_frame] ]
    registration = 0
    adm_ids = []
    # print("###",cur_frame)
    for bb in cur_frame[1]:
        centbb = tlwh_xywh(bb)
        if centbb[0]<=(init_distribution[0]/2+w_margin) or \
            centbb[0]>=(frame_size[0]-init_distribution[0]/2-w_margin) or \
            centbb[1]<=(init_distribution[1]/2+h_margin) or \
            centbb[1]>=(frame_size[1]-init_distribution[1]/2-h_margin):
            # register valid birth.
            registration += 1
            # print(cur_frame)
            # print(cur_frame, bb)
            adm_ids.append(cur_frame[0][cur_frame[1].index(bb)])
    # print(adm_ids)
    return registration, adm_ids

##################

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

def find_nn(bbs_cur, bbs_last, iou_threshold):
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


# def iou_suppress_k_frame(cur_frame, last_frame):
#     # tlwh
#     registration = 0
#     for bb in cur_frame[1]:


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

    print(distribution)
    print(frame_size)
    print(len(results[0][0]))

    for idx,i in enumerate(results):
        if idx == 0:
            total_count += len(i[0])
        else:
            total_count += len(find_nn(i, all_dets[idx-1], 0.3333333333333))
            # t_c, ids = pedestrain_count(frame_size, 
            #                                 i, 
            #                                 distribution)#,
            #                                 # w_margin=0, 
            #                                 # h_margin=-distribution[1]/1.6)
            # total_count += t_c
    
    print("Total Pedestrain seen in 1050 frames: ", total_count)




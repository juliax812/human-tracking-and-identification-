import glob
import cv2
import numpy as np
import math
# import misc.counter as counter
import utils.io_count as counter
import utils.bound_count as counter3

def gt_count(path, gt=True):
    with open(path, 'r') as f:
        lines = f.readlines()
    print("Total number of records: ", len(lines))

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
            frame_wise[int(line[0])][0].append(int(line[1]))
            frame_wise[int(line[0])][1].append([int(float(line[2])),int(float(line[3])),int(float(line[4])),int(float(line[5]))])

    list_frames = []

    for i in frame_wise:
        list_frames.append([i, frame_wise[i]])

    # list_frames = [ frame_id, [ [ids], [bbs] ] ]    - all dets
    return list_frames


if __name__ == "__main__":

    # byte_trk_path = '/content/drive/MyDrive/P_Count/PTC_PC/YOLOX_outputs/yolox_nano_mix_det/track_vis/2023_02_11_05_56_33.txt'
    byte_trk_path = "./dataset/2023_02_11_05_56_33.txt"
    gt_file = False
    gt_path = None
    approach = 2 # [line=1, bound=2]
    cleaning = True

    w,h = 1920, 1080
    margin = 120

    frame_wise = gt_count(byte_trk_path, gt=False)

    # print(len(frame_wise[0][1][0]))
    print("\nLast frane trklets info: ", frame_wise[-1],"\n")

    if approach == 1: # IO-Count
        # argparser
        p1, p2 = [0, int(1080/3+108)], [1920, int(1080/3+108-1)]

        # slope, bias, m_in = counter.get_hypers(p1,p2,frame_wise[0][1][1])
        p_count = counter.IOCount(p1, p2, [], margin)

        print("\nSlope = ",p_count.slope,"\nOriginal bias = ",p_count.bias)

        print("\nFeeding frames...")
        for trk in frame_wise:
            p_count.update(trk[1][0], trk[1][1])
            p_count.collect()
            # print(len(p_count.states))

        if cleaning:
            p_count.finalize()

        print("\nHistorical Pedestrian Counts: ")
        print("trespassing @ A:",p_count.lCount_tres_a)
        print("trespassing @ B:",p_count.lCount_tres_b)
        print("From B to A:    ",p_count.sCount_2a)
        print("From A to B:    ",p_count.sCount_2b,"\n")
    else:
        p_count = counter3.BndCount(w,h,margin)
        p_count.suppression_rate()

        print("Feeding frames...\n")
        for trk in frame_wise:
            p_count.update(trk[1][0], trk[1][1])
            p_count.collect()
        
        if cleaning:
            p_count.finalize()
        
        print("Historical Pedestrian Counts:")
        print("People entering: ", p_count.enter)
        print("People leaving: ", p_count.leave)
        print("side in: ", p_count.side_in)
        print("side out: ", p_count.side_out)
        print("ID switch: ", p_count.remain)



        












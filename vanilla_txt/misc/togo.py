# Zhang Ziliang - Pedestrian Count
import math

'''
Approach:
    1. Only track regions within 2 lines (single line extended with a small region) - margin 1
    2. only track regions within 4 lines (extra evidence) - margin 2

Practice:
    1. Filter all dets, remain region_dets_4_lines, find incoming position_2_lines (compare previous state)
    2. Passing 4_line_region to trker
    3. Filtering conditions based on 2 lines based on trker states

L2 space:
    1. Top left = 0,0
    2. going right = + (ideal)
    3. going down = - (ideal)
    4. apply transformation to incoorporate y = ax + b in defined space
'''


# given critical line's 1) slope and 2) intersection, within defined space.
def biases(slope, bias, margin_inner, margin_outer):
    y_angle = math.atan(abs(bias/slope)/abs(bias))
    marg_in = abs(margin_inner)/abs(math.sin(y_angle))
    inner_top = bias+marg_in
    inner_bot = bias-marg_in

    marg_out = abs(margin_outer+margin_inner)/abs(math.sin(y_angle))
    outer_top = bias+marg_out
    outer_bot = bias-marg_out
    return outer_top, inner_top, inner_bot, outer_bot

# inner &/ outer region filter for:
#       1) conditioning trks
#       2) conditioning counts
# return: [bbs] within region in between b1 & b2.
# def bb_filter(bbs, slope, b_min, b_max):
#     # assuming tlwh form
#     in_trk = [], out_trk = []
#     for bb in bbs:
#         # conditions
#         cond1 = ((bb[0]+bb[2]/2)*slope+b_min) < bb[1]+bb[3]/2 and bb[1]+bb[3]/2 < ((bb[0]+bb[2]/2)*slope+b_max)
#         # cond2 = ((bb[0]+bb[2]/2)*slope+b1) > bb[1]+bb[3]/2 and bb[1]+bb[3]/2 > ((bb[0]+bb[2]/2)*slope+b2)
#         if cond1: # or cond2:
#             in_trk.append(bb)
#         else:
#             out_trk.append(bb)
#     return in_trk, out_trk


# 2 classes:
# strict count class
# loose count class


# tracklet info = [id, centx, centy]
class LCount(object):
    def __init__(self, slope, bias, m1, m2): # m1=margin_in, m2=margin_out
        # a set of bbs across K frames [t+1, t, t-1, etc.]
        self.cur_id = []
        self.cur_bb = []
        self.flags = []
        # K value (frame-wise)
        self.memoryL = 30
        # margin biases, slope
        self.slope = slope
        self.bias = bias
        self.out0, self.in0, self.in1, self.out1 = biases(slope, bias, m1, m2)
        # marking states
        self.L_in, self.L_out = [], []
        self.iL_in = []
        # self.dismiss = []
    
    # marker encoding (bb_filter):
    def enflagging(self, bb):
        cond_L1 = ((bb[0]+bb[2]/2)*self.slope+self.out1) < bb[1]+bb[3]/2 and \
        bb[1]+bb[3]/2 < ((bb[0]+bb[2]/2)*self.slope+self.out0)
        # cond2 = ((bb[0]+bb[2]/2)*slope+b1) > bb[1]+bb[3]/2 and bb[1]+bb[3]/2 > ((bb[0]+bb[2]/2)*slope+b2)
        cond_L2_a = ((bb[0]+bb[2]/2)*self.slope+self.out1) >= bb[1]+bb[3]/2
        if cond_L1: # or cond2:
            out = Flag.L1
        elif cond_L2_a:
            out = Flag.L2_a
        else:
            out = Flag.L2_b
        return out
    # marker decoding:
    def deflagging(self, bb):
        pass


    # register 1 frame information (trk states) [ [id, bb4], [id, bb4], [id, bb4] ], [etc], [etc]
    def update(self, ids, bbs):
        new_flags = []
        if len(self.cur_id) == self.memoryL:
            self.cur_id.pop(-1)
            self.cur_bb.pop(-1)
            self.flags.pop(-1)
        self.cur_id.insert(0,ids)
        self.cur_bb.insert(0,bbs)
        for bb in bbs:
            flag = self.enflagging(bb)
            new_flags.append()
    
    def search(self, trks): # within L2 region
        # tracklet search:
        L1, L2 = bb_filter(trks, self.slope, self.out1, self.out0)
        L1_id, L2_id = list(list(zip(*L1))[0]), list(list(zip(*L2))[0])
        for inner in L1:
            if inner[0] in self.cur_id:
                pass
                # TBC




class Flag(object):
    L1 = 0  # within inner region
    L2_a = 1  # side a => within outer region, outside of inner region
    L2_b = 2  # side b


class SCount(object):
    def __init__(self, args, frame_rate=30):
        self.frame_id = 0
        self.args = args









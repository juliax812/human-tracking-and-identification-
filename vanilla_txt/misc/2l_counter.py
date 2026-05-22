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

# p1/p2 in real space
def get_hypers(p1, p2, bbs, m1=None, m2=None):
    slope = (-p2[1]+p1[1])/(p2[0]-p1[0])
    bias = -p1[1]-slope*p1[0]
    width_total = []
    if m1 is None:
        for bb in bbs:
            width_total.append(bb[2])
        m_in = sum(width_total)/len(width_total)
        m_out = m_in/2
    else:
        m_in = m1
        m_out = m2
    return slope, bias, m_in, m_out

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

class Flag(object):
    # encode
    L1 = 0  # within inner region
    L2_a = 1  # side a => within outer region, outside of inner region
    L2_b = 2  # side b
    # decode
    L2_a_L1 = 3
    L2_b_L1 = 4
    L1_L1 = 5
    no_L1 = 6

    L1_L2_a = 7
    # L2_a_L2_a = None
    # L2_b_L2_a = None
    # no_L2_a = None

    L1_L2_b = 8
    # L2_b_L2_b = None
    # L2_a_L2_b = None
    # no_L2_b = None


# tracklet info = [id, centx, centy]
class LCount(object):
    def __init__(self, slope, bias, m1, m2): # m1=margin_in, m2=margin_out
        # a set of bbs across K frames [t+1, t, t-1, etc.]
        self.ids = []
        self.flags = []
        self.states = {}
        # K value (frame-wise)
        self.memoryL = 30
        # margin biases, slope
        self.slope = slope
        self.bias = bias
        self.out0, self.in0, self.in1, self.out1 = biases(slope, bias, m1, m2)
        # marking states
        self.lCount_tres = 0 # level 0 - L2-side-trespassing
        self.lCount_van = 0   # level 1 - L1_L1
        self.lCount_ab = 0    # level 2 - L1-L2
        self.lCount_los = 0   # level 3 - L2-L1
        self.sCount_a = 0     # level 4 - L2-a-L2-b
        self.sCount_b = 0     # level 5 - L2-b-L2-a
        # self.dismiss = []
    
    def match_case_L1(self,f_p):
        if f_p == Flag.L2_a:
            # legal initiation 1
            return Flag.L2_a_L1
        elif f_p == Flag.L2_b:
            # legal initiation 2
            return Flag.L2_b_L1
        elif f_p == Flag.L1:
            # assume registered
            return Flag.L1_L1
        else:
            # illegal initiation = f_p not exists
            return Flag.no_L1

    def match_case_L2a(self,f_p):
        if f_p == Flag.L1:
            # legal completion 2
            return Flag.L1_L2_a
        else:
            # dismiss
            return None
    
    def match_case_L2b(self,f_p):
        if f_p == Flag.L2_a:
            # legal completion 1
            return Flag.L1_L2_b
        else:
            # dismiss
            return None

    # marker encoding (bb_filter):
    def enflagging(self, bb):
        cond_L1 = ((bb[0]+bb[2]/2)*self.slope+self.out1) < -bb[1]-bb[3]/2 and \
        -bb[1]-bb[3]/2 < ((bb[0]+bb[2]/2)*self.slope+self.out0)
        # cond2 = ((bb[0]+bb[2]/2)*slope+b1) > bb[1]+bb[3]/2 and bb[1]+bb[3]/2 > ((bb[0]+bb[2]/2)*slope+b2)
        cond_L2_a = ((bb[0]+bb[2]/2)*self.slope+self.out1) >= -bb[1]-bb[3]/2
        if cond_L1: # or cond2:
            out = Flag.L1
        elif cond_L2_a:
            out = Flag.L2_a
        else:
            out = Flag.L2_b
        return out
    # marker decoding:
    def deflagging(self):
        hist_id = [item for sublist in self.ids[1:] for item in sublist]
        hist_flag = [item for sublist in self.flags[1:] for item in sublist]
        if len(hist_id) == 0:
            return
        for idx1,p in enumerate(self.ids[0]):
            f_cur = self.flags[0][idx1]
            state = None  # transition state (not row state)
            for idx2,q in enumerate(hist_id):
                if f_cur == Flag.L1:
                    # inner region
                    if p == q:
                        state = self.match_case_L1(hist_flag[idx2])
                    break
                elif f_cur == Flag.L2_a:
                    # outer region a
                    if p == q:
                        state = self.match_case_L2a(hist_flag[idx2])
                    break
                elif f_cur == Flag.L2_b:
                    # outer region b
                    if p == q:
                        state = self.match_case_L2b(hist_flag[idx2])
                    break
            # information check:
            if state is None:
                continue
            else:
                # activation based on different transition states:
                # L2_a_L1 = 3 in
                # L2_b_l1 = 4 in
                # no_L1 = 6   in
                # L1_L1 = 5   trk
                # L1_L2_a = 7 out
                # L1_L2_b = 8 out
                
                # registration - exit condition
                if p in self.states.copy():
                    if state == Flag.L1_L1:
                        continue
                    elif state == Flag.L1_L2_a:
                        if self.states[p][0] == Flag.L2_a_L1:
                            # delete, continue, trespassing
                            del self.states[p]
                            self.lCount_tres += 1
                        elif self.states[p][0] == Flag.L2_b_L1:
                            # completed SCount
                            del self.states[p]
                            self.sCount_b += 1
                        elif self.states[p][0] == Flag.no_L1:
                            # completed LCount
                            del self.states[p]
                            self.lCount_ab += 1
                    elif state == Flag.L1_L2_b:
                        if self.states[p][0] == Flag.L2_b_L1:
                            # delete, continue, trespassing 
                            del self.states[p]
                            self.lCount_tres += 1
                        elif self.states[p][0] == Flag.L2_a_L1:
                            # completed SCount
                            del self.states[p]
                            self.sCount_a += 1
                        elif self.states[p][0] == Flag.no_L1:
                            # completed LCount
                            del self.states[p]
                            self.lCount_ab += 1
                # initiation - entry condition
                else:
                    self.states[p] = [state,0]
        
        # delete counter exceeding memory size
        for i in self.states.copy():
            if self.states[i][1] >= self.memoryL:
                if self.states[i][0] == Flag.no_L1:
                    self.lCount_van += 1
                else:
                    self.lCount_los += 1
                # if i in self.states:
                del self.states[i]
        # forwarding 1 time size on counter
        for i in self.states:
            self.states[i][1] += 1

    # register 1 frame information (trk states) [ [id, bb4], [id, bb4], [id, bb4] ], [etc], [etc]
    def update(self, ids, bbs):
        new_flags = []
        if len(self.ids) == self.memoryL:
            self.ids.pop(-1)
            self.flags.pop(-1)
        for bb in bbs:
            new_flags.append(self.enflagging(bb))
        self.ids.insert(0,ids)
        self.flags.insert(0,new_flags)
    

    

if __name__ == '__main__':
    print("Pedestrian Count.")
    slope, bias, m1, m2 = 0, 0, 0, 0

    # initiate Counter
    p_count = LCount(slope, bias, m1, m2)


    # starting frame

    # get detections

    # filter detections based on m2

    # passing all filtered dets to tracker

    # start counting
    trks = []

    for trk in trks:
        ids = []
        bbs = []
        p_count.update(ids, bbs)
        p_count.deflagging()
    
    print(p_count.lCount_tres)
    print(p_count.lCount_van)
    print(p_count.lCount_ab)
    print(p_count.lCount_los)
    print(p_count.sCount_a)
    print(p_count.sCount_b)

    # TBC = data processing + experiments on MOT16-04-nano-byte







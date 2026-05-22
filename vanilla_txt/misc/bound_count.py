# Pedestrian Count (bound-wise) - ZZL
# import math

'''
Bounding Approach:
  1) reduce info passed to trker from detector, to reduce computation cost (exp?)
  2) idealy reduce half of bbs (-50%)
  3) Side-A, Side-B, side-C, side-D
  4) ignore center region trks (det phase)
'''


class Flag(object):
    A,B,C,D = 0, 1, 2, 3
    Center = 4
    init = 5

class BndCount(object):
    
    def __init__(self, w, h, margin, keep_lost=10):
        self.memoryL = 30*keep_lost
        self.margin = margin
        self.frame_width, self.frame_height = w,h
        self.states = {}

        self.enter, self.leave = 0, 0       # 2 (higher = more reliable)
        self.side_in, self.side_out = 0, 0  # 1
        self.remain = 0                     # 0
    
    def suppression_rate(self):
        sprs = (self.frame_width-4*self.margin)*(self.frame_height-4*self.margin)/(self.frame_width*self.frame_height)*100
        print("suppression rate: ", sprs, "%\n")

    def encode(self, bb):
        centx = bb[0]+bb[2]/2
        centy = bb[1]+bb[3]/2
        if centx < self.margin and centy < self.frame_height-self.margin:
            return Flag.A
        elif centx > self.margin and centy < self.margin:
            return Flag.B
        elif centx > self.frame_width-self.margin and centy > self.margin:
            return Flag.C
        elif centx < self.frame_width-self.margin and centy > self.frame_height-self.margin:
            return Flag.D
        else:
            return Flag.Center
    
    def decode(self, state):
        if state[0] == state[1] or state[1]==Flag.init:
            self.remain += 1
        elif state[0] in [Flag.A, Flag.B, Flag.C, Flag.D]:
            if state[1] != Flag.Center and (state[1]-state[0])%2 == 0:
                self.enter += 1
                self.leave += 1
            elif state[1] in [Flag.A, Flag.B, Flag.C, Flag.D]:
                self.side_in += 1
                self.side_out += 1
            else:
                self.enter += 1
        else:
            self.leave += 1
            
    def update(self, ids, bbs):
        for idx,i in enumerate(ids):
            if i in self.states:
                self.states[i][1] = self.encode(bbs[idx])
                self.states[i][2] = 0
            else:
                self.states[i] = [self.encode(bbs[idx]),Flag.init,0]
    
    def collect(self):
        for i in self.states.copy():
            self.states[i][2] += 1
            if self.states[i][2] >= self.memoryL:
                self.decode(self.states[i])
                del self.states[i]
    
    def finalize(self):
        for i in self.states:
            self.decode(self.states[i])













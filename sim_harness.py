"""
sim_harness — drive the whole app headlessly, with no camera.

The agent working on this repo has no camera access (cv2.VideoCapture(0)
returns "not authorized"; that prompt only goes to the user).  This harness
monkeypatches OpenCV before importing `main`, feeds synthetic frames and
scripted hand landmarks, and writes rendered frames to simout/ so changes
can actually be looked at.

    python3 sim_harness.py

It exercises the real code path end to end: lasso -> countdown -> capture ->
email typed key-by-key -> ENTER -> written to captures/.

CAVEAT: the synthetic face is not a real face.  Good for logic, layout,
geometry and regressions; NOT a substitute for tuning the portrait look.
For that, get a real frame (press S in the app -> screenshots/).
"""

import cv2, numpy as np, math, types, sys, time
from pathlib import Path
OUT=Path(__file__).parent / "simout"
OUT.mkdir(exist_ok=True)
H,W=720,1280
def room():
    im=np.full((H,W,3),150,np.uint8); im[:250]=196
    for gx in range(0,W,120): cv2.line(im,(gx,0),(gx-140,250),(168,168,166),2)
    for gy in range(0,250,52): cv2.line(im,(0,gy),(W,gy),(168,168,166),2)
    cv2.rectangle(im,(120,300),(430,700),(196,194,188),-1)
    for wx in range(470,900,90): cv2.rectangle(im,(wx,300),(wx+70,560),(205,208,206),3)
    cv2.rectangle(im,(60,180),(210,260),(240,240,235),-1)
    return cv2.GaussianBlur(im,(0,0),1.2)
BASE=room()
def head(im,cx,cy,s):
    cv2.ellipse(im,(cx,cy+int(s*2.3)),(int(s*2.0),int(s*1.7)),0,180,360,(78,80,92),-1)
    cv2.ellipse(im,(cx,cy),(int(s*0.92),int(s*1.20)),0,0,360,(188,168,150),-1)
    ov=im.copy(); cv2.ellipse(ov,(cx-int(s*0.35),cy),(int(s*0.55),int(s*1.0)),0,0,360,(206,188,170),-1)
    cv2.addWeighted(ov,0.45,im,0.55,0,im)
    cv2.ellipse(im,(cx,cy-int(s*0.88)),(int(s*1.04),int(s*0.78)),0,180,360,(42,34,32),-1)
    for k in (-1,1):
        ex=cx+k*int(s*0.40)
        cv2.ellipse(im,(ex,cy-int(s*0.12)),(int(s*0.20),int(s*0.11)),0,0,360,(242,240,236),-1)
        cv2.circle(im,(ex,cy-int(s*0.12)),int(s*0.085),(52,42,40),-1)
        cv2.circle(im,(ex,cy-int(s*0.12)),int(s*0.035),(16,14,14),-1)
        cv2.ellipse(im,(ex,cy-int(s*0.12)),(int(s*0.27),int(s*0.20)),0,0,360,(40,34,34),3)
        cv2.ellipse(im,(ex,cy-int(s*0.40)),(int(s*0.24),int(s*0.07)),0,180,360,(58,46,42),-1)
    cv2.line(im,(cx-int(s*0.07),cy-int(s*0.12)),(cx+int(s*0.07),cy-int(s*0.12)),(40,34,34),3)
    cv2.ellipse(im,(cx,cy+int(s*0.22)),(int(s*0.17),int(s*0.32)),0,0,360,(172,150,134),-1)
    cv2.ellipse(im,(cx,cy+int(s*0.70)),(int(s*0.31),int(s*0.13)),0,0,180,(122,84,82),-1)
    return im
CX,CY,S=760,380,105
def make_frame(): return head(BASE.copy(),CX,CY,S)
class C:
    def __init__(s,*a,**k): pass
    def set(s,*a): return True
    def isOpened(s): return True
    def read(s): return True, make_frame()
    def release(s): pass
cv2.VideoCapture=C; cv2.namedWindow=lambda *a,**k:None; cv2.resizeWindow=lambda *a,**k:None
fr=[]; cv2.imshow=lambda n,i: fr.append(i.copy())
EMAIL="dwight@dundermifflin.com"
script=[255]*150+[ord(c) for c in EMAIL]+[13]+[255]*20+[ord('q')]
K=[0]
def wk(d):
    i=K[0]; K[0]+=1
    if i>=len(script): sys.exit("overran")
    time.sleep(0.012)          # let the wall-clock countdown actually elapse
    return script[i]
cv2.waitKey=wk; cv2.destroyAllWindows=lambda:None
import main, face_ascii
MCX=W-CX          # the app mirrors the frame before tracking
BOX=(MCX-S,CY-int(S*1.2),2*S,int(S*2.4))
class T:
    confirmed=True; id=1; last_seen=0
    box=np.array(BOX,np.float32)
    @property
    def rect(s): return tuple(int(v) for v in s.box)
    @property
    def center(s): x,y,w,h=s.box; return (x+w/2,y+h/2)
_t=T()
class FT:
    backend="stub"; raw_count=1
    def __init__(s,*a,**k): s.manual=[]; s.tracks=[_t]
    def update(s,f): return s.tracks
    def add_manual(s,b,f):
        m=face_ascii._ManualTrack(99,b,f); s.manual.append(m); return m
    def drop_manual(s,i): s.manual=[m for m in s.manual if m.id!=i]
    def get(s,i): return _t if i==1 else next((m for m in s.manual if m.id==i),None)
    @property
    def visible(s): return [_t]
    @property
    def lockable(s): return [_t]+s.manual
main.FaceTracker=FT
class LM:
    __slots__=("x","y","z")
    def __init__(s,x,y): s.x,s.y,s.z=float(x),float(y),0.0

def gesture_landmarks(raised, thumb=False, angle=0.0):
    """Build a synthetic hand and rotate it to test gesture invariance."""
    wrist = np.array((0.50, 0.55), np.float32)
    pts = [wrist.copy() for _ in range(21)]
    fingers = (
        (main.INDEX_MCP, main.INDEX_PIP, main.INDEX_DIP, main.INDEX_TIP, -0.060, "i"),
        (main.MIDDLE_MCP, main.MIDDLE_PIP, main.MIDDLE_DIP, main.MIDDLE_TIP, -0.020, "m"),
        (main.RING_MCP, main.RING_PIP, main.RING_DIP, main.RING_TIP, 0.025, "r"),
        (main.PINKY_MCP, main.PINKY_PIP, main.PINKY_DIP, main.PINKY_TIP, 0.065, "p"),
    )
    for mcp, pip, dip, tip, x, name in fingers:
        pts[mcp] = wrist + (x, -0.10)
        pts[pip] = wrist + (x, -0.23)
        if name in raised:
            pts[dip] = wrist + (x, -0.34)
            pts[tip] = wrist + (x, -0.47)
        else:
            pts[dip] = wrist + (x, -0.14)
            pts[tip] = wrist + (x, -0.08)
    pts[main.THUMB_IP] = wrist + (-0.05, -0.06)
    pts[main.THUMB_TIP] = wrist + ((-0.25, -0.08) if thumb else (-0.04, -0.07))

    c, s = math.cos(angle), math.sin(angle)
    rot = np.array(((c, -s), (s, c)), np.float32)
    return [LM(*(wrist + rot @ (p - wrist))) for p in pts]

for expected, raised, thumb in (
        ("point", "i", False),
        ("peace", "im", False),
        ("three", "imr", False),
        ("palm", "imrp", True),
        ("fist", "", False)):
    for angle in np.linspace(0, 2 * math.pi, 25):
        got = main.GestureEngine(threshold=0).detect(
            gesture_landmarks(raised, thumb, float(angle)))
        assert got == expected, (expected, got, angle)
print("[sim] gesture map passed through 360°")

class D:
    def __init__(s): s.i=-1
    def close(s): pass
    def detect_for_video(s,img,ts):
        s.i+=1; k=s.i-30
        if k<0 or k>70: return types.SimpleNamespace(hand_landmarks=[])
        a=2*math.pi*(k/70)-math.pi/2
        px,py=MCX+250*math.cos(a), CY+250*math.sin(a)
        pts={main.WRIST:(px,py+90),main.INDEX_MCP:(px,py+45),main.INDEX_PIP:(px,py+25),
             main.INDEX_DIP:(px,py+12),main.INDEX_TIP:(px,py)}
        for mcp,pip,dip,tip,dx in ((main.MIDDLE_MCP,main.MIDDLE_PIP,main.MIDDLE_DIP,main.MIDDLE_TIP,16),
                                   (main.RING_MCP,main.RING_PIP,main.RING_DIP,main.RING_TIP,30),
                                   (main.PINKY_MCP,main.PINKY_PIP,main.PINKY_DIP,main.PINKY_TIP,42)):
            pts[mcp]=(px+dx,py+45); pts[pip]=(px+dx,py+30)
            pts[dip]=(px+dx,py+42); pts[tip]=(px+dx,py+52)
        pts[main.THUMB_TIP]=(px-14,py+56); pts[main.THUMB_IP]=(px-8,py+66)
        lm=[LM(0.5,0.5) for _ in range(21)]
        for i,(x,y) in pts.items(): lm[i]=LM(x/W,y/H)
        return types.SimpleNamespace(hand_landmarks=[lm])
main.HandLandmarker=types.SimpleNamespace(create_from_options=lambda o: D())
# note which frames were countdown frames
CD=[]
_cd=face_ascii._countdown
def spy(out,s,now): CD.append(len(fr)); return _cd(out,s,now)
face_ascii._countdown=spy
main.main()
print(f"[sim] {len(fr)} frames; countdown drawn on {len(CD)} frames "
      f"({CD[0] if CD else '-'}..{CD[-1] if CD else '-'})")
cv2.imwrite(str(OUT/"s_detect.png"), fr[20])
if CD: cv2.imwrite(str(OUT/"s_count.png"), fr[CD[len(CD)//3]])
cv2.imwrite(str(OUT/"s_final.png"), fr[-1])

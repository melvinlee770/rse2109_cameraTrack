"""
Staging Zone Tracker v14 - Persistent Zone Memory
====================================================
Features:
- Robust multi-pass ArUco detection
- Partial detection (3/4 markers)
- Temporal smoothing + jump rejection
- ZONE LOCK: press 'l' to lock zone permanently
- ZONE SAVE: locked zone is saved to file automatically
- AUTO LOAD: zone is restored on next program start
- YOLO pallet detection + ArUco false positive filtering
- Utilization % calculation

Usage:
    python custom_track_v14.py --model best_v4.pt
    python custom_track_v14.py --model best_v4.pt --zone-file my_zone.npz
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import argparse
import os
import sys
import json
from ultralytics import YOLO

TAG_SIZE_CM=2.4; MODEL_PATH="best_v4.pt"; CONF_THRESH=0.5
MEM_TIMEOUT=5.0; SMOOTH_A=0.1; UTIL_A=0.2
MAX_JUMP_PX=30; ROLLING_N=10
ZONE_SAVE_FILE="zone_data.json"


# =============================================================================
# Zone Save / Load
# =============================================================================

def save_zone(filepath, zone_pts, ppm, real_area, pixel_area):
    """Save locked zone data to a JSON file."""
    data = {
        "zone_pts": zone_pts.tolist(),
        "ppm": float(ppm) if ppm else None,
        "real_area_m2": float(real_area) if real_area else None,
        "pixel_area": float(pixel_area) if pixel_area else None,
        "tag_size_cm": TAG_SIZE_CM,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Zone saved to: {filepath}")


def load_zone(filepath):
    """
    Load saved zone data from JSON file.

    Returns:
        zone_pts, ppm, real_area, pixel_area
        or (None, None, None, None) if file not found
    """
    if not os.path.exists(filepath):
        return None, None, None, None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        zone_pts = np.array(data["zone_pts"], dtype=np.float32)
        ppm = data.get("ppm")
        real_area = data.get("real_area_m2")
        pixel_area = data.get("pixel_area")

        print(f"[INFO] Zone loaded from: {filepath}")
        print(f"[INFO]   Saved at: {data.get('saved_at', 'unknown')}")
        print(f"[INFO]   Area: {real_area:.4f} m^2" if real_area else "[INFO]   Area: unknown")
        print(f"[INFO]   Zone will be shown automatically (LOCKED)")

        return zone_pts, ppm, real_area, pixel_area
    except Exception as e:
        print(f"[WARN] Could not load zone file: {e}")
        return None, None, None, None


def delete_zone_file(filepath):
    """Delete saved zone file."""
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"[INFO] Zone file deleted: {filepath}")


# =============================================================================
# Robust ArUco Detector
# =============================================================================

class RobustArucoDetector:
    def __init__(self, dt=aruco.DICT_4X4_50):
        self.ad=aruco.getPredefinedDictionary(dt); self.dets=[]
        p1=aruco.DetectorParameters()
        p1.adaptiveThreshWinSizeMin=3; p1.adaptiveThreshWinSizeMax=53
        p1.adaptiveThreshWinSizeStep=4; p1.minMarkerPerimeterRate=0.01
        p1.cornerRefinementMethod=aruco.CORNER_REFINE_SUBPIX
        self.dets.append(aruco.ArucoDetector(self.ad,p1))
        p2=aruco.DetectorParameters()
        p2.adaptiveThreshWinSizeMin=3; p2.adaptiveThreshWinSizeMax=73
        p2.adaptiveThreshWinSizeStep=2; p2.minMarkerPerimeterRate=0.005
        p2.adaptiveThreshConstant=12
        p2.cornerRefinementMethod=aruco.CORNER_REFINE_SUBPIX
        self.dets.append(aruco.ArucoDetector(self.ad,p2))

    def _prevars(self,g):
        vs=[g]
        cl=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)); vs.append(cl.apply(g))
        k=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]); vs.append(cv2.filter2D(g,-1,k))
        vs.append(cv2.GaussianBlur(g,(3,3),0))
        return vs

    def detect(self,gray):
        best_c=None; best_i=None; best_n=0
        for img in self._prevars(gray):
            for det in self.dets:
                c,i,_=det.detectMarkers(img); n=len(i) if i is not None else 0
                if n>best_n: best_n=n; best_c=c; best_i=i
                if n>=4: return best_c,best_i,best_n
        return best_c,best_i,best_n


# =============================================================================
# Zone Smoother
# =============================================================================

class ZoneSmoother:
    def __init__(self, alpha=0.1, max_jump=30, history_size=10):
        self.alpha=alpha; self.max_jump=max_jump
        self.sp=None; self.history=[]; self.hsize=history_size
        self.locked=False; self.lock_count=0; self.LOCK_AFTER=15

    def update(self, p):
        if self.sp is not None:
            max_dist=np.max(np.linalg.norm(p-self.sp, axis=1))
            if max_dist>self.max_jump and self.locked:
                return self.sp.copy()
        self.history.append(p.copy())
        if len(self.history)>self.hsize: self.history.pop(0)
        if len(self.history)>=3:
            avg=np.mean(self.history, axis=0).astype(np.float32)
        else:
            avg=p
        if self.sp is None: self.sp=avg.copy()
        else: self.sp=self.alpha*avg+(1-self.alpha)*self.sp
        self.lock_count+=1
        if self.lock_count>=self.LOCK_AFTER: self.locked=True
        return self.sp.copy()

    def reset(self): self.sp=None; self.history=[]; self.locked=False; self.lock_count=0


# =============================================================================
# Helpers
# =============================================================================

def predict_missing(known,kidx):
    miss=(set([0,1,2,3])-set(kidx)).pop()
    pm={0:(1,3,2),1:(0,2,3),2:(1,3,0),3:(0,2,1)}
    a,b,c=pm[miss]; pred=known[a]+known[b]-known[c]
    fp=np.zeros((4,2),dtype=np.float32)
    for i in kidx: fp[i]=known[i]
    fp[miss]=pred
    return fp,miss

def load_cal(f):
    if not os.path.exists(f): return None,None,None
    d=np.load(f); return d['camera_matrix'],d['dist_coeffs'],d['reprojection_error'][0]

def setup_undist(cm,dc,w,h):
    nc,roi=cv2.getOptimalNewCameraMatrix(cm,dc,(w,h),0,(w,h))
    m1,m2=cv2.initUndistortRectifyMap(cm,dc,None,nc,(w,h),cv2.CV_32FC1)
    return m1,m2

def order_pts(pts):
    r=np.zeros((4,2),dtype="float32"); s=pts.sum(axis=1)
    r[0]=pts[np.argmin(s)]; r[2]=pts[np.argmax(s)]
    d=np.diff(pts,axis=1); r[1]=pts[np.argmin(d)]; r[3]=pts[np.argmax(d)]
    return r

def get_centers(c): return np.array([np.mean(x[0],axis=0) for x in c])

def get_inner(corners):
    ac=np.array([np.mean(c[0],axis=0) for c in corners]); ct=np.mean(ac,axis=0)
    return np.array([c[0][np.argmin(np.linalg.norm(c[0]-ct,axis=1))] for c in corners],dtype=np.float32)

def get_ppm(corners,tag_cm):
    el=[]
    for mc in corners:
        p=mc[0]
        for i in range(4): el.append(np.linalg.norm(p[(i+1)%4]-p[i]))
    return (np.mean(el)/tag_cm)*100.0 if el else None

def get_areg(corners,mg=15):
    return [(int(c[0][:,0].min())-mg,int(c[0][:,1].min())-mg,
             int(c[0][:,0].max())+mg,int(c[0][:,1].max())+mg) for c in corners]

def bovr(box,reg):
    x1=max(box[0],reg[0]);y1=max(box[1],reg[1])
    x2=min(box[2],reg[2]);y2=min(box[3],reg[3])
    if x2<=x1 or y2<=y1: return 0.0
    return (x2-x1)*(y2-y1)/max((box[2]-box[0])*(box[3]-box[1]),1)

def fyolo(res,areg,mina=500,ovt=0.3,ct=0.5):
    fl=[]
    if res[0].boxes is None or len(res[0].boxes)==0: return fl
    bx=res[0].boxes
    for i in range(len(bx)):
        cf=float(bx.conf[i]);ci=int(bx.cls[i]);cn=res[0].names[ci]
        x1,y1,x2,y2=bx.xyxy[i].cpu().numpy()
        if cf<ct or (x2-x1)*(y2-y1)<mina: continue
        if any(bovr((x1,y1,x2,y2),r)>ovt for r in areg): continue
        fl.append((int(x1),int(y1),int(x2),int(y2),cf,ci,cn))
    return fl

def inzone(box,zp):
    if zp is None: return False
    cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
    return cv2.pointPolygonTest(zp.astype(np.float32).reshape((-1,1,2)),
                                (float(cx),float(cy)),False)>=0


# =============================================================================
# Utilization (direct mask — no bird's-eye)
# =============================================================================

def calc_util_direct(frame_shape, zone_pts, dets):
    """
    Calculate utilization using direct pixel masks.
    Zone mask + pallet mask → AND → percentage.
    """
    h, w = frame_shape[:2]
    zm = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(zm, [zone_pts.astype(np.int32).reshape((-1,1,2))], 255)
    za = cv2.countNonZero(zm)
    if za == 0: return 0.0, 0, 0, 0

    pm = np.zeros((h, w), dtype=np.uint8)
    nz = 0
    for d in dets:
        x1,y1,x2,y2 = d[0],d[1],d[2],d[3]
        if not inzone((x1,y1,x2,y2), zone_pts): continue
        nz += 1
        x1c=max(0,x1); y1c=max(0,y1); x2c=min(w,x2); y2c=min(h,y2)
        if x2c>x1c and y2c>y1c:
            cv2.rectangle(pm, (x1c,y1c), (x2c,y2c), 255, -1)

    oa = cv2.countNonZero(cv2.bitwise_and(pm, zm))
    return (oa/za)*100.0, za, oa, nz


# =============================================================================
# Visualization
# =============================================================================

def ucol(p):
    if p<30: return(0,255,0)
    elif p<50: return(0,255,255)
    elif p<70: return(0,165,255)
    elif p<90: return(0,0,255)
    return(0,0,200)

def ubar(f,p,x,y,bw=300,bh=22):
    c=ucol(p); cv2.rectangle(f,(x,y),(x+bw,y+bh),(50,50,50),-1)
    cv2.rectangle(f,(x,y),(x+bw,y+bh),(150,150,150),1)
    fw=int(bw*min(p,100)/100)
    if fw>0: cv2.rectangle(f,(x,y),(x+fw,y+bh),c,-1)
    t=f"{p:.1f}%";sz,_=cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
    cv2.putText(f,t,(x+(bw-sz[0])//2,y+(bh+sz[1])//2),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

def ddets(f,dets,zp):
    iz=oz=0
    for(x1,y1,x2,y2,cf,ci,cn) in dets:
        ins=inzone((x1,y1,x2,y2),zp); col=(0,165,255) if ins else (255,0,0)
        if ins: iz+=1
        else: oz+=1
        cv2.rectangle(f,(x1,y1),(x2,y2),col,2)
        lb=f"{cn} {cf:.2f}";ls,_=cv2.getTextSize(lb,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        cv2.rectangle(f,(x1,y1-ls[1]-8),(x1+ls[0]+4,y1),col,-1)
        cv2.putText(f,lb,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(f,"IN ZONE" if ins else "OUTSIDE",(x1,y2+18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,col,2)
    return iz,oz


# =============================================================================
# Main
# =============================================================================

def main():
    pa = argparse.ArgumentParser(description="Staging Zone v14 - Persistent Zone")
    pa.add_argument("--camera", type=int, default=0)
    pa.add_argument("--model", type=str, default=MODEL_PATH)
    pa.add_argument("--calib-file", type=str, default="camera_calibration.npz")
    pa.add_argument("--no-calib", action="store_true")
    pa.add_argument("--tag-size", type=float, default=TAG_SIZE_CM)
    pa.add_argument("--conf", type=float, default=CONF_THRESH)
    pa.add_argument("--memory-timeout", type=float, default=MEM_TIMEOUT)
    pa.add_argument("--zone-file", type=str, default=ZONE_SAVE_FILE,
                    help=f"File to save/load locked zone (default: {ZONE_SAVE_FILE})")
    args = pa.parse_args()

    # --- YOLO ---
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}"); sys.exit(1)
    print(f"[INFO] Loading YOLO: {args.model}")
    model = YOLO(args.model)
    print(f"[INFO] Classes: {model.names}")

    # --- ArUco ---
    rd = RobustArucoDetector(aruco.DICT_4X4_50)

    # --- Calibration ---
    ucal=False; m1=m2=cmx=dcc=None; cerr=0
    if not args.no_calib:
        cmx,dcc,cerr = load_cal(args.calib_file)
        if cmx is not None:
            ucal=True; print(f"[INFO] Calibrated (err:{cerr:.4f}px)")
        else:
            print("[WARN] No calibration file.")

    # --- Camera ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Camera {args.camera}"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    udi = False

    # --- State ---
    zs = ZoneSmoother(alpha=SMOOTH_A, max_jump=MAX_JUMP_PX, history_size=ROLLING_N)
    lvp=None; ldt=0; lppm=None; lra=None; lpa=0
    rm="CENTER"
    us=0.0
    ft=time.time(); fc=0; fps=0.0
    stats={"4/4":0, "3/4":0, "memory":0, "lost":0}
    zone_locked = False
    show_ui = True  # Toggle with 'h' key

    # --- Try to load saved zone ---
    saved_pts, saved_ppm, saved_area, saved_pa = load_zone(args.zone_file)
    if saved_pts is not None:
        lvp = saved_pts
        lppm = saved_ppm
        lra = saved_area
        lpa = saved_pa if saved_pa else 0
        zone_locked = True
        ldt = time.time()
        print(f"[INFO] Restored locked zone from previous session")

    print(f"[INFO] Tag:{args.tag_size}cm  Conf:{args.conf}  Timeout:{args.memory_timeout}s")
    print(f"[INFO] Zone file: {args.zone_file}")
    print(f"[INFO] Multi-pass:ON  Partial(3/4):ON  Smoothing:ON")
    print(f"[INFO] Controls:")
    print(f"  'l' = LOCK/UNLOCK zone (saves to file)")
    print(f"  'h' = HIDE/SHOW data overlay")
    print(f"  'q' = quit  'c' = ref mode  'r' = reset (clears saved zone)")
    print(f"  'u' = undistort  '+'/'-' = confidence  's' = snapshot")

    while True:
        ret, raw = cap.read()
        if not ret: break

        if ucal and not udi:
            h,w = raw.shape[:2]
            m1,m2 = setup_undist(cmx, dcc, w, h)
            udi = True

        frame = cv2.remap(raw,m1,m2,cv2.INTER_LINEAR) if ucal and m1 is not None else raw.copy()
        ct = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =============================================================
        # STAGE 1: ArUco detection
        # =============================================================
        ac, ai, nd = rd.detect(gray)
        areg = get_areg(ac) if ac else []
        st = "SEARCHING"; dq = "NONE"; pc = -1

        if zone_locked:
            st = "LOCKED"; dq = "LOCKED (press 'l' to unlock)"

        elif nd >= 4:
            rp = get_centers(ac) if rm=="CENTER" else get_inner(ac)
            op = order_pts(rp); sp = zs.update(op)
            lvp = sp; ldt = ct; lpa = cv2.contourArea(sp)
            ppm = get_ppm(ac, args.tag_size)
            if ppm and ppm > 0:
                lppm = ppm; lra = lpa / (ppm**2)
            st = "DETECTED"; dq = "4/4"; stats["4/4"] += 1

        elif nd == 3 and lvp is not None:
            rp = get_centers(ac) if rm=="CENTER" else get_inner(ac)
            asgn = {}; used = set()
            for pi in range(len(rp)):
                bd = float('inf'); bc = -1
                for ci in range(4):
                    if ci in used: continue
                    d = np.linalg.norm(rp[pi] - lvp[ci])
                    if d < bd: bd = d; bc = ci
                asgn[bc] = rp[pi]; used.add(bc)
            if len(asgn) == 3:
                fp, pc = predict_missing(asgn, list(asgn.keys()))
                sp = zs.update(fp); lvp = sp; ldt = ct
                lpa = cv2.contourArea(sp)
                ppm = get_ppm(ac, args.tag_size)
                if ppm and ppm > 0: lppm = ppm; lra = lpa / (ppm**2)
                st = "PARTIAL"
                dq = f"3/4 (pred:{'TL,TR,BR,BL'.split(',')[pc]})"
                stats["3/4"] += 1

        if st == "SEARCHING" and lvp is not None:
            ts = ct - ldt
            if ts < args.memory_timeout:
                st = "MEMORY"; dq = f"memory({ts:.1f}s)"; stats["memory"] += 1
            elif not zone_locked:
                st = "LOST"; lvp = None; zs.reset(); stats["lost"] += 1
            else:
                st = "LOCKED"; dq = "LOCKED"

        # =============================================================
        # STAGE 2: YOLO detection
        # =============================================================
        res = model(frame, verbose=False, conf=args.conf)
        fd = fyolo(res, areg, ct=args.conf)

        # =============================================================
        # STAGE 3: Utilization (direct mask)
        # =============================================================
        nz = 0; oam = None; up = 0.0
        if lvp is not None:
            up, zpx, opx, nz = calc_util_direct(frame.shape, lvp, fd)
            us = UTIL_A * up + (1 - UTIL_A) * us
            if lppm and lppm > 0 and zpx > 0:
                oam = lra * (opx / zpx) if lra else None

        # =============================================================
        # DRAW: Zone overlay
        # =============================================================
        if lvp is not None:
            pd = lvp.astype(int).reshape((-1, 1, 2))
            ov = frame.copy()
            cv2.fillPoly(ov, [pd], ucol(us))
            cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

            zone_col = (255, 0, 255) if zone_locked else (0, 255, 0)
            cv2.polylines(frame, [pd], True, zone_col, 2)

            if zone_locked:
                zc = np.mean(lvp, axis=0).astype(int)
                cv2.putText(frame, "LOCKED", (zc[0]-40, zc[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            for i, (pt, lb) in enumerate(zip(lvp, ["TL", "TR", "BR", "BL"])):
                x, y = int(pt[0]), int(pt[1])
                if i == pc:
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), 2)
                    cv2.putText(frame, f"{lb}?", (x+8, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
                else:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, lb, (x+8, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            if lppm and lppm > 0 and show_ui:
                for i in range(4):
                    p1 = lvp[i]; p2 = lvp[(i+1) % 4]
                    mid = ((p1 + p2) / 2).astype(int)
                    dm = np.linalg.norm(p2 - p1) / lppm
                    cv2.putText(frame, f"{dm:.2f}m", (mid[0]-25, mid[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # DRAW: Pallet detections
        iz, oz = ddets(frame, fd, lvp)

        # ArUco outlines
        if ai is not None and len(ac) > 0:
            aruco.drawDetectedMarkers(frame, ac, ai)

        # =============================================================
        # Utilization display (top center) — toggle with 'h'
        # =============================================================
        h, w = frame.shape[:2]
        if lvp is not None and show_ui:
            uc = ucol(us)
            ut = f"UTILIZATION: {us:.1f}%"
            tsz, _ = cv2.getTextSize(ut, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            bxw = tsz[0] + 40; bxx = (w - bxw) // 2
            cv2.rectangle(frame, (bxx, 5), (bxx+bxw, 72), (0, 0, 0), -1)
            cv2.rectangle(frame, (bxx, 5), (bxx+bxw, 72), uc, 2)
            cv2.putText(frame, ut, (bxx+20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, uc, 2)
            ubar(frame, us, bxx+10, 45, bxw-20, 20)

        # =============================================================
        # Info panel (bottom-left) — toggle with 'h'
        # =============================================================
        if show_ui:
            px, py = 10, h - 280; pw = 430
            cv2.rectangle(frame, (px, py), (px+pw, h-10), (0, 0, 0), -1)

            bc = ((255,0,255) if st=="LOCKED" else (0,255,0) if st=="DETECTED"
                  else (0,255,255) if st=="PARTIAL" else (0,165,255) if st=="MEMORY"
                  else (0,0,255))
            cv2.rectangle(frame, (px, py), (px+pw, h-10), bc, 2)

            y = py + 22; lh = 22
            cv2.putText(frame, f"ZONE: {st}", (px+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2); y += lh
            cv2.putText(frame, f"Detection: {dq}", (px+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, bc, 1); y += lh

            if ucal:
                cv2.putText(frame, f"CALIBRATED (err:{cerr:.3f}px)", (px+10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "UNCALIBRATED", (px+10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            y += lh

            cv2.putText(frame, f"Ref:{rm}", (px+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1); y += lh

            if lra:
                cv2.putText(frame, f"Zone: {lra:.4f} m^2", (px+10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2); y += lh
            if oam:
                cv2.putText(frame, f"Occupied: {oam:.4f} m^2", (px+10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2); y += lh

            td = len(fd)
            rd2 = len(res[0].boxes) if res[0].boxes is not None else 0
            cv2.putText(frame, f"Pallets:{td} ({rd2-td} filtered)", (px+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1); y += lh
            cv2.putText(frame, f"  In:{iz}  Out:{oz}", (px+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1); y += lh

            if lppm:
                cv2.putText(frame, f"Scale:{lppm:.0f}px/m  Markers:{nd}/4", (px+10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1); y += lh

            cv2.putText(frame, f"Conf:{args.conf:.2f}  Timeout:{args.memory_timeout}s", (px+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1); y += lh

            zf_status = "SAVED" if zone_locked and os.path.exists(args.zone_file) else "none"
            cv2.putText(frame, f"Zone file: {zf_status} ({args.zone_file})", (px+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1); y += lh

            cv2.putText(frame, "h=HIDE  l=LOCK  q=quit  c=ref  +/-=conf  r=RESET", (px+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # FPS
        fc += 1; el = time.time() - ft
        if el > 1: fps = fc / el; fc = 0; ft = time.time()
        if show_ui:
            cv2.putText(frame, f"FPS:{fps:.1f}", (w-130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Staging Zone v14", frame)

        # =============================================================
        # Key handling
        # =============================================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('h'):
            show_ui = not show_ui
            print(f"[INFO] UI overlay: {'ON' if show_ui else 'OFF'}")

        elif key == ord('l'):
            if not zone_locked and lvp is not None:
                zone_locked = True
                save_zone(args.zone_file, lvp, lppm, lra, lpa)
                print(f"[INFO] ZONE LOCKED & SAVED")
                print(f"[INFO]   Next time you run this program, zone will auto-load")
            elif zone_locked:
                zone_locked = False
                zs.reset()
                print(f"[INFO] ZONE UNLOCKED — tracking markers again")
                print(f"[INFO]   Saved zone file kept. Press 'r' to delete it.")
            else:
                print(f"[INFO] No zone to lock — detect 4 markers first")

        elif key == ord('c'):
            rm = "CORNER" if rm == "CENTER" else "CENTER"
            lvp=lra=lppm=None; lpa=0; us=0; zs.reset(); zone_locked=False
            print(f"[INFO] Ref:{rm}")

        elif key == ord('u') and cmx is not None:
            ucal = not ucal
            lvp=lra=lppm=None; lpa=0; us=0; zs.reset(); zone_locked=False
            print(f"[INFO] Undistort:{'ON' if ucal else 'OFF'}")

        elif key in [ord('+'), ord('=')]:
            args.conf = min(0.95, args.conf + 0.05)
            print(f"[INFO] Conf:{args.conf:.2f}")

        elif key == ord('-'):
            args.conf = max(0.05, args.conf - 0.05)
            print(f"[INFO] Conf:{args.conf:.2f}")

        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"snap_{ts}.png", frame)
            print(f"[INFO] Saved snapshot")

        elif key == ord('r'):
            lvp=lppm=lra=None; lpa=0; us=0; zs.reset(); zone_locked=False
            delete_zone_file(args.zone_file)
            print("[INFO] RESET — zone cleared and saved file deleted")

    # Print stats
    tot = sum(stats.values())
    if tot > 0:
        print(f"\n{'='*50}\n  DETECTION STATS\n{'='*50}")
        for k, v in stats.items():
            print(f"  {k:>8}: {v:>6} ({v/tot*100:.1f}%)")
        print(f"{'='*50}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
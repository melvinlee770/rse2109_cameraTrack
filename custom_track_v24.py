"""
Staging Zone Tracker v22 - Perspective-Corrected + Dashboard + Boundary Strip
==============================================================================
Based on v14. Fixes dimension calculation for tilted (non-top-down)
camera angles by measuring ArUco tag sizes in a perspective-free
warped space instead of averaging pixel sizes in the tilted image.

Added: Dashboard server integration — sends utilization snapshots
       to an HTML dashboard at a configurable interval.
Added: Boundary strip — along a configurable zone edge, pallets must
       have >=50% overlap to be counted as in-zone (prevents double-
       counting when two cameras share a boundary).

Usage:
    python custom_track_v22.py --model best.pt
    python custom_track_v22.py --model best.pt --boundary-edge bottom --boundary-width 80
    python custom_track_v22.py --model best.pt --dashboard-interval 10 --port 8080
"""
import cv2, cv2.aruco as aruco, numpy as np, time, argparse, os, sys, json
from ultralytics import YOLO
from dashboard_server import DashboardServer

TAG_SIZE_CM=2.4; MODEL_PATH="best.pt"; CONF_THRESH=0.5
MEM_TIMEOUT=5.0; SMOOTH_A=0.1; UTIL_A=0.2; MAX_JUMP_PX=30; ROLLING_N=10
ZONE_SAVE_FILE="zone_data.json"; BEW=600; BEH=400

PALLET_TYPES={"US":{"width_m":1.219,"depth_m":1.016,"label":"US(48x40in)","color":(0,165,255)},
              "UK":{"width_m":1.200,"depth_m":0.800,"label":"UK(1200x800mm)","color":(255,165,0)}}
YOLO_CLASS_TO_PALLET={"uk_pallet_empty":"UK", "uk_pallet_loaded":"UK", "cage":"UK", "us_pallet_empty":"US", "us_pallet_loaded":"US",
                      "eu_pallet_empty":"UK", "eu_pallet_loaded":"UK"}

DISPLAY_NAMES={"uk_pallet_empty":"eu_pallet_empty", "uk_pallet_loaded":"eu_pallet_loaded"}

def save_zone(fp,pts,ppm,area,pa):
    d={"zone_pts":pts.tolist(),"ppm":float(ppm) if ppm else None,"real_area_m2":float(area) if area else None,
       "pixel_area":float(pa) if pa else None,"tag_size_cm":TAG_SIZE_CM,"saved_at":time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(fp,'w') as f: json.dump(d,f,indent=2)
    print(f"[INFO] Zone saved to: {fp}")

def load_zone(fp):
    if not os.path.exists(fp): return None,None,None,None
    try:
        with open(fp,'r') as f: d=json.load(f)
        print(f"[INFO] Zone loaded from: {fp} (saved: {d.get('saved_at','?')})")
        return np.array(d["zone_pts"],dtype=np.float32),d.get("ppm"),d.get("real_area_m2"),d.get("pixel_area")
    except Exception as e: print(f"[WARN] Load zone failed: {e}"); return None,None,None,None

def delete_zone_file(fp):
    if os.path.exists(fp): os.remove(fp); print(f"[INFO] Zone file deleted: {fp}")

class RobustArucoDetector:
    def __init__(self,dt=aruco.DICT_4X4_50):
        self.ad=aruco.getPredefinedDictionary(dt); self.dets=[]
        p1=aruco.DetectorParameters(); p1.adaptiveThreshWinSizeMin=3; p1.adaptiveThreshWinSizeMax=53
        p1.adaptiveThreshWinSizeStep=4; p1.minMarkerPerimeterRate=0.01
        p1.cornerRefinementMethod=aruco.CORNER_REFINE_SUBPIX
        self.dets.append(aruco.ArucoDetector(self.ad,p1))
        p2=aruco.DetectorParameters(); p2.adaptiveThreshWinSizeMin=3; p2.adaptiveThreshWinSizeMax=73
        p2.adaptiveThreshWinSizeStep=2; p2.minMarkerPerimeterRate=0.005; p2.adaptiveThreshConstant=12
        p2.cornerRefinementMethod=aruco.CORNER_REFINE_SUBPIX
        self.dets.append(aruco.ArucoDetector(self.ad,p2))
    def _pre(self,g):
        vs=[g]; cl=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)); vs.append(cl.apply(g))
        vs.append(cv2.filter2D(g,-1,np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])))
        vs.append(cv2.GaussianBlur(g,(3,3),0)); return vs
    def detect(self,gray):
        bc=bi=None; bn=0
        for img in self._pre(gray):
            for det in self.dets:
                c,i,_=det.detectMarkers(img); n=len(i) if i is not None else 0
                if n>bn: bn=n; bc=c; bi=i
                if n>=4: return bc,bi,bn
        return bc,bi,bn

class ZoneSmoother:
    def __init__(self,alpha=0.1,max_jump=30,hsize=10):
        self.alpha=alpha; self.max_jump=max_jump; self.sp=None; self.hist=[]; self.hsize=hsize
        self.locked=False; self.lc=0; self.LA=15
    def update(self,p):
        if self.sp is not None:
            if np.max(np.linalg.norm(p-self.sp,axis=1))>self.max_jump and self.locked: return self.sp.copy()
        self.hist.append(p.copy())
        if len(self.hist)>self.hsize: self.hist.pop(0)
        avg=np.mean(self.hist,axis=0).astype(np.float32) if len(self.hist)>=3 else p
        if self.sp is None: self.sp=avg.copy()
        else: self.sp=self.alpha*avg+(1-self.alpha)*self.sp
        self.lc+=1
        if self.lc>=self.LA: self.locked=True
        return self.sp.copy()
    def reset(self): self.sp=None; self.hist=[]; self.locked=False; self.lc=0

def predict_missing(known,kidx):
    miss=(set([0,1,2,3])-set(kidx)).pop()
    pm={0:(1,3,2),1:(0,2,3),2:(1,3,0),3:(0,2,1)}; a,b,c=pm[miss]
    fp=np.zeros((4,2),dtype=np.float32)
    for i in kidx: fp[i]=known[i]
    fp[miss]=known[a]+known[b]-known[c]; return fp,miss

def load_cal(f):
    if not os.path.exists(f): return None,None,None
    d=np.load(f); return d['camera_matrix'],d['dist_coeffs'],d['reprojection_error'][0]

def setup_undist(cm,dc,w,h):
    nc,_=cv2.getOptimalNewCameraMatrix(cm,dc,(w,h),0,(w,h))
    m1,m2=cv2.initUndistortRectifyMap(cm,dc,None,nc,(w,h),cv2.CV_32FC1); return m1,m2

def order_pts(pts):
    r=np.zeros((4,2),dtype="float32"); s=pts.sum(axis=1); r[0]=pts[np.argmin(s)]; r[2]=pts[np.argmax(s)]
    d=np.diff(pts,axis=1); r[1]=pts[np.argmin(d)]; r[3]=pts[np.argmax(d)]; return r

def get_centers(c): return np.array([np.mean(x[0],axis=0) for x in c])
def get_inner(corners):
    ac=np.array([np.mean(c[0],axis=0) for c in corners]); ct=np.mean(ac,axis=0)
    return np.array([c[0][np.argmin(np.linalg.norm(c[0]-ct,axis=1))] for c in corners],dtype=np.float32)
def get_outer(corners):
    ac=np.array([np.mean(c[0],axis=0) for c in corners]); ct=np.mean(ac,axis=0)
    return np.array([c[0][np.argmax(np.linalg.norm(c[0]-ct,axis=1))] for c in corners],dtype=np.float32)

def get_ppm(corners,tag_cm):
    el=[]
    for mc in corners:
        p=mc[0]
        for i in range(4): el.append(np.linalg.norm(p[(i+1)%4]-p[i]))
    return (np.mean(el)/tag_cm)*100.0 if el else None

def compute_real_dims_perspective(zone_pts, aruco_corners, tag_cm):
    TEMP_SZ = 500.0
    temp_dst = np.array([[0,0],[TEMP_SZ,0],[TEMP_SZ,TEMP_SZ],[0,TEMP_SZ]], dtype=np.float32)
    M_temp = cv2.getPerspectiveTransform(zone_pts, temp_dst)
    edges_h, edges_v = [], []
    for mc in aruco_corners:
        for i in range(4):
            p1 = mc[0][i]; p2 = mc[0][(i+1)%4]
            w1 = warp_pt(p1, M_temp); w2 = warp_pt(p2, M_temp)
            elen = np.linalg.norm(np.array(w2) - np.array(w1))
            if abs(w2[0]-w1[0]) >= abs(w2[1]-w1[1]): edges_h.append(elen)
            else: edges_v.append(elen)
    if not edges_h or not edges_v: return None, None, None
    tag_m = tag_cm / 100.0
    ppm_x = np.mean(edges_h) / tag_m; ppm_y = np.mean(edges_v) / tag_m
    if ppm_x <= 0 or ppm_y <= 0: return None, None, None
    real_w = TEMP_SZ / ppm_x; real_h = TEMP_SZ / ppm_y
    return real_w, real_h, real_w * real_h

def get_areg(corners,mg=15):
    return [(int(c[0][:,0].min())-mg,int(c[0][:,1].min())-mg,int(c[0][:,0].max())+mg,int(c[0][:,1].max())+mg) for c in corners]

def bovr(box,reg):
    x1=max(box[0],reg[0]);y1=max(box[1],reg[1]);x2=min(box[2],reg[2]);y2=min(box[3],reg[3])
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
        fl.append({"box":(int(x1),int(y1),int(x2),int(y2)),"conf":cf,"cls_id":ci,"cls_name":DISPLAY_NAMES.get(cn,cn),
                    "pallet_type":YOLO_CLASS_TO_PALLET.get(cn,None)})
    return fl

# =====================================================================
# Boundary strip
# =====================================================================

def compute_boundary_strip(zone_pts, edge, width_px):
    """
    Compute a boundary strip polygon along one edge of the zone.
    Zone points ordered: TL(0), TR(1), BR(2), BL(3).
    The strip extends width_px pixels INTO the zone from the specified edge.
    """
    if zone_pts is None or width_px <= 0: return None
    tl, tr, br, bl = zone_pts[0], zone_pts[1], zone_pts[2], zone_pts[3]

    if edge == "bottom":
        dir_l = (tl - bl); dir_l = dir_l / (np.linalg.norm(dir_l) + 1e-9) * width_px
        dir_r = (tr - br); dir_r = dir_r / (np.linalg.norm(dir_r) + 1e-9) * width_px
        return np.array([bl, br, br + dir_r, bl + dir_l], dtype=np.float32)
    elif edge == "top":
        dir_l = (bl - tl); dir_l = dir_l / (np.linalg.norm(dir_l) + 1e-9) * width_px
        dir_r = (br - tr); dir_r = dir_r / (np.linalg.norm(dir_r) + 1e-9) * width_px
        return np.array([tl, tr, tr + dir_r, tl + dir_l], dtype=np.float32)
    elif edge == "left":
        dir_t = (tr - tl); dir_t = dir_t / (np.linalg.norm(dir_t) + 1e-9) * width_px
        dir_b = (br - bl); dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-9) * width_px
        return np.array([tl, tl + dir_t, bl + dir_b, bl], dtype=np.float32)
    elif edge == "right":
        dir_t = (tl - tr); dir_t = dir_t / (np.linalg.norm(dir_t) + 1e-9) * width_px
        dir_b = (bl - br); dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-9) * width_px
        return np.array([tr, tr + dir_t, br + dir_b, br], dtype=np.float32)
    return None

def _overlap_ratio(box, zone_pts):
    """Compute overlap ratio: intersection(box, zone) / box_area."""
    x1, y1, x2, y2 = box
    box_pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    box_area = (x2-x1)*(y2-y1)
    if box_area <= 0: return 0.0
    zx_min = int(max(0, min(zone_pts[:,0].min(), x1)))
    zy_min = int(max(0, min(zone_pts[:,1].min(), y1)))
    zx_max = int(max(zone_pts[:,0].max(), x2) + 1)
    zy_max = int(max(zone_pts[:,1].max(), y2) + 1)
    rw = zx_max - zx_min; rh = zy_max - zy_min
    if rw <= 0 or rh <= 0: return 0.0
    zm = np.zeros((rh, rw), dtype=np.uint8)
    bm = np.zeros((rh, rw), dtype=np.uint8)
    zp_s = (zone_pts - [zx_min, zy_min]).astype(np.int32).reshape((-1,1,2))
    bp_s = (box_pts - [zx_min, zy_min]).astype(np.int32).reshape((-1,1,2))
    cv2.fillPoly(zm, [zp_s], 255)
    cv2.fillPoly(bm, [bp_s], 255)
    inter = cv2.countNonZero(cv2.bitwise_and(zm, bm))
    return inter / box_area

def _box_in_strip(box, strip):
    """Check if bbox center is inside the boundary strip."""
    if strip is None: return False
    x1, y1, x2, y2 = box
    cx = (x1+x2)/2; cy = (y1+y2)/2
    sp = strip.astype(np.float32).reshape((-1,1,2))
    return cv2.pointPolygonTest(sp, (float(cx), float(cy)), False) >= 0

def inzone(box, zp, overlap_thresh=0.3, boundary_strip=None, boundary_thresh=0.5):
    """
    Check if a bounding box is inside the zone.
    Two modes:
    1) Pallet center is in boundary strip -> strict overlap-only (default 50%)
    2) Otherwise -> original 3-stage check (center, bottom-center, 30% overlap)
    """
    if zp is None: return False

    # --- BOUNDARY STRIP: strict overlap only, no shortcuts ---
    if boundary_strip is not None and _box_in_strip(box, boundary_strip):
        return _overlap_ratio(box, zp) >= boundary_thresh

    # --- NORMAL ZONE: original 3-stage logic ---
    zp2 = zp.astype(np.float32).reshape((-1,1,2))
    x1, y1, x2, y2 = box
    cx = (x1+x2)/2; cy = (y1+y2)/2
    if cv2.pointPolygonTest(zp2, (float(cx), float(cy)), False) >= 0: return True
    if cv2.pointPolygonTest(zp2, (float(cx), float(y2)), False) >= 0: return True
    return _overlap_ratio(box, zp) >= overlap_thresh

# =====================================================================
# Bird's-eye, warping, utilization
# =====================================================================

def compute_be(zp,rw,rh,bw,bh):
    asp=rw/rh if rh>0 else 1.0
    if asp>bw/bh: ow=bw;oh=int(bw/asp)
    else: oh=bh;ow=int(bh*asp)
    m=20;owi=ow-2*m;ohi=oh-2*m
    dst=np.array([[m,m],[m+owi,m],[m+owi,m+ohi],[m,m+ohi]],dtype=np.float32)
    M=cv2.getPerspectiveTransform(zp,dst); return M,dst,ow,oh,owi/rw,ohi/rh

def warp_pt(pt,M):
    p=np.array([pt[0],pt[1],1.0],dtype=np.float64);r=M@p
    return (r[0]/r[2],r[1]/r[2]) if r[2]!=0 else (pt[0],pt[1])

def calc_util_fixed(besz, dst, dets, zcam, M, ppx, ppy, boundary_strip=None, boundary_thresh=0.5):
    bw,bh=besz; zm=np.zeros((bh,bw),dtype=np.uint8)
    cv2.fillPoly(zm,[dst.astype(np.int32).reshape((-1,1,2))],255); za=cv2.countNonZero(zm)
    if za==0: return 0.0,0,0,{"US":0,"UK":0,"unknown":0},{}
    pm=np.zeros((bh,bw),dtype=np.uint8); counts={"US":0,"UK":0,"unknown":0}; cls_counts={}
    for det in dets:
        box=det["box"];pt=det["pallet_type"];cn=det["cls_name"]
        if not inzone(box, zcam, boundary_strip=boundary_strip, boundary_thresh=boundary_thresh): continue
        if pt and pt in PALLET_TYPES: counts[pt]+=1;cls_counts[cn]=cls_counts.get(cn,0)+1
        else: counts["unknown"]+=1
        x1,y1,x2,y2=box
        corners_cam=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        corners_be=[warp_pt(c,M) for c in corners_cam]
        pts_be=np.array(corners_be,dtype=np.int32)
        pts_be[:,0]=np.clip(pts_be[:,0],0,bw-1); pts_be[:,1]=np.clip(pts_be[:,1],0,bh-1)
        cv2.fillConvexPoly(pm,pts_be,255)
    oa=cv2.countNonZero(cv2.bitwise_and(pm,zm)); return (oa/za)*100.0,za,oa,counts,cls_counts

def ucol(p):
    if p<30:return(0,255,0)
    elif p<50:return(0,255,255)
    elif p<70:return(0,165,255)
    elif p<90:return(0,0,255)
    return(0,0,200)

def ubar(f,p,x,y,bw=300,bh=26):
    c=ucol(p);cv2.rectangle(f,(x,y),(x+bw,y+bh),(50,50,50),-1);cv2.rectangle(f,(x,y),(x+bw,y+bh),(150,150,150),1)
    fw=int(bw*min(p,100)/100)
    if fw>0:cv2.rectangle(f,(x,y),(x+fw,y+bh),c,-1)
    t=f"{p:.1f}%";sz,_=cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.putText(f,t,(x+(bw-sz[0])//2,y+(bh+sz[1])//2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

def ddets(f, dets, zp, boundary_strip=None, boundary_thresh=0.5):
    iz=oz=0
    for det in dets:
        box=det["box"];cf=det["conf"];cn=det["cls_name"];pt=det["pallet_type"]
        ins=inzone(box, zp, boundary_strip=boundary_strip, boundary_thresh=boundary_thresh)
        x1,y1,x2,y2=box
        if pt and pt in PALLET_TYPES: col=PALLET_TYPES[pt]["color"];tl=cn
        else: col=(128,128,128);tl=cn
        in_strip = boundary_strip is not None and _box_in_strip(box, boundary_strip)
        if ins: iz+=1
        else: col=tuple(c//2 for c in col);oz+=1
        cv2.rectangle(f,(x1,y1),(x2,y2),col,2)
        lb=f"{tl} {cf:.2f}";ls,_=cv2.getTextSize(lb,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        cv2.rectangle(f,(x1,y1-ls[1]-10),(x1+ls[0]+6,y1),col,-1)
        cv2.putText(f,lb,(x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        status_txt = "IN ZONE" if ins else "OUTSIDE"
        if in_strip: status_txt += " [BOUNDARY]"
        cv2.putText(f, status_txt, (x1, y2+22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if ins and pt and pt in PALLET_TYPES:
            cv2.putText(f,f"{PALLET_TYPES[pt]['width_m']:.2f}x{PALLET_TYPES[pt]['depth_m']:.2f}m",
                        (x1,y2+44),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
    return iz,oz

def main():
    pa=argparse.ArgumentParser(description="Staging Zone v22 - Perspective Corrected + Dashboard + Boundary Strip")
    pa.add_argument("--camera",type=int,default=0);pa.add_argument("--model",type=str,default=MODEL_PATH)
    pa.add_argument("--calib-file",type=str,default="camera_calibration.npz");pa.add_argument("--no-calib",action="store_true")
    pa.add_argument("--tag-size",type=float,default=TAG_SIZE_CM);pa.add_argument("--conf",type=float,default=CONF_THRESH)
    pa.add_argument("--memory-timeout",type=float,default=MEM_TIMEOUT);pa.add_argument("--zone-file",type=str,default=ZONE_SAVE_FILE)
    pa.add_argument("--port", type=int, default=5000, help="Dashboard server port (default: 5000)")
    pa.add_argument("--dashboard-interval", type=float, default=30, help="Seconds between dashboard updates (default: 30)")
    pa.add_argument("--boundary-edge", type=str, default="none", choices=["none","top","bottom","left","right"],
                    help="Which zone edge is the shared camera boundary (default: none = disabled)")
    pa.add_argument("--boundary-width", type=int, default=80,
                    help="Boundary strip width in pixels extending into the zone (default: 80)")
    pa.add_argument("--boundary-thresh", type=float, default=0.5,
                    help="Overlap threshold for boundary strip (default: 0.5 = 50%%)")
    args=pa.parse_args()

    dashboard = DashboardServer(port=args.port, interval=args.dashboard_interval)
    dashboard.start()

    if not os.path.exists(args.model): print(f"[ERROR] Model not found:{args.model}");sys.exit(1)
    print(f"[INFO] Loading YOLO:{args.model}"); model=YOLO(args.model)
    print(f"[INFO] Classes:{model.names}"); print(f"[INFO] Mapping:{YOLO_CLASS_TO_PALLET}")
    for k,v in PALLET_TYPES.items(): print(f"[INFO]   {k}:{v['width_m']}x{v['depth_m']}m")
    mcls=list(model.names.values())
    for yc in YOLO_CLASS_TO_PALLET:
        if yc not in mcls: print(f"[WARN] Mapped class '{yc}' not in model — check spelling!")
    unmapped=[c for c in mcls if c not in YOLO_CLASS_TO_PALLET]
    if unmapped: print(f"[WARN] UNMAPPED model classes (will show as grey/unknown): {unmapped}")
    else: print(f"[INFO] All model classes are mapped to US/UK.")

    rd=RobustArucoDetector(aruco.DICT_4X4_50)
    ucal=False;m1=m2=cmx=dcc=None;cerr=0
    if not args.no_calib:
        cmx,dcc,cerr=load_cal(args.calib_file)
        if cmx is not None: ucal=True;print(f"[INFO] Calibrated(err:{cerr:.4f}px)")
        else: print("[WARN] No calibration file.")

    cap=cv2.VideoCapture(args.camera)
    if not cap.isOpened(): print(f"[ERROR] Camera {args.camera}");sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720);udi=False

    zs=ZoneSmoother(alpha=SMOOTH_A,max_jump=MAX_JUMP_PX,hsize=ROLLING_N)
    lvp=None;ldt=0;lppm=None;lra=None;lpa=0;rm="CENTER";us_val=0.0
    ft=time.time();fc=0;fps=0.0;stats={"4/4":0,"3/4":0,"memory":0,"lost":0}
    zone_locked=False;show_ui=True;show_be=False
    lM=None;ldst=None;lbw=BEW;lbh=BEH;lppx=lppy=100
    bstrip=None

    saved_pts,saved_ppm,saved_area,saved_pa=load_zone(args.zone_file)
    if saved_pts is not None:
        lvp=saved_pts;lppm=saved_ppm;lra=saved_area;lpa=saved_pa if saved_pa else 0
        zone_locked=True;ldt=time.time();print("[INFO] Restored locked zone")
        if args.boundary_edge != "none":
            bstrip=compute_boundary_strip(lvp, args.boundary_edge, args.boundary_width)
            print(f"[INFO] Boundary strip restored: edge={args.boundary_edge} width={args.boundary_width}px thresh={args.boundary_thresh}")

    print(f"\n[INFO] Tag:{args.tag_size}cm Conf:{args.conf} Timeout:{args.memory_timeout}s")
    print(f"[INFO] Mode: ACTUAL FOOTPRINT (warped bounding box)")
    if args.boundary_edge != "none":
        print(f"[INFO] Boundary strip: {args.boundary_edge} edge, {args.boundary_width}px wide, {args.boundary_thresh*100:.0f}% threshold")
    else:
        print(f"[INFO] Boundary strip: DISABLED (use --boundary-edge to enable)")
    print(f"[INFO] Dashboard: http://localhost:{args.port} (interval: {args.dashboard_interval}s)")
    print(f"[INFO] Controls:")
    print(f"  'l'=LOCK/UNLOCK  'h'=HIDE/SHOW UI  'b'=bird's-eye  'q'=quit")
    print(f"  'c'=ref mode  'u'=undistort  '+'/'-'=conf  's'=snap  'r'=RESET")

    while True:
        ret,raw=cap.read()
        if not ret: break
        if ucal and not udi: h,w=raw.shape[:2];m1,m2=setup_undist(cmx,dcc,w,h);udi=True
        frame=cv2.remap(raw,m1,m2,cv2.INTER_LINEAR) if ucal and m1 is not None else raw.copy()
        ct=time.time();gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        ac,ai,nd=rd.detect(gray);areg=get_areg(ac) if ac else [];st="SEARCHING";dq="NONE";pc=-1

        if zone_locked:
            st="LOCKED";dq="LOCKED (press 'l' to unlock)"
            if lM is None and nd>=3 and lvp is not None:
                lppm=get_ppm(ac,args.tag_size)
                rdims=compute_real_dims_perspective(lvp,ac,args.tag_size)
                if rdims[0] is not None:
                    rw,rh,lra=rdims
                    if rw>0 and rh>0: lM,ldst,lbw,lbh,lppx,lppy=compute_be(lvp,rw,rh,BEW,BEH)
        elif nd>=4:
            rp=get_centers(ac) if rm=="CENTER" else get_outer(ac) if rm=="OUTER" else get_inner(ac);op=order_pts(rp);sp=zs.update(op)
            lvp=sp;ldt=ct;lpa=cv2.contourArea(sp);lppm=get_ppm(ac,args.tag_size)
            rdims=compute_real_dims_perspective(sp,ac,args.tag_size)
            if rdims[0] is not None:
                rw,rh,lra=rdims
                if rw>0 and rh>0: lM,ldst,lbw,lbh,lppx,lppy=compute_be(sp,rw,rh,BEW,BEH)
            st="DETECTED";dq="4/4";stats["4/4"]+=1
            if args.boundary_edge != "none": bstrip=compute_boundary_strip(lvp, args.boundary_edge, args.boundary_width)
        elif nd==3 and lvp is not None:
            rp=get_centers(ac) if rm=="CENTER" else get_outer(ac) if rm=="OUTER" else get_inner(ac);asgn={};used=set()
            for pi in range(len(rp)):
                bd=float('inf');bc=-1
                for ci in range(4):
                    if ci in used: continue
                    d=np.linalg.norm(rp[pi]-lvp[ci])
                    if d<bd: bd=d;bc=ci
                asgn[bc]=rp[pi];used.add(bc)
            if len(asgn)==3:
                fp,pc=predict_missing(asgn,list(asgn.keys()));sp=zs.update(fp)
                lvp=sp;ldt=ct;lpa=cv2.contourArea(sp);lppm=get_ppm(ac,args.tag_size)
                rdims=compute_real_dims_perspective(sp,ac,args.tag_size)
                if rdims[0] is not None:
                    rw,rh,lra=rdims
                    if rw>0 and rh>0: lM,ldst,lbw,lbh,lppx,lppy=compute_be(sp,rw,rh,BEW,BEH)
                st="PARTIAL";dq=f"3/4(pred:{'TL,TR,BR,BL'.split(',')[pc]})";stats["3/4"]+=1
                if args.boundary_edge != "none": bstrip=compute_boundary_strip(lvp, args.boundary_edge, args.boundary_width)

        if st=="SEARCHING" and lvp is not None:
            ts=ct-ldt
            if ts<args.memory_timeout: st="MEMORY";dq=f"memory({ts:.1f}s)";stats["memory"]+=1
            elif not zone_locked: st="LOST";lvp=None;zs.reset();stats["lost"]+=1;bstrip=None
            else: st="LOCKED";dq="LOCKED"

        res=model(frame,verbose=False,conf=args.conf);fd=fyolo(res,areg,ct=args.conf)

        counts={"US":0,"UK":0,"unknown":0};cls_counts={};oam=None;up=0.0
        if lvp is not None and lM is not None and ldst is not None:
            up,zpx,opx,counts,cls_counts=calc_util_fixed((lbw,lbh),ldst,fd,lvp,lM,lppx,lppy,
                                                          boundary_strip=bstrip, boundary_thresh=args.boundary_thresh)
            us_val=UTIL_A*up+(1-UTIL_A)*us_val
            if lppm and lppm>0 and zpx>0: oam=lra*(opx/zpx) if lra else None
        elif lvp is not None:
            for det in fd:
                if inzone(det["box"], lvp, boundary_strip=bstrip, boundary_thresh=args.boundary_thresh):
                    pt=det["pallet_type"];cn=det["cls_name"]
                    counts[pt if pt in counts else "unknown"]+=1
                    if pt and pt in PALLET_TYPES: cls_counts[cn]=cls_counts.get(cn,0)+1

        # === DRAW: Zone ===
        if lvp is not None:
            pd=lvp.astype(int).reshape((-1,1,2));ov=frame.copy()
            cv2.fillPoly(ov,[pd],ucol(us_val));cv2.addWeighted(ov,0.15,frame,0.85,0,frame)
            zcol=(255,0,255) if zone_locked else (0,255,0);cv2.polylines(frame,[pd],True,zcol,2)
            if zone_locked:
                zc=np.mean(lvp,axis=0).astype(int);cv2.putText(frame,"LOCKED",(zc[0]-50,zc[1]),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,255),2)
            for i,(pt,lb) in enumerate(zip(lvp,["TL","TR","BR","BL"])):
                x,y=int(pt[0]),int(pt[1])
                if i==pc: cv2.circle(frame,(x,y),8,(0,255,255),2);cv2.putText(frame,f"{lb}?",(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                else: cv2.circle(frame,(x,y),5,(0,0,255),-1);cv2.putText(frame,lb,(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            # Draw boundary strip overlay
            if bstrip is not None:
                bs_pts=bstrip.astype(int).reshape((-1,1,2))
                bs_ov=frame.copy();cv2.fillPoly(bs_ov,[bs_pts],(255,0,255))
                cv2.addWeighted(bs_ov,0.12,frame,0.88,0,frame)
                cv2.polylines(frame,[bs_pts],True,(255,0,255),1,cv2.LINE_AA)
                bs_center=np.mean(bstrip,axis=0).astype(int)
                cv2.putText(frame,"BOUNDARY 50%",(bs_center[0]-60,bs_center[1]+5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,255),1)
            # Edge dimensions
            if lM is not None and lppx and lppy:
                for i in range(4):
                    p1=lvp[i];p2=lvp[(i+1)%4];mid=((p1+p2)/2).astype(int)
                    w1=warp_pt(p1,lM);w2=warp_pt(p2,lM)
                    dx=(w2[0]-w1[0])/lppx;dy=(w2[1]-w1[1])/lppy
                    edge_m=np.sqrt(dx**2+dy**2)
                    cv2.putText(frame,f"{edge_m:.2f}m",(mid[0]-30,mid[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
            elif lppm and lppm>0:
                for i in range(4):
                    p1=lvp[i];p2=lvp[(i+1)%4];mid=((p1+p2)/2).astype(int)
                    cv2.putText(frame,f"{np.linalg.norm(p2-p1)/lppm:.2f}m",(mid[0]-30,mid[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

        iz,oz=ddets(frame, fd, lvp, boundary_strip=bstrip, boundary_thresh=args.boundary_thresh)
        if ai is not None and len(ac)>0: aruco.drawDetectedMarkers(frame,ac,ai)

        # === Utilization (top center) ===
        h,w=frame.shape[:2]
        if lvp is not None:
            uc=ucol(us_val);ut=f"UTILIZATION: {us_val:.1f}%"
            tsz,_=cv2.getTextSize(ut,cv2.FONT_HERSHEY_SIMPLEX,0.9,2);bxw=tsz[0]+50;bxx=(w-bxw)//2
            cv2.rectangle(frame,(bxx,5),(bxx+bxw,80),(0,0,0),-1);cv2.rectangle(frame,(bxx,5),(bxx+bxw,80),uc,2)
            cv2.putText(frame,ut,(bxx+25,38),cv2.FONT_HERSHEY_SIMPLEX,0.9,uc,2)
            ubar(frame,us_val,bxx+10,48,bxw-20,24)

        # === Info panel ===
        if show_ui:
            px,py_p=10,h-490;pw=520;cv2.rectangle(frame,(px,py_p),(px+pw,h-10),(0,0,0),-1)
            bc=((255,0,255) if st=="LOCKED" else (0,255,0) if st=="DETECTED" else (0,255,255) if st=="PARTIAL" else (0,165,255) if st=="MEMORY" else (0,0,255))
            cv2.rectangle(frame,(px,py_p),(px+pw,h-10),bc,2);y=py_p+28;lh=28
            cv2.putText(frame,f"ZONE: {st}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,bc,2);y+=lh
            cv2.putText(frame,f"Detection: {dq}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,bc,1);y+=lh
            if ucal: cv2.putText(frame,f"CALIBRATED(err:{cerr:.3f}px)",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
            else: cv2.putText(frame,"UNCALIBRATED",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),1)
            y+=lh;cv2.putText(frame,f"Ref:{rm} | ACTUAL FOOTPRINT",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,200,0),1);y+=lh
            if lra: cv2.putText(frame,f"Zone:{lra:.4f}m^2",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2);y+=lh
            if oam: cv2.putText(frame,f"Occupied:{oam:.4f}m^2",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2);y+=lh
            uc_us=PALLET_TYPES["US"]["color"];uc_uk=PALLET_TYPES["UK"]["color"]
            cv2.putText(frame,f"US:{counts['US']}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,uc_us,2)
            cv2.putText(frame,f"EU:{counts['UK']}",(px+140,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,uc_uk,2)
            if counts["unknown"]>0: cv2.putText(frame,f"?:{counts['unknown']}",(px+270,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(128,128,128),2)
            y+=lh
            if cls_counts:
                us_cls=[f"{cn}:{n}" for cn,n in cls_counts.items() if YOLO_CLASS_TO_PALLET.get(cn)=="US"]
                eu_cls=[f"{cn}:{n}" for cn,n in cls_counts.items() if YOLO_CLASS_TO_PALLET.get(cn)=="UK"]
                if us_cls: cv2.putText(frame,f"  {', '.join(us_cls)}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,uc_us,1);y+=lh
                if eu_cls: cv2.putText(frame,f"  {', '.join(eu_cls)}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,uc_uk,1);y+=lh
            td=len(fd);rd2=len(res[0].boxes) if res[0].boxes is not None else 0
            cv2.putText(frame,f"Pallets:{td}({rd2-td} filtered)",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1);y+=lh
            cv2.putText(frame,f"  In:{iz} Out:{oz}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1);y+=lh
            if lppm: cv2.putText(frame,f"Scale:{lppm:.0f}px/m Markers:{nd}/4",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1);y+=lh
            cv2.putText(frame,f"Conf:{args.conf:.2f} Timeout:{args.memory_timeout}s",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1);y+=lh
            bnd_txt=f"Boundary:{args.boundary_edge}({args.boundary_width}px,{args.boundary_thresh*100:.0f}%)" if args.boundary_edge!="none" else "Boundary:OFF"
            cv2.putText(frame,bnd_txt,(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,255) if args.boundary_edge!="none" else (150,150,150),1);y+=lh
            zf_st="SAVED" if zone_locked and os.path.exists(args.zone_file) else "none"
            cv2.putText(frame,f"Zone:{zf_st}({args.zone_file})",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.38,(150,150,150),1);y+=lh
            cv2.putText(frame,"h=HIDE l=LOCK b=BIRD q=quit c=ref +/-=conf r=RESET",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)

        fc+=1;el=time.time()-ft
        if el>1: fps=fc/el;fc=0;ft=time.time()
        if show_ui: cv2.putText(frame,f"FPS:{fps:.1f}",(w-150,35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        # --- Feed to dashboard ---
        edge_lengths = []
        if lM is not None and lppx and lppy and lvp is not None:
            for i in range(4):
                p1=lvp[i]; p2=lvp[(i+1)%4]
                w1=warp_pt(p1,lM); w2=warp_pt(p2,lM)
                dx=(w2[0]-w1[0])/lppx; dy=(w2[1]-w1[1])/lppy
                edge_lengths.append(round(np.sqrt(dx**2+dy**2), 3))

        dashboard.feed({
            "utilization":      round(us_val, 2),
            "zone_area_m2":     round(lra, 4) if lra else None,
            "occupied_area_m2": round(oam, 4) if oam else None,
            "counts":           counts,
            "cls_counts":       cls_counts,
            "zone_status":      st,
            "detection_quality": dq,
            "fps":              round(fps, 1),
            "markers_detected": nd,
            "pallets_total":    len(fd),
            "pallets_in_zone":  iz,
            "pallets_outside":  oz,
            "pallets_filtered": (len(res[0].boxes) if res[0].boxes is not None else 0) - len(fd),
            "confidence_threshold": args.conf,
            "zone_locked":      zone_locked,
            "ppm":              round(lppm, 1) if lppm else None,
            "edge_lengths_m":   edge_lengths if edge_lengths else None,
        })

        cv2.imshow("Staging Zone v22",frame)

        # Bird's-eye
        if show_be and lM is not None and lvp is not None:
            bei=cv2.warpPerspective(frame,lM,(lbw,lbh))
            cv2.polylines(bei,[ldst.astype(np.int32).reshape((-1,1,2))],True,(0,255,0),2)
            for det in fd:
                box=det["box"];pt=det["pallet_type"];cn=det["cls_name"]
                if not inzone(box, lvp, boundary_strip=bstrip, boundary_thresh=args.boundary_thresh): continue
                if pt and pt in PALLET_TYPES: col=PALLET_TYPES[pt]["color"]
                else: col=(128,128,128)
                lb=cn;x1,y1,x2,y2=box
                corners_cam=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                corners_be=[warp_pt(c,lM) for c in corners_cam]
                pts_be=np.array(corners_be,dtype=np.int32)
                pts_be[:,0]=np.clip(pts_be[:,0],0,lbw-1);pts_be[:,1]=np.clip(pts_be[:,1],0,lbh-1)
                fx1,fy1=pts_be.min(axis=0);fx2,fy2=pts_be.max(axis=0)
                if fx2>fx1 and fy2>fy1:
                    ov2=bei.copy();cv2.fillConvexPoly(ov2,pts_be,col);cv2.addWeighted(ov2,0.4,bei,0.6,0,bei)
                    cv2.polylines(bei,[pts_be.reshape((-1,1,2))],True,col,2);cv2.putText(bei,lb,(fx1+4,fy1+16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(bei,"BIRD'S-EYE(Actual Footprints)",(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.imshow("Bird's-Eye View",bei)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('h'): show_ui=not show_ui;print(f"[INFO] UI:{'ON' if show_ui else 'OFF'}")
        elif key==ord('b'):
            show_be=not show_be
            if not show_be: cv2.destroyWindow("Bird's-Eye View")
            print(f"[INFO] Bird's-eye:{'ON' if show_be else 'OFF'}")
        elif key==ord('l'):
            if not zone_locked and lvp is not None:
                zone_locked=True;save_zone(args.zone_file,lvp,lppm,lra,lpa);print("[INFO] ZONE LOCKED & SAVED")
                if args.boundary_edge != "none":
                    bstrip=compute_boundary_strip(lvp, args.boundary_edge, args.boundary_width)
                    print(f"[INFO] Boundary strip computed: {args.boundary_edge} edge")
            elif zone_locked: zone_locked=False;zs.reset();print("[INFO] ZONE UNLOCKED")
            else: print("[INFO] No zone to lock")
        elif key==ord('c'): rm={"CENTER":"INNER","INNER":"OUTER","OUTER":"CENTER"}[rm];lvp=lra=lppm=lM=None;lpa=0;us_val=0;zs.reset();zone_locked=False;bstrip=None;print(f"[INFO] Ref:{rm}")
        elif key==ord('u') and cmx is not None: ucal=not ucal;lvp=lra=lppm=lM=None;lpa=0;us_val=0;zs.reset();zone_locked=False;bstrip=None;print(f"[INFO] Undistort:{'ON' if ucal else 'OFF'}")
        elif key in [ord('+'),ord('=')]: args.conf=min(0.95,args.conf+0.05);print(f"[INFO] Conf:{args.conf:.2f}")
        elif key==ord('-'): args.conf=max(0.05,args.conf-0.05);print(f"[INFO] Conf:{args.conf:.2f}")
        elif key==ord('s'): cv2.imwrite(f"snap_{time.strftime('%Y%m%d_%H%M%S')}.png",frame);print("[INFO] Snap saved")
        elif key==ord('r'): lvp=lppm=lra=lM=None;lpa=0;us_val=0;zs.reset();zone_locked=False;bstrip=None;delete_zone_file(args.zone_file);print("[INFO] RESET")

    dashboard.stop()
    tot=sum(stats.values())
    if tot>0:
        print(f"\n{'='*50}\n  DETECTION STATS\n{'='*50}")
        for k,v in stats.items(): print(f"  {k:>8}:{v:>6}({v/tot*100:.1f}%)")
        print(f"{'='*50}")
    cap.release();cv2.destroyAllWindows()

if __name__=="__main__": main()
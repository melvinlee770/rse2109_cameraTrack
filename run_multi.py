"""
Multi-Camera Launcher
======================
Runs two camera tracker instances in parallel threads,
feeding into a single unified dashboard.

Usage:
    python run_multi.py --model best.pt
    python run_multi.py --model best.pt --camera-a 0 --camera-b 1
    python run_multi.py --model best.pt --dashboard-interval 30 --port 5000
"""
import cv2, cv2.aruco as aruco, numpy as np, time, argparse, os, sys, json, threading
from ultralytics import YOLO
from multi_dashboard_server import MultiDashboardServer

# Import everything from the tracker module
TAG_SIZE_CM=2.4; MODEL_PATH="best.pt"; CONF_THRESH=0.5
MEM_TIMEOUT=5.0; SMOOTH_A=0.1; UTIL_A=0.2; MAX_JUMP_PX=30; ROLLING_N=10
BEW=600; BEH=400

PALLET_TYPES={"US":{"width_m":1.219,"depth_m":1.016,"label":"US(48x40in)","color":(0,165,255)},
              "UK":{"width_m":1.200,"depth_m":0.800,"label":"UK(1200x800mm)","color":(255,165,0)}}
YOLO_CLASS_TO_PALLET={"uk_pallet_empty":"UK", "uk_pallet_loaded":"UK", "cage":"UK",
                      "us_pallet_empty":"US", "us_pallet_loaded":"US",
                      "eu_pallet_empty":"UK", "eu_pallet_loaded":"UK"}
DISPLAY_NAMES={"uk_pallet_empty":"eu_pallet_empty", "uk_pallet_loaded":"eu_pallet_loaded"}


# =====================================================================
# Import all helper classes/functions from custom_track_v22
# (copied here so this file is self-contained)
# =====================================================================

def save_zone(fp,pts,ppm,area,pa):
    d={"zone_pts":pts.tolist(),"ppm":float(ppm) if ppm else None,"real_area_m2":float(area) if area else None,
       "pixel_area":float(pa) if pa else None,"tag_size_cm":TAG_SIZE_CM,"saved_at":time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(fp,'w') as f: json.dump(d,f,indent=2)

def load_zone(fp):
    if not os.path.exists(fp): return None,None,None,None
    try:
        with open(fp,'r') as f: d=json.load(f)
        return np.array(d["zone_pts"],dtype=np.float32),d.get("ppm"),d.get("real_area_m2"),d.get("pixel_area")
    except: return None,None,None,None

def delete_zone_file(fp):
    if os.path.exists(fp): os.remove(fp)

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
    TEMP_SZ=500.0; temp_dst=np.array([[0,0],[TEMP_SZ,0],[TEMP_SZ,TEMP_SZ],[0,TEMP_SZ]],dtype=np.float32)
    M_temp=cv2.getPerspectiveTransform(zone_pts,temp_dst)
    edges_h,edges_v=[],[]
    for mc in aruco_corners:
        for i in range(4):
            p1=mc[0][i];p2=mc[0][(i+1)%4];w1=warp_pt(p1,M_temp);w2=warp_pt(p2,M_temp)
            elen=np.linalg.norm(np.array(w2)-np.array(w1))
            if abs(w2[0]-w1[0])>=abs(w2[1]-w1[1]): edges_h.append(elen)
            else: edges_v.append(elen)
    if not edges_h or not edges_v: return None,None,None
    tag_m=tag_cm/100.0;ppm_x=np.mean(edges_h)/tag_m;ppm_y=np.mean(edges_v)/tag_m
    if ppm_x<=0 or ppm_y<=0: return None,None,None
    rw=TEMP_SZ/ppm_x;rh=TEMP_SZ/ppm_y; return rw,rh,rw*rh

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

def compute_boundary_strip(zone_pts, edge, width_px):
    if zone_pts is None or width_px<=0: return None
    tl,tr,br,bl=zone_pts[0],zone_pts[1],zone_pts[2],zone_pts[3]
    if edge=="bottom":
        dl=(tl-bl);dl=dl/(np.linalg.norm(dl)+1e-9)*width_px
        dr=(tr-br);dr=dr/(np.linalg.norm(dr)+1e-9)*width_px
        return np.array([bl,br,br+dr,bl+dl],dtype=np.float32)
    elif edge=="top":
        dl=(bl-tl);dl=dl/(np.linalg.norm(dl)+1e-9)*width_px
        dr=(br-tr);dr=dr/(np.linalg.norm(dr)+1e-9)*width_px
        return np.array([tl,tr,tr+dr,tl+dl],dtype=np.float32)
    elif edge=="left":
        dt=(tr-tl);dt=dt/(np.linalg.norm(dt)+1e-9)*width_px
        db=(br-bl);db=db/(np.linalg.norm(db)+1e-9)*width_px
        return np.array([tl,tl+dt,bl+db,bl],dtype=np.float32)
    elif edge=="right":
        dt=(tl-tr);dt=dt/(np.linalg.norm(dt)+1e-9)*width_px
        db=(bl-br);db=db/(np.linalg.norm(db)+1e-9)*width_px
        return np.array([tr,tr+dt,br+db,br],dtype=np.float32)
    return None

def _overlap_ratio(box, zone_pts):
    x1,y1,x2,y2=box
    box_pts=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)
    box_area=(x2-x1)*(y2-y1)
    if box_area<=0: return 0.0
    zx_min=int(max(0,min(zone_pts[:,0].min(),x1)));zy_min=int(max(0,min(zone_pts[:,1].min(),y1)))
    zx_max=int(max(zone_pts[:,0].max(),x2)+1);zy_max=int(max(zone_pts[:,1].max(),y2)+1)
    rw=zx_max-zx_min;rh=zy_max-zy_min
    if rw<=0 or rh<=0: return 0.0
    zm=np.zeros((rh,rw),dtype=np.uint8);bm=np.zeros((rh,rw),dtype=np.uint8)
    zp_s=(zone_pts-[zx_min,zy_min]).astype(np.int32).reshape((-1,1,2))
    bp_s=(box_pts-[zx_min,zy_min]).astype(np.int32).reshape((-1,1,2))
    cv2.fillPoly(zm,[zp_s],255);cv2.fillPoly(bm,[bp_s],255)
    return cv2.countNonZero(cv2.bitwise_and(zm,bm))/box_area

def _box_in_strip(box, strip):
    if strip is None: return False
    x1,y1,x2,y2=box
    box_pts=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)
    if (x2-x1)*(y2-y1)<=0: return False
    sp=strip.astype(np.float32)
    sx_min=int(min(sp[:,0].min(),x1));sy_min=int(min(sp[:,1].min(),y1))
    sx_max=int(max(sp[:,0].max(),x2)+1);sy_max=int(max(sp[:,1].max(),y2)+1)
    rw=sx_max-sx_min;rh=sy_max-sy_min
    if rw<=0 or rh<=0: return False
    sm=np.zeros((rh,rw),dtype=np.uint8);bm=np.zeros((rh,rw),dtype=np.uint8)
    sp_s=(sp-[sx_min,sy_min]).astype(np.int32).reshape((-1,1,2))
    bp_s=(box_pts-[sx_min,sy_min]).astype(np.int32).reshape((-1,1,2))
    cv2.fillPoly(sm,[sp_s],255);cv2.fillPoly(bm,[bp_s],255)
    return cv2.countNonZero(cv2.bitwise_and(sm,bm))>0

def inzone(box,zp,overlap_thresh=0.3,boundary_strip=None,boundary_thresh=0.5):
    if zp is None: return False
    if boundary_strip is not None and _box_in_strip(box,boundary_strip):
        return _overlap_ratio(box,zp)>=boundary_thresh
    zp2=zp.astype(np.float32).reshape((-1,1,2))
    x1,y1,x2,y2=box;cx=(x1+x2)/2;cy=(y1+y2)/2
    if cv2.pointPolygonTest(zp2,(float(cx),float(cy)),False)>=0: return True
    if cv2.pointPolygonTest(zp2,(float(cx),float(y2)),False)>=0: return True
    return _overlap_ratio(box,zp)>=overlap_thresh

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

def calc_util_fixed(besz,dst,dets,zcam,M,ppx,ppy,boundary_strip=None,boundary_thresh=0.5):
    bw,bh=besz;zm=np.zeros((bh,bw),dtype=np.uint8)
    cv2.fillPoly(zm,[dst.astype(np.int32).reshape((-1,1,2))],255);za=cv2.countNonZero(zm)
    if za==0: return 0.0,0,0,{"US":0,"UK":0,"unknown":0},{}
    pm=np.zeros((bh,bw),dtype=np.uint8);counts={"US":0,"UK":0,"unknown":0};cls_counts={}
    for det in dets:
        box=det["box"];pt=det["pallet_type"];cn=det["cls_name"]
        if not inzone(box,zcam,boundary_strip=boundary_strip,boundary_thresh=boundary_thresh): continue
        if pt and pt in PALLET_TYPES: counts[pt]+=1;cls_counts[cn]=cls_counts.get(cn,0)+1
        else: counts["unknown"]+=1
        x1,y1,x2,y2=box
        corners_cam=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        corners_be=[warp_pt(c,M) for c in corners_cam]
        pts_be=np.array(corners_be,dtype=np.int32)
        pts_be[:,0]=np.clip(pts_be[:,0],0,bw-1);pts_be[:,1]=np.clip(pts_be[:,1],0,bh-1)
        cv2.fillConvexPoly(pm,pts_be,255)
    oa=cv2.countNonZero(cv2.bitwise_and(pm,zm)); return (oa/za)*100.0,za,oa,counts,cls_counts

GRID_COLS=20; GRID_ROWS=14

def compute_occupancy_grid(besz, dst, dets, zcam, M, ppx, ppy, boundary_strip=None, boundary_thresh=0.5):
    """
    Compute a grid showing which cells are occupied in bird's-eye space.
    Returns a 2D list: 0=empty zone, 1=US pallet, 2=EU pallet, 3=unknown pallet, -1=outside zone.
    Also returns per-detection positions for label overlay.
    """
    bw, bh = besz
    grid = [[-1]*GRID_COLS for _ in range(GRID_ROWS)]
    det_labels = []  # [{col, row, label, type}]

    # Zone mask
    zm = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(zm, [dst.astype(np.int32).reshape((-1,1,2))], 255)
    if cv2.countNonZero(zm) == 0:
        return grid, det_labels

    # Per-type pallet masks
    pm_us = np.zeros((bh, bw), dtype=np.uint8)
    pm_uk = np.zeros((bh, bw), dtype=np.uint8)
    pm_unk = np.zeros((bh, bw), dtype=np.uint8)

    for det in dets:
        box = det["box"]; pt = det["pallet_type"]; cn = det["cls_name"]
        if not inzone(box, zcam, boundary_strip=boundary_strip, boundary_thresh=boundary_thresh):
            continue
        x1,y1,x2,y2 = box
        corners_cam = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        corners_be = [warp_pt(c, M) for c in corners_cam]
        pts_be = np.array(corners_be, dtype=np.int32)
        pts_be[:,0] = np.clip(pts_be[:,0], 0, bw-1)
        pts_be[:,1] = np.clip(pts_be[:,1], 0, bh-1)

        if pt == "US":
            cv2.fillConvexPoly(pm_us, pts_be, 255)
        elif pt == "UK":
            cv2.fillConvexPoly(pm_uk, pts_be, 255)
        else:
            cv2.fillConvexPoly(pm_unk, pts_be, 255)

        # Record label position in grid coords
        cx = (pts_be[:,0].min() + pts_be[:,0].max()) / 2
        cy = (pts_be[:,1].min() + pts_be[:,1].max()) / 2
        gc = int(cx / max(bw,1) * GRID_COLS)
        gr = int(cy / max(bh,1) * GRID_ROWS)
        gc = max(0, min(GRID_COLS-1, gc))
        gr = max(0, min(GRID_ROWS-1, gr))
        det_labels.append({"col": gc, "row": gr, "label": cn, "type": pt or "unknown"})

    # Sample grid cells
    cell_w = bw / GRID_COLS
    cell_h = bh / GRID_ROWS
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx1 = int(c * cell_w); cy1 = int(r * cell_h)
            cx2 = int((c+1) * cell_w); cy2 = int((r+1) * cell_h)
            cx2 = min(cx2, bw); cy2 = min(cy2, bh)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            cell_zone = zm[cy1:cy2, cx1:cx2]
            zone_px = cv2.countNonZero(cell_zone)
            cell_area = (cx2-cx1)*(cy2-cy1)
            if zone_px < cell_area * 0.3:
                grid[r][c] = -1  # outside zone
                continue
            grid[r][c] = 0  # empty zone cell
            # Check which pallet type dominates this cell
            us_px = cv2.countNonZero(pm_us[cy1:cy2, cx1:cx2])
            uk_px = cv2.countNonZero(pm_uk[cy1:cy2, cx1:cx2])
            unk_px = cv2.countNonZero(pm_unk[cy1:cy2, cx1:cx2])
            max_px = max(us_px, uk_px, unk_px)
            if max_px > cell_area * 0.1:  # at least 10% of cell occupied
                if us_px == max_px: grid[r][c] = 1
                elif uk_px == max_px: grid[r][c] = 2
                else: grid[r][c] = 3

    return grid, det_labels

def ucol(p):
    if p<30:return(0,255,0)
    elif p<50:return(0,255,255)
    elif p<70:return(0,165,255)
    elif p<90:return(0,0,255)
    return(0,0,200)


# =====================================================================
# Camera loop — runs in a thread per camera
# =====================================================================

def camera_loop(zone_id, camera_idx, model_path, args, dashboard, stop_event):
    """Main loop for one camera. Runs in its own thread."""
    print(f"[ZONE {zone_id}] Starting camera {camera_idx}...")

    model = YOLO(model_path)
    rd = RobustArucoDetector(aruco.DICT_4X4_50)

    # Calibration
    ucal=False; m1=m2=cmx=dcc=None; cerr=0
    calib_file = args.calib_file_a if zone_id=="A" else args.calib_file_b
    if not args.no_calib:
        cmx,dcc,cerr=load_cal(calib_file)
        if cmx is not None: ucal=True; print(f"[ZONE {zone_id}] Calibrated (err:{cerr:.4f}px)")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ZONE {zone_id}] ERROR: Cannot open camera {camera_idx}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    udi = False

    zs = ZoneSmoother(alpha=SMOOTH_A, max_jump=MAX_JUMP_PX, hsize=ROLLING_N)
    lvp=None;ldt=0;lppm=None;lra=None;lpa=0;rm=args.ref_mode;us_val=0.0
    ft=time.time();fc=0;fps=0.0
    zone_locked=False;show_ui=True
    lM=None;ldst=None;lbw=BEW;lbh=BEH;lppx=lppy=100
    bstrip=None

    # Zone file and boundary config per zone
    zone_file = args.zone_file_a if zone_id=="A" else args.zone_file_b
    boundary_edge = args.boundary_edge_a if zone_id=="A" else args.boundary_edge_b
    boundary_width = args.boundary_width
    boundary_thresh = args.boundary_thresh
    conf = args.conf

    saved_pts,saved_ppm,saved_area,saved_pa = load_zone(zone_file)
    if saved_pts is not None:
        lvp=saved_pts;lppm=saved_ppm;lra=saved_area;lpa=saved_pa if saved_pa else 0
        zone_locked=True;ldt=time.time()
        if boundary_edge!="none":
            bstrip=compute_boundary_strip(lvp, boundary_edge, boundary_width)
        print(f"[ZONE {zone_id}] Restored locked zone from {zone_file}")

    win_name = f"Zone {zone_id} (Camera {camera_idx})"
    print(f"[ZONE {zone_id}] Running — boundary:{boundary_edge} conf:{conf}")

    while not stop_event.is_set():
        ret,raw=cap.read()
        if not ret: break
        if ucal and not udi: h,w=raw.shape[:2];m1,m2=setup_undist(cmx,dcc,w,h);udi=True
        frame=cv2.remap(raw,m1,m2,cv2.INTER_LINEAR) if ucal and m1 is not None else raw.copy()
        ct=time.time();gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        ac,ai,nd=rd.detect(gray);areg=get_areg(ac) if ac else [];st="SEARCHING";dq="NONE";pc=-1

        if zone_locked:
            st="LOCKED";dq="LOCKED"
            if lM is None and nd>=3 and lvp is not None:
                lppm=get_ppm(ac,args.tag_size)
                rdims=compute_real_dims_perspective(lvp,ac,args.tag_size)
                if rdims[0] is not None:
                    rw,rh,lra=rdims
                    if rw>0 and rh>0: lM,ldst,lbw,lbh,lppx,lppy=compute_be(lvp,rw,rh,BEW,BEH)
        elif nd>=4:
            rp=get_centers(ac) if rm=="CENTER" else get_outer(ac) if rm=="OUTER" else get_inner(ac)
            op=order_pts(rp);sp=zs.update(op)
            lvp=sp;ldt=ct;lpa=cv2.contourArea(sp);lppm=get_ppm(ac,args.tag_size)
            rdims=compute_real_dims_perspective(sp,ac,args.tag_size)
            if rdims[0] is not None:
                rw,rh,lra=rdims
                if rw>0 and rh>0: lM,ldst,lbw,lbh,lppx,lppy=compute_be(sp,rw,rh,BEW,BEH)
            st="DETECTED";dq="4/4"
            if boundary_edge!="none": bstrip=compute_boundary_strip(lvp,boundary_edge,boundary_width)
        elif nd==3 and lvp is not None:
            rp=get_centers(ac) if rm=="CENTER" else get_outer(ac) if rm=="OUTER" else get_inner(ac)
            asgn={};used=set()
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
                st="PARTIAL";dq=f"3/4"
                if boundary_edge!="none": bstrip=compute_boundary_strip(lvp,boundary_edge,boundary_width)

        if st=="SEARCHING" and lvp is not None:
            ts=ct-ldt
            if ts<args.memory_timeout: st="MEMORY"
            elif not zone_locked: st="LOST";lvp=None;zs.reset();bstrip=None
            else: st="LOCKED"

        res=model(frame,verbose=False,conf=conf);fd=fyolo(res,areg,ct=conf)

        counts={"US":0,"UK":0,"unknown":0};cls_counts={};oam=None;up=0.0;iz=oz=0
        if lvp is not None and lM is not None and ldst is not None:
            up,zpx,opx,counts,cls_counts=calc_util_fixed((lbw,lbh),ldst,fd,lvp,lM,lppx,lppy,
                                                          boundary_strip=bstrip,boundary_thresh=boundary_thresh)
            us_val=UTIL_A*up+(1-UTIL_A)*us_val
            if lppm and lppm>0 and zpx>0: oam=lra*(opx/zpx) if lra else None
            for det in fd:
                if inzone(det["box"],lvp,boundary_strip=bstrip,boundary_thresh=boundary_thresh): iz+=1
                else: oz+=1
        elif lvp is not None:
            for det in fd:
                if inzone(det["box"],lvp,boundary_strip=bstrip,boundary_thresh=boundary_thresh):
                    iz+=1;pt=det["pallet_type"];cn=det["cls_name"]
                    counts[pt if pt in counts else "unknown"]+=1
                else: oz+=1

        # --- Compute occupancy grid ---
        grid_data = None; grid_labels = []
        if lvp is not None and lM is not None and ldst is not None:
            grid_data, grid_labels = compute_occupancy_grid(
                (lbw,lbh), ldst, fd, lvp, lM, lppx, lppy,
                boundary_strip=bstrip, boundary_thresh=boundary_thresh)

        # --- Feed to unified dashboard ---
        dashboard.feed(zone_id, {
            "utilization":      round(float(us_val), 2),
            "zone_area_m2":     round(float(lra), 4) if lra else None,
            "occupied_area_m2": round(float(oam), 4) if oam else None,
            "counts":           counts,
            "cls_counts":       cls_counts,
            "zone_status":      st,
            "detection_quality": dq,
            "fps":              round(fps, 1),
            "markers_detected": nd,
            "pallets_total":    len(fd),
            "pallets_in_zone":  iz,
            "pallets_outside":  oz,
            "confidence_threshold": conf,
            "zone_locked":      zone_locked,
            "ppm":              round(float(lppm), 1) if lppm else None,
            "grid":             grid_data,
            "grid_labels":      grid_labels,
        })

        # --- Draw on frame ---
        if lvp is not None:
            pd=lvp.astype(int).reshape((-1,1,2));ov=frame.copy()
            cv2.fillPoly(ov,[pd],ucol(us_val));cv2.addWeighted(ov,0.15,frame,0.85,0,frame)
            zcol=(255,0,255) if zone_locked else (0,255,0);cv2.polylines(frame,[pd],True,zcol,2)
            if zone_locked:
                zc=np.mean(lvp,axis=0).astype(int);cv2.putText(frame,"LOCKED",(zc[0]-50,zc[1]),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,255),2)
            if bstrip is not None:
                bs_pts=bstrip.astype(int).reshape((-1,1,2));bs_ov=frame.copy()
                cv2.fillPoly(bs_ov,[bs_pts],(255,0,255));cv2.addWeighted(bs_ov,0.12,frame,0.88,0,frame)
                cv2.polylines(frame,[bs_pts],True,(255,0,255),1)

        # Draw detections
        for det in fd:
            box=det["box"];cf=det["conf"];cn=det["cls_name"];pt=det["pallet_type"]
            ins=inzone(box,lvp,boundary_strip=bstrip,boundary_thresh=boundary_thresh) if lvp is not None else False
            x1,y1,x2,y2=box
            col=PALLET_TYPES[pt]["color"] if pt and pt in PALLET_TYPES else (128,128,128)
            if not ins: col=tuple(c//2 for c in col)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            cv2.putText(frame,f"{cn} {cf:.2f}",(x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            cv2.putText(frame,"IN ZONE" if ins else "OUTSIDE",(x1,y2+18),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,2)

        if ai is not None and len(ac)>0: aruco.drawDetectedMarkers(frame,ac,ai)

        # Utilization bar
        h,w=frame.shape[:2]
        if lvp is not None:
            uc=ucol(us_val);ut=f"ZONE {zone_id}: {us_val:.1f}%"
            tsz,_=cv2.getTextSize(ut,cv2.FONT_HERSHEY_SIMPLEX,0.8,2);bxw=tsz[0]+40;bxx=(w-bxw)//2
            cv2.rectangle(frame,(bxx,5),(bxx+bxw,45),(0,0,0),-1);cv2.rectangle(frame,(bxx,5),(bxx+bxw,45),uc,2)
            cv2.putText(frame,ut,(bxx+20,33),cv2.FONT_HERSHEY_SIMPLEX,0.8,uc,2)

        fc+=1;el=time.time()-ft
        if el>1: fps=fc/el;fc=0;ft=time.time()
        cv2.putText(frame,f"FPS:{fps:.1f}",(w-130,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.putText(frame,f"Ref:{rm}",(w-130,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,0),1)

        cv2.imshow(win_name, frame)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): stop_event.set(); break
        elif key==ord('l'):
            if not zone_locked and lvp is not None:
                zone_locked=True;save_zone(zone_file,lvp,lppm,lra,lpa)
                if boundary_edge!="none": bstrip=compute_boundary_strip(lvp,boundary_edge,boundary_width)
                print(f"[ZONE {zone_id}] LOCKED & SAVED")
            elif zone_locked: zone_locked=False;zs.reset();print(f"[ZONE {zone_id}] UNLOCKED")
        elif key==ord('r'):
            lvp=lppm=lra=lM=None;lpa=0;us_val=0;zs.reset();zone_locked=False;bstrip=None
            delete_zone_file(zone_file);print(f"[ZONE {zone_id}] RESET")
        elif key==ord('c'):
            rm={"CENTER":"INNER","INNER":"OUTER","OUTER":"CENTER"}[rm]
            lvp=lra=lppm=lM=None;lpa=0;us_val=0;zs.reset();zone_locked=False;bstrip=None
            print(f"[ZONE {zone_id}] Ref mode: {rm}")
        elif key in [ord('+'),ord('=')]: conf=min(0.95,conf+0.05);print(f"[ZONE {zone_id}] Conf:{conf:.2f}")
        elif key==ord('-'): conf=max(0.05,conf-0.05);print(f"[ZONE {zone_id}] Conf:{conf:.2f}")
        elif key==ord('s'): cv2.imwrite(f"snap_{zone_id}_{time.strftime('%Y%m%d_%H%M%S')}.png",frame);print(f"[ZONE {zone_id}] Snap saved")

    cap.release()
    print(f"[ZONE {zone_id}] Stopped.")


# =====================================================================
# Main
# =====================================================================

def main():
    pa = argparse.ArgumentParser(description="Multi-Camera Staging Zone Tracker")
    pa.add_argument("--model", type=str, default=MODEL_PATH, help="YOLO model path")
    pa.add_argument("--camera-a", type=int, default=0, help="Camera index for Zone A")
    pa.add_argument("--camera-b", type=int, default=1, help="Camera index for Zone B")
    pa.add_argument("--zone-file-a", type=str, default="zone_A.json", help="Zone file for Camera A")
    pa.add_argument("--zone-file-b", type=str, default="zone_B.json", help="Zone file for Camera B")
    pa.add_argument("--calib-file-a", type=str, default="camera_calibration_a.npz")
    pa.add_argument("--calib-file-b", type=str, default="camera_calibration_b.npz")
    pa.add_argument("--boundary-edge-a", type=str, default="bottom", choices=["none","top","bottom","left","right"],
                    help="Boundary edge for Zone A (default: bottom)")
    pa.add_argument("--boundary-edge-b", type=str, default="top", choices=["none","top","bottom","left","right"],
                    help="Boundary edge for Zone B (default: top)")
    pa.add_argument("--boundary-width", type=int, default=80, help="Boundary strip width in pixels")
    pa.add_argument("--boundary-thresh", type=float, default=0.5, help="Overlap threshold for boundary")
    pa.add_argument("--tag-size", type=float, default=TAG_SIZE_CM)
    pa.add_argument("--ref-mode", type=str, default="CENTER", choices=["CENTER","INNER","OUTER"],
                    help="ArUco reference point: CENTER, INNER, or OUTER corners (default: CENTER)")
    pa.add_argument("--conf", type=float, default=CONF_THRESH)
    pa.add_argument("--memory-timeout", type=float, default=MEM_TIMEOUT)
    pa.add_argument("--no-calib", action="store_true")
    pa.add_argument("--port", type=int, default=5000, help="Dashboard port")
    pa.add_argument("--dashboard-interval", type=float, default=30, help="Dashboard snapshot interval")
    args = pa.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}"); sys.exit(1)

    # Start unified dashboard
    dashboard = MultiDashboardServer(port=args.port, interval=args.dashboard_interval)
    dashboard.start()

    print(f"\n{'='*60}")
    print(f"  MULTI-CAMERA STAGING ZONE TRACKER")
    print(f"{'='*60}")
    print(f"  Zone A: Camera {args.camera_a} | boundary: {args.boundary_edge_a}")
    print(f"  Zone B: Camera {args.camera_b} | boundary: {args.boundary_edge_b}")
    print(f"  Ref mode: {args.ref_mode} | Conf: {args.conf}")
    print(f"  Dashboard: http://localhost:{args.port}")
    print(f"  Interval: {args.dashboard_interval}s")
    print(f"{'='*60}")
    print(f"  Controls (per window):")
    print(f"    'l' = Lock/Unlock zone")
    print(f"    'r' = Reset zone")
    print(f"    'c' = Cycle ref mode (CENTER/INNER/OUTER)")
    print(f"    '+'/'-' = Adjust confidence")
    print(f"    's' = Save snapshot")
    print(f"    'q' = Quit all")
    print(f"{'='*60}\n")

    stop_event = threading.Event()

    thread_a = threading.Thread(
        target=camera_loop, daemon=True,
        args=("A", args.camera_a, args.model, args, dashboard, stop_event)
    )
    thread_b = threading.Thread(
        target=camera_loop, daemon=True,
        args=("B", args.camera_b, args.model, args, dashboard, stop_event)
    )

    thread_a.start()
    thread_b.start()

    # Wait for both to finish
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C — shutting down...")
        stop_event.set()

    thread_a.join(timeout=3)
    thread_b.join(timeout=3)
    dashboard.stop()
    cv2.destroyAllWindows()
    print("[INFO] All cameras stopped. Goodbye.")

if __name__ == "__main__":
    main()
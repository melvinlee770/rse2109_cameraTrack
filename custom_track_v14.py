"""
Staging Zone Tracker v14 - Stable Zone + Fixed Footprint Utilization
======================================================================
Combines:
  FROM v13_4: Robust multi-pass ArUco, zone smoothing, zone lock/save,
              partial detection (3/4), hide/show UI, memory timeout
  FROM v13:   Fixed pallet footprint (US/UK dimensions), bird's-eye warp,
              YOLO class-to-pallet mapping, perspective-corrected utilization

Usage:
    python custom_track_v14.py --model best_v4.pt
    python custom_track_v14.py --model best_v4.pt --zone-file my_zone.json
"""
import cv2, cv2.aruco as aruco, numpy as np, time, argparse, os, sys, json
from ultralytics import YOLO

TAG_SIZE_CM=2.4; MODEL_PATH="best_v4.pt"; CONF_THRESH=0.5
MEM_TIMEOUT=5.0; SMOOTH_A=0.1; UTIL_A=0.2; MAX_JUMP_PX=30; ROLLING_N=10
ZONE_SAVE_FILE="zone_data.json"; BEW=600; BEH=400

PALLET_TYPES={"US":{"width_m":1.219,"depth_m":1.016,"label":"US(48x40in)","color":(0,165,255)},
              "UK":{"width_m":1.200,"depth_m":0.800,"label":"UK(1200x800mm)","color":(255,165,0)}}
YOLO_CLASS_TO_PALLET={"uk_pallet_empty":"UK", "cage":"UK", "uk_pallet_loaded":"UK"}

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

def get_ppm(corners,tag_cm):
    el=[]
    for mc in corners:
        p=mc[0]
        for i in range(4): el.append(np.linalg.norm(p[(i+1)%4]-p[i]))
    return (np.mean(el)/tag_cm)*100.0 if el else None

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
        fl.append({"box":(int(x1),int(y1),int(x2),int(y2)),"conf":cf,"cls_id":ci,"cls_name":cn,
                    "pallet_type":YOLO_CLASS_TO_PALLET.get(cn,None)})
    return fl

def inzone(box,zp):
    if zp is None: return False
    cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
    return cv2.pointPolygonTest(zp.astype(np.float32).reshape((-1,1,2)),(float(cx),float(cy)),False)>=0

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

def calc_util_fixed(besz,dst,dets,zcam,M,ppx,ppy):
    bw,bh=besz; zm=np.zeros((bh,bw),dtype=np.uint8)
    cv2.fillPoly(zm,[dst.astype(np.int32).reshape((-1,1,2))],255); za=cv2.countNonZero(zm)
    if za==0: return 0.0,0,0,{"US":0,"UK":0,"unknown":0}
    pm=np.zeros((bh,bw),dtype=np.uint8); counts={"US":0,"UK":0,"unknown":0}
    for det in dets:
        box=det["box"];pt=det["pallet_type"]
        if not inzone(box,zcam): continue
        cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2;bx,by=warp_pt((cx,cy),M)
        if pt and pt in PALLET_TYPES:
            pw=PALLET_TYPES[pt]["width_m"]*ppx;pd=PALLET_TYPES[pt]["depth_m"]*ppy;counts[pt]+=1
        else: pw=1.2*ppx;pd=0.9*ppy;counts["unknown"]+=1
        fx1=max(0,int(bx-pw/2));fy1=max(0,int(by-pd/2));fx2=min(bw,int(bx+pw/2));fy2=min(bh,int(by+pd/2))
        if fx2>fx1 and fy2>fy1: cv2.rectangle(pm,(fx1,fy1),(fx2,fy2),255,-1)
    oa=cv2.countNonZero(cv2.bitwise_and(pm,zm)); return (oa/za)*100.0,za,oa,counts

def ucol(p):
    if p<30:return(0,255,0)
    elif p<50:return(0,255,255)
    elif p<70:return(0,165,255)
    elif p<90:return(0,0,255)
    return(0,0,200)

def ubar(f,p,x,y,bw=300,bh=22):
    c=ucol(p);cv2.rectangle(f,(x,y),(x+bw,y+bh),(50,50,50),-1);cv2.rectangle(f,(x,y),(x+bw,y+bh),(150,150,150),1)
    fw=int(bw*min(p,100)/100)
    if fw>0:cv2.rectangle(f,(x,y),(x+fw,y+bh),c,-1)
    t=f"{p:.1f}%";sz,_=cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
    cv2.putText(f,t,(x+(bw-sz[0])//2,y+(bh+sz[1])//2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

def ddets(f,dets,zp):
    iz=oz=0
    for det in dets:
        box=det["box"];cf=det["conf"];cn=det["cls_name"];pt=det["pallet_type"]
        ins=inzone(box,zp);x1,y1,x2,y2=box
        if pt and pt in PALLET_TYPES: col=PALLET_TYPES[pt]["color"];tl=pt
        else: col=(128,128,128);tl="???"
        if ins: iz+=1
        else: col=tuple(c//2 for c in col);oz+=1
        cv2.rectangle(f,(x1,y1),(x2,y2),col,2)
        lb=f"{tl} {cf:.2f}";ls,_=cv2.getTextSize(lb,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        cv2.rectangle(f,(x1,y1-ls[1]-8),(x1+ls[0]+4,y1),col,-1)
        cv2.putText(f,lb,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(f,"IN ZONE" if ins else "OUTSIDE",(x1,y2+18),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,2)
        if ins and pt and pt in PALLET_TYPES:
            cv2.putText(f,f"{PALLET_TYPES[pt]['width_m']:.2f}x{PALLET_TYPES[pt]['depth_m']:.2f}m",
                        (x1,y2+36),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,200),1)
    return iz,oz

def main():
    pa=argparse.ArgumentParser(description="Staging Zone v14")
    pa.add_argument("--camera",type=int,default=0);pa.add_argument("--model",type=str,default=MODEL_PATH)
    pa.add_argument("--calib-file",type=str,default="camera_calibration.npz");pa.add_argument("--no-calib",action="store_true")
    pa.add_argument("--tag-size",type=float,default=TAG_SIZE_CM);pa.add_argument("--conf",type=float,default=CONF_THRESH)
    pa.add_argument("--memory-timeout",type=float,default=MEM_TIMEOUT);pa.add_argument("--zone-file",type=str,default=ZONE_SAVE_FILE)
    args=pa.parse_args()

    if not os.path.exists(args.model): print(f"[ERROR] Model not found:{args.model}");sys.exit(1)
    print(f"[INFO] Loading YOLO:{args.model}"); model=YOLO(args.model)
    print(f"[INFO] Classes:{model.names}"); print(f"[INFO] Mapping:{YOLO_CLASS_TO_PALLET}")
    for k,v in PALLET_TYPES.items(): print(f"[INFO]   {k}:{v['width_m']}x{v['depth_m']}m")
    mcls=list(model.names.values())
    for yc in YOLO_CLASS_TO_PALLET:
        if yc not in mcls: print(f"[WARN] '{yc}' not in model classes:{mcls} — UPDATE YOLO_CLASS_TO_PALLET!")

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

    saved_pts,saved_ppm,saved_area,saved_pa=load_zone(args.zone_file)
    if saved_pts is not None:
        lvp=saved_pts;lppm=saved_ppm;lra=saved_area;lpa=saved_pa if saved_pa else 0
        zone_locked=True;ldt=time.time();print("[INFO] Restored locked zone")

    print(f"\n[INFO] Tag:{args.tag_size}cm Conf:{args.conf} Timeout:{args.memory_timeout}s")
    print(f"[INFO] Mode: FIXED FOOTPRINT (US/UK pallet dimensions)")
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

        if zone_locked: st="LOCKED";dq="LOCKED (press 'l' to unlock)"
        elif nd>=4:
            rp=get_centers(ac) if rm=="CENTER" else get_inner(ac);op=order_pts(rp);sp=zs.update(op)
            lvp=sp;ldt=ct;lpa=cv2.contourArea(sp);ppm=get_ppm(ac,args.tag_size)
            if ppm and ppm>0:
                lppm=ppm;lra=lpa/(ppm**2)
                rw=np.linalg.norm(sp[1]-sp[0])/ppm;rh=np.linalg.norm(sp[3]-sp[0])/ppm
                if rw>0 and rh>0: lM,ldst,lbw,lbh,lppx,lppy=compute_be(sp,rw,rh,BEW,BEH)
            st="DETECTED";dq="4/4";stats["4/4"]+=1
        elif nd==3 and lvp is not None:
            rp=get_centers(ac) if rm=="CENTER" else get_inner(ac);asgn={};used=set()
            for pi in range(len(rp)):
                bd=float('inf');bc=-1
                for ci in range(4):
                    if ci in used: continue
                    d=np.linalg.norm(rp[pi]-lvp[ci])
                    if d<bd: bd=d;bc=ci
                asgn[bc]=rp[pi];used.add(bc)
            if len(asgn)==3:
                fp,pc=predict_missing(asgn,list(asgn.keys()));sp=zs.update(fp)
                lvp=sp;ldt=ct;lpa=cv2.contourArea(sp);ppm=get_ppm(ac,args.tag_size)
                if ppm and ppm>0:
                    lppm=ppm;lra=lpa/(ppm**2)
                    rw=np.linalg.norm(sp[1]-sp[0])/ppm;rh=np.linalg.norm(sp[3]-sp[0])/ppm
                    if rw>0 and rh>0: lM,ldst,lbw,lbh,lppx,lppy=compute_be(sp,rw,rh,BEW,BEH)
                st="PARTIAL";dq=f"3/4(pred:{'TL,TR,BR,BL'.split(',')[pc]})";stats["3/4"]+=1

        if st=="SEARCHING" and lvp is not None:
            ts=ct-ldt
            if ts<args.memory_timeout: st="MEMORY";dq=f"memory({ts:.1f}s)";stats["memory"]+=1
            elif not zone_locked: st="LOST";lvp=None;zs.reset();stats["lost"]+=1
            else: st="LOCKED";dq="LOCKED"

        res=model(frame,verbose=False,conf=args.conf);fd=fyolo(res,areg,ct=args.conf)

        counts={"US":0,"UK":0,"unknown":0};oam=None;up=0.0
        if lvp is not None and lM is not None and ldst is not None:
            up,zpx,opx,counts=calc_util_fixed((lbw,lbh),ldst,fd,lvp,lM,lppx,lppy)
            us_val=UTIL_A*up+(1-UTIL_A)*us_val
            if lppm and lppm>0 and zpx>0: oam=lra*(opx/zpx) if lra else None
        elif lvp is not None:
            for det in fd:
                if inzone(det["box"],lvp):
                    pt=det["pallet_type"];counts[pt if pt in counts else "unknown"]+=1

        # === DRAW: Zone (ALWAYS visible) ===
        if lvp is not None:
            pd=lvp.astype(int).reshape((-1,1,2));ov=frame.copy()
            cv2.fillPoly(ov,[pd],ucol(us_val));cv2.addWeighted(ov,0.15,frame,0.85,0,frame)
            zcol=(255,0,255) if zone_locked else (0,255,0);cv2.polylines(frame,[pd],True,zcol,2)
            if zone_locked:
                zc=np.mean(lvp,axis=0).astype(int);cv2.putText(frame,"LOCKED",(zc[0]-40,zc[1]),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2)
            for i,(pt,lb) in enumerate(zip(lvp,["TL","TR","BR","BL"])):
                x,y=int(pt[0]),int(pt[1])
                if i==pc: cv2.circle(frame,(x,y),8,(0,255,255),2);cv2.putText(frame,f"{lb}?",(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),2)
                else: cv2.circle(frame,(x,y),5,(0,0,255),-1);cv2.putText(frame,lb,(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)
            # Dimensions ALWAYS visible (even when UI hidden)
            if lppm and lppm>0:
                for i in range(4):
                    p1=lvp[i];p2=lvp[(i+1)%4];mid=((p1+p2)/2).astype(int)
                    cv2.putText(frame,f"{np.linalg.norm(p2-p1)/lppm:.2f}m",(mid[0]-25,mid[1]-8),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),2)

        iz,oz=ddets(frame,fd,lvp)
        if ai is not None and len(ac)>0: aruco.drawDetectedMarkers(frame,ac,ai)

        # === Utilization (top center) — toggled by 'h' ===
        h,w=frame.shape[:2]
        if lvp is not None and show_ui:
            uc=ucol(us_val);ut=f"UTILIZATION: {us_val:.1f}%"
            tsz,_=cv2.getTextSize(ut,cv2.FONT_HERSHEY_SIMPLEX,0.8,2);bxw=tsz[0]+40;bxx=(w-bxw)//2
            cv2.rectangle(frame,(bxx,5),(bxx+bxw,72),(0,0,0),-1);cv2.rectangle(frame,(bxx,5),(bxx+bxw,72),uc,2)
            cv2.putText(frame,ut,(bxx+20,35),cv2.FONT_HERSHEY_SIMPLEX,0.8,uc,2)
            ubar(frame,us_val,bxx+10,45,bxw-20,20)

        # === Info panel — toggled by 'h' ===
        if show_ui:
            px,py_p=10,h-310;pw=430;cv2.rectangle(frame,(px,py_p),(px+pw,h-10),(0,0,0),-1)
            bc=((255,0,255) if st=="LOCKED" else (0,255,0) if st=="DETECTED" else (0,255,255) if st=="PARTIAL" else (0,165,255) if st=="MEMORY" else (0,0,255))
            cv2.rectangle(frame,(px,py_p),(px+pw,h-10),bc,2);y=py_p+22;lh=22
            cv2.putText(frame,f"ZONE: {st}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,bc,2);y+=lh
            cv2.putText(frame,f"Detection: {dq}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.35,bc,1);y+=lh
            if ucal: cv2.putText(frame,f"CALIBRATED(err:{cerr:.3f}px)",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,0),1)
            else: cv2.putText(frame,"UNCALIBRATED",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)
            y+=lh;cv2.putText(frame,f"Ref:{rm} | FIXED FOOTPRINT",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,200,0),1);y+=lh
            if lra: cv2.putText(frame,f"Zone:{lra:.4f}m^2",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2);y+=lh
            if oam: cv2.putText(frame,f"Occupied:{oam:.4f}m^2",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),2);y+=lh
            uc_us=PALLET_TYPES["US"]["color"];uc_uk=PALLET_TYPES["UK"]["color"]
            cv2.putText(frame,f"US:{counts['US']}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,uc_us,2)
            cv2.putText(frame,f"UK:{counts['UK']}",(px+120,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,uc_uk,2)
            if counts["unknown"]>0: cv2.putText(frame,f"?:{counts['unknown']}",(px+230,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(128,128,128),2)
            y+=lh;td=len(fd);rd2=len(res[0].boxes) if res[0].boxes is not None else 0
            cv2.putText(frame,f"Pallets:{td}({rd2-td} filtered)",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1);y+=lh
            cv2.putText(frame,f"  In:{iz} Out:{oz}",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,200),1);y+=lh
            if lppm: cv2.putText(frame,f"Scale:{lppm:.0f}px/m Markers:{nd}/4",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,200),1);y+=lh
            cv2.putText(frame,f"Conf:{args.conf:.2f} Timeout:{args.memory_timeout}s",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,200),1);y+=lh
            zf_st="SAVED" if zone_locked and os.path.exists(args.zone_file) else "none"
            cv2.putText(frame,f"Zone:{zf_st}({args.zone_file})",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(150,150,150),1);y+=lh
            cv2.putText(frame,"h=HIDE l=LOCK b=BIRD q=quit c=ref +/-=conf r=RESET",(px+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(150,150,150),1)

        fc+=1;el=time.time()-ft
        if el>1: fps=fc/el;fc=0;ft=time.time()
        if show_ui: cv2.putText(frame,f"FPS:{fps:.1f}",(w-130,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.imshow("Staging Zone v14",frame)

        # Bird's-eye window
        if show_be and lM is not None and lvp is not None:
            bei=cv2.warpPerspective(frame,lM,(lbw,lbh))
            cv2.polylines(bei,[ldst.astype(np.int32).reshape((-1,1,2))],True,(0,255,0),2)
            for det in fd:
                box=det["box"];pt=det["pallet_type"]
                if not inzone(box,lvp): continue
                cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2;bx,by=warp_pt((cx,cy),lM)
                if pt and pt in PALLET_TYPES: pw=PALLET_TYPES[pt]["width_m"]*lppx;pdv=PALLET_TYPES[pt]["depth_m"]*lppy;col=PALLET_TYPES[pt]["color"];lb=pt
                else: pw=1.2*lppx;pdv=0.9*lppy;col=(128,128,128);lb="???"
                fx1=max(0,int(bx-pw/2));fy1=max(0,int(by-pdv/2));fx2=min(lbw,int(bx+pw/2));fy2=min(lbh,int(by+pdv/2))
                if fx2>fx1 and fy2>fy1:
                    ov2=bei.copy();cv2.rectangle(ov2,(fx1,fy1),(fx2,fy2),col,-1);cv2.addWeighted(ov2,0.4,bei,0.6,0,bei)
                    cv2.rectangle(bei,(fx1,fy1),(fx2,fy2),col,2);cv2.putText(bei,lb,(fx1+4,fy1+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
            cv2.putText(bei,"BIRD'S-EYE(Fixed Footprints)",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.imshow("Bird's-Eye View",bei)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('h'): show_ui=not show_ui;print(f"[INFO] UI:{'ON' if show_ui else 'OFF'} (dims always visible)")
        elif key==ord('b'):
            show_be=not show_be
            if not show_be: cv2.destroyWindow("Bird's-Eye View")
            print(f"[INFO] Bird's-eye:{'ON' if show_be else 'OFF'}")
        elif key==ord('l'):
            if not zone_locked and lvp is not None: zone_locked=True;save_zone(args.zone_file,lvp,lppm,lra,lpa);print("[INFO] ZONE LOCKED & SAVED")
            elif zone_locked: zone_locked=False;zs.reset();print("[INFO] ZONE UNLOCKED")
            else: print("[INFO] No zone to lock")
        elif key==ord('c'): rm="CORNER" if rm=="CENTER" else "CENTER";lvp=lra=lppm=lM=None;lpa=0;us_val=0;zs.reset();zone_locked=False;print(f"[INFO] Ref:{rm}")
        elif key==ord('u') and cmx is not None: ucal=not ucal;lvp=lra=lppm=lM=None;lpa=0;us_val=0;zs.reset();zone_locked=False;print(f"[INFO] Undistort:{'ON' if ucal else 'OFF'}")
        elif key in [ord('+'),ord('=')]: args.conf=min(0.95,args.conf+0.05);print(f"[INFO] Conf:{args.conf:.2f}")
        elif key==ord('-'): args.conf=max(0.05,args.conf-0.05);print(f"[INFO] Conf:{args.conf:.2f}")
        elif key==ord('s'): cv2.imwrite(f"snap_{time.strftime('%Y%m%d_%H%M%S')}.png",frame);print("[INFO] Snap saved")
        elif key==ord('r'): lvp=lppm=lra=lM=None;lpa=0;us_val=0;zs.reset();zone_locked=False;delete_zone_file(args.zone_file);print("[INFO] RESET")

    tot=sum(stats.values())
    if tot>0:
        print(f"\n{'='*50}\n  DETECTION STATS\n{'='*50}")
        for k,v in stats.items(): print(f"  {k:>8}:{v:>6}({v/tot*100:.1f}%)")
        print(f"{'='*50}")
    cap.release();cv2.destroyAllWindows()

if __name__=="__main__": main()
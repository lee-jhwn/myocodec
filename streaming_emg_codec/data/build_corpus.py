import glob, os, json, hashlib, re, sys, argparse, numpy as np, h5py
from math import gcd
from scipy.signal import resample_poly
from streaming_emg_codec.data.readers import CONFIG, DATA_ROOT
D=DATA_ROOT; CACHE=os.environ.get("EMG_CORPUS_ROOT","data/corpus2k").rstrip("/")+"/"; TGT=2000

_POSE=None
def pose_split(fp):
    global _POSE
    if _POSE is None:
        _POSE={}
        import csv
        p=D+"emg2pose/emg2pose_metadata.csv"
        if os.path.exists(p):
            for row in csv.DictReader(open(p)):
                fn=row.get("filename") or row.get("recording") or ""
                _POSE[fn]=(row.get("split") or "train").lower()
    key=os.path.basename(fp).replace(".hdf5","")
    return _POSE.get(key, "train")

_QW_USER=None
def qwerty_user(fp):
    global _QW_USER
    if _QW_USER is None:
        import csv
        _QW_USER={r["session"]:r["user"] for r in csv.DictReader(open(D+"emg2qwerty/metadata.csv"))}
    return _QW_USER.get(os.path.basename(fp).replace(".hdf5",""), os.path.basename(fp))

def subject(ds, fp):
    b=os.path.basename(fp)
    pats={"hyser":r"subject(\d+)","grabmyo":r"participant(\d+)","csl":r"subject(\d+)",
          "emgepn":r"user(\d+)","meganepro":r"S(\d+)"}
    if ds in pats:
        m=re.search(pats[ds], fp); return f"{ds}_{m.group(1)}" if m else b
    if ds=="ninapro":
        db=re.search(r"DB(\d+)",fp); s=re.search(r"[Ss](\d+)_",b); return f"nina_DB{db and db.group(1)}_s{s and s.group(1)}"
    if ds=="emg2qwerty": return "q_user"+str(qwerty_user(fp))
    if ds=="capgmyo": return f"capg_{b[:3]}"
    if ds=="putemg": m=re.search(r"-(\d+)-",b); return f"put_{m.group(1)}" if m else b
    if ds=="gaddy": return "gaddy_"+os.path.basename(os.path.dirname(fp))
    return b

def split_of(ds, fp, subj, key=None):
    if ds=="emg2speech":
        h=int(hashlib.md5((key or subj).encode()).hexdigest(),16)%100
        return "train" if h<90 else ("val" if h<95 else "test")
    if ds=="emg2pose": return pose_split(fp)
    h=int(hashlib.md5(subj.encode()).hexdigest(),16)%100
    return "train" if h<80 else ("val" if h<90 else "test")

def resample(a, fs):
    if fs==TGT: return a
    g=gcd(int(fs),TGT); return resample_poly(a, TGT//g, int(fs)//g, axis=1).astype(np.float32)

def run(ds, worker=0, nworkers=1):
    pat, reader = CONFIG[ds]
    files=sorted(glob.glob(pat if pat.startswith('/') else D+pat, recursive=True))[worker::nworkers]
    os.makedirs(CACHE+ds, exist_ok=True)
    h5=h5py.File(f"{CACHE}{ds}/data_w{worker}.h5","a"); grp=h5.require_group("rec")
    man=open(f"{CACHE}manifest_{ds}_w{worker}.jsonl","a")
    n=0
    for fp in files:
        subj=subject(ds,fp); sp=split_of(ds,fp,subj)
        try: recs=reader(fp)
        except Exception as e: print("READ_ERR",fp,repr(e)[:60]); continue
        for i,rec in enumerate(recs):
            fs,a,_spov = (rec+(None,))[:3] if len(rec)>=3 else (rec[0],rec[1],None)
            key=f"{subj}_{os.path.basename(fp)}_{i}".replace("/","_")
            if key in grp: continue
            sp=_spov if _spov else split_of(ds,fp,subj,key)
            a=resample(a,fs)
            a=a[np.std(a,axis=1)>1e-6]
            if a.shape[0]==0 or a.shape[1]<200: continue
            grp.create_dataset(key, data=a.astype(np.float32), chunks=(a.shape[0],min(4000,a.shape[1])), compression="lzf")
            man.write(json.dumps({"dataset":ds,"h5":f"{CACHE}{ds}/data_w{worker}.h5","key":key,"subject":subj,"split":sp,"n_ch":int(a.shape[0]),"n_samp":int(a.shape[1])})+"\n")
            n+=1
        if n and n%500==0: man.flush(); print(ds,n,flush=True)
    man.flush(); h5.close()
    print(f"DONE {ds}: {n} recordings")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--worker",type=int,default=0); ap.add_argument("--nworkers",type=int,default=1); a=ap.parse_args()
    run(a.dataset, a.worker, a.nworkers)

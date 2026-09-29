import numpy as np, h5py, os, glob, json
import scipy.io as sio

DATA_ROOT = os.environ.get("EMG_DATA_ROOT", "data/raw").rstrip("/") + "/"

def _ct_from_tc(a):  # time-major [T,C] -> [C,T]
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    if a.ndim == 1: a = a[None, :]
    return a.T.copy()

def read_emg2pose(fp):
    with h5py.File(fp, "r") as f:
        emg = f["emg2pose/timeseries"]["emg"][:]      # [T,16]
    return [(2000, emg.T.astype(np.float32))]

def read_wfdb(fp, default_fs=2048):
    import wfdb
    r = wfdb.rdrecord(fp[:-4])
    return [(int(r.fs or default_fs), r.p_signal.T.astype(np.float32))]

MIN_NATIVE_FS = 1000

def read_ninapro(fp):
    m = sio.loadmat(fp)
    fs = int(m["frequency"][0,0]) if "frequency" in m else 2000
    if fs < MIN_NATIVE_FS:
        return []
    emg = m["emg"].T.astype(np.float32)                 # [C,T]
    rep = m["repetition"].ravel() if "repetition" in m else None
    stim = (m.get("restimulus", m.get("stimulus")))
    if rep is None or stim is None: return [(fs, emg)]
    stim = np.asarray(stim).ravel()
    chg = np.where((np.diff(rep)!=0) | (np.diff(stim)!=0))[0] + 1
    b = [0] + list(chg) + [len(rep)]; out=[]; smap={5:"test", 2:"val"}
    for a, c in zip(b[:-1], b[1:]):
        r = int(rep[a])
        if r == 0 or c-a < 200: continue
        out.append((fs, emg[:, a:c], smap.get(r, "train")))
    return out       # [C,T]

def read_capgmyo(fp):
    return [(1000, sio.loadmat(fp)["data"].T.astype(np.float32))]  # [128,T]

def read_csl(fp):
    m = sio.loadmat(fp)
    return [(2048, np.asarray(c, dtype=np.float32)) for c in m["gestures"].ravel()
            if np.asarray(c).ndim == 2]                # each [192,6144] already [C,T]

_GADDY=None
def _gaddy_map():
    global _GADDY
    if _GADDY is None:
        import json
        d=json.load(open(DATA_ROOT+"gaddy/testset_largedev.json"))
        _GADDY={"test":set(tuple(x) for x in d["test"]), "dev":set(tuple(x) for x in d["dev"])}
    return _GADDY

def read_gaddy(fp):
    import os, json
    sp="train"
    if "voiced" not in fp:
        ifp=fp.replace("_emg.npy","_info.json")
        if os.path.exists(ifp):
            o=json.load(open(ifp)); loc=(o.get("book"), o.get("sentence_index")); m=_gaddy_map()
            if loc in m["test"]: sp="test"
            elif loc in m["dev"]: sp="val"
    return [(1000, np.load(fp).T.astype(np.float32), sp)]

def read_putemg(fp):
    import pandas as pd
    df = pd.read_hdf(fp, "data")
    cols = [c for c in df.columns if str(c).upper().startswith("EMG")]
    return [(5120, df[cols].values.T.astype(np.float32))]

def read_meganepro(fp):
    try: m = sio.loadmat(fp); emg = m.get("emg")
    except NotImplementedError:
        with h5py.File(fp,"r") as f: emg = f["emg"][:].T if "emg" in f else None
    return [(2000, np.asarray(emg, dtype=np.float32).T)] if emg is not None else []

def read_emgepn(fp):
    d = json.load(open(fp)); out=[]
    for grp, sp in (("trainingSamples","train"), ("testingSamples","test")):
        g = d.get(grp) or {}
        for s in (g.values() if isinstance(g, dict) else g):
            e = s.get("emg") if isinstance(s, dict) else None
            if e: out.append((200, np.array([e[k] for k in sorted(e)], dtype=np.float32), sp))
    return out

def read_emg2qwerty(fp):
    with h5py.File(fp, "r") as f:
        ts = f["emg2qwerty/timeseries"]
        emg = np.concatenate([ts["emg_left"][:], ts["emg_right"][:]], axis=1)  # [T,32]
    return [(2000, emg.T.astype(np.float32))]

def read_emg2speech(fp):                              # emg_2khz.h5, /emg/<utt> = [31,T]
    out=[]
    with h5py.File(fp, "r") as f:
        for k in f["emg"]:
            out.append((2000, np.asarray(f["emg"][k][:], dtype=np.float32)))
    return out

# dataset -> (glob relative to $EMG_DATA_ROOT, reader)
CONFIG = {
  "emg2speech":("emg2speech/*_emg_2khz.h5", read_emg2speech),
  "emg2qwerty":("emg2qwerty/*.hdf5", read_emg2qwerty),
  "emg2pose": ("emg2pose/emg2pose_data/*.hdf5", read_emg2pose),
  "hyser":    ("hyser/**/*raw*.dat", read_wfdb),
  "grabmyo":  ("grabmyo/Session*/**/*.dat", read_wfdb),
  "ninapro":  ("ninapro/**/*.mat", read_ninapro),
  "putemg":   ("putemg/*.hdf5", read_putemg),
  "csl":      ("csl/subject*/session*/*.mat", read_csl),
  "capgmyo":  ("capgmyo/*.mat", read_capgmyo),
  "gaddy":    ("gaddy/**/*_emg.npy", read_gaddy),
  "emgepn":   ("emgepn/**/trainingJSON/user*/*.json", read_emgepn),
  "meganepro":("meganepro/*.mat", read_meganepro),
}

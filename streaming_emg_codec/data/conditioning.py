from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi

NOTCH_Q = 30.0
TRIM_MS = 150.0


PAPER_SPEC: dict[str, dict] = {
    'gaddy':      dict(hp=(3, 2.0), bp=None, notch=(60.0, 7), rate=1000,
                       src='code', act=True),
    'hyser':      dict(hp=None, bp=(8, 10.0, 500.0), notch=(50.0, 8), rate=2048,
                       src='paper', act=True),
    'csl':        dict(hp=None, bp=(4, 20.0, 400.0), notch=None, rate=2048,
                       src='paper', act=True),
    'emg2qwerty': dict(hp=None, bp=None, notch=None, rate=2000, src='paper+code', act=False),
    'emg2pose':   dict(hp=None, bp=None, notch=None, rate=2000, src='paper+code', act=False),
    'putemg':     dict(hp=None, bp=None, notch=None, rate=5120, src='paper', act=False),
    'grabmyo':    dict(hp=None, bp=None, notch=None, rate=2048, src='paper', act=False),
    'ninapro':    dict(hp=None, bp=None, notch=None, rate=2000, src='paper', act=False),
    'meganepro':  dict(hp=None, bp=None, notch=None, rate=2000, src='paper', act=False),
    'capgmyo':    dict(hp=None, bp=None, notch=None, rate=1000, src='paper', act=False),
    'emgepn':     dict(hp=None, bp=None, notch=(50.0, 1), rate=200, src='requested', act=True),
    'emg2speech': dict(hp=None, bp=None, notch=(50.0, 1), rate=2000, src='requested', act=True),
}


def paper_sos(dataset: str, fs: float):
    """The filter chain a dataset's own paper specifies, as [(label, sos), ...]."""
    spec = PAPER_SPEC.get(dataset)
    if not spec:
        return []
    chain = []
    if spec.get('hp'):
        order, corner = spec['hp']
        chain.append(('hp%g' % corner,
                      butter(order, corner / (0.5 * fs), btype='high', output='sos')))
    if spec.get('bp'):
        order, lo, hi = spec['bp']
        hi = min(hi, 0.45 * fs)
        chain.append(('bp%g-%g' % (lo, hi),
                      butter(order, [lo / (0.5 * fs), hi / (0.5 * fs)],
                             btype='band', output='sos')))
    if spec.get('notch'):
        f0, kmax = spec['notch']
        for k in range(1, int(kmax) + 1):
            fk = f0 * k
            if fk >= 0.45 * fs:
                break
            b, a = iirnotch(fk, NOTCH_Q, fs)
            chain.append(('notch%g' % fk, tf2sos(b, a)))
    return chain


def condition_paper(x: np.ndarray, fs: float, dataset: str, trim_ms: float = 0.0):
    """Apply a dataset's own documented preprocessing."""
    x = np.asarray(x, dtype=np.float64)
    chain = paper_sos(dataset, fs)
    y = x
    for _, sos in chain:
        y = _causal(sos, y)
    ntrim = int(round(trim_ms * fs / 1000.0))
    diag = {'dataset': dataset, 'stages': [lab for lab, _ in chain],
            'src': (PAPER_SPEC.get(dataset) or {}).get('src'), 'trimmed': 0}
    if ntrim and y.shape[-1] > 4 * ntrim:
        y = y[..., ntrim:]
        diag['trimmed'] = ntrim
    return y.astype(np.float32), diag


def _causal(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    zi = sosfilt_zi(sos)
    if x.ndim == 1:
        y, _ = sosfilt(sos, x, zi=zi * x[0])
        return y
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        out[c], _ = sosfilt(sos, x[c], zi=zi * x[c, 0])
    return out


def group_delay_ms(dataset: str, fs: float, at_hz=(20, 60, 100, 200, 400)):
    """Causal group delay of a dataset's chain in ms, at a few probe frequencies."""
    from scipy.signal import sosfreqz
    total, grid, mag = None, None, None
    for _, sos in paper_sos(dataset, fs):
        w, h = sosfreqz(sos, worN=16384, fs=fs)
        omega = 2.0 * np.pi * w / fs
        gd = -np.gradient(np.unwrap(np.angle(h)), omega)      # samples
        total = gd if total is None else total + gd
        mag = np.abs(h) if mag is None else mag * np.abs(h)
        grid = w
    if total is None:
        return {}
    out = {}
    for f in at_hz:
        if f >= 0.45 * fs:
            continue
        if float(np.interp(f, grid, mag)) < 0.1:
            out[f] = None
        else:
            out[f] = float(1000.0 * np.interp(f, grid, total) / fs)
    return out

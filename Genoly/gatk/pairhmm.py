"""
PairHMM (GATK): P(lectura | haplotipo) en log10.

Port del motor de likelihoods de HaplotypeCaller/GenotypeGVCFs de GATK
(Java + librería nativa C++ vía JNI). Tres implementaciones equivalentes:

- **native**: ``Genoly.gatk._pairhmm_c`` (extensión C incluida en el
  paquete, la más rápida; se compila en la instalación si hay toolchain).
- **torch**: DP affine lotada y vectorizada por pares, ejecutable en GPU
  CUDA o CPU (float64 para estabilidad).
- **numpy**: lazo de referencia sobre pares (siempre disponible).

``backend='auto'`` (por defecto) usa native si está compilado y torch si
no (CPU o CUDA según DeviceManager). Todas comparten las mismas fórmulas:

- Gotoh affine en log10 con gaps de apertura/extensión Phred.
- Gaps afines al inicio (la lectura puede empezar dentro del haplotipo)
  y gaps de fin libres (la lectura puede terminar antes que el haplotipo).
- Emisión de match/mismatch con las calidades de la lectura y la calidad
  fija del haplotipo (45, como GATK), con suelo ``min_prob`` (1e-4).
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from Genoly.gatk.reads import PHRED_MISSING

#: Calidad fija de las bases del haplotipo (GATK: HAPLOTYPE_BASE_QUALITY).
HAP_QUALITY = 45.0

#: Suelo de probabilidad de emisión (GATK: PairHMM.minProb).
MIN_PROB = 1e-4

#: Penalties Phred por defecto (HaplotypeCaller).
DEFAULT_GAP_OPEN = 10.0
DEFAULT_GAP_EXTEND = 1.0

ReadLike = Union[str, Tuple[str, Optional[str]]]

try:  # extensión C opcional (se compila en la instalación)
    from Genoly.gatk import _pairhmm_c as _native
except ImportError:  # pragma: no cover - depende del toolchain
    _native = None


def has_native() -> bool:
    """¿Está compilado el kernel nativo C del PairHMM?"""
    return _native is not None


_BASE_LUT = np.full(256, 4, dtype=np.uint8)
for _b, _c in zip(b'ACGT', range(4)):
    _BASE_LUT[_b] = _c
    _BASE_LUT[_b + 32] = _c  # minúsculas


def encode_bases(sequence: str) -> np.ndarray:
    """Secuencia a códigos uint8 (A=0, C=1, G=2, T=3, N/otro=4)."""
    raw = np.frombuffer(sequence.encode('ascii', errors='replace'),
                        dtype=np.uint8)
    return _BASE_LUT[raw].copy()


def _split_reads(reads: Sequence[ReadLike]) -> Tuple[List[str], List[Optional[str]]]:
    seqs: List[str] = []
    quals: List[Optional[str]] = []
    for item in reads:
        if isinstance(item, str):
            seqs.append(item)
            quals.append(None)
        else:
            seqs.append(item[0])
            quals.append(item[1])
    return seqs, quals


def _qual_matrix(quals: Sequence[Optional[str]], n: int, max_len: int) -> np.ndarray:
    out = np.full((n, max_len), float(PHRED_MISSING), dtype=np.float64)
    for i, q in enumerate(quals):
        if not q:
            continue
        row = np.frombuffer(q.encode('ascii', errors='replace'),
                            dtype=np.uint8).astype(np.float64) - 33.0
        k = min(row.size, max_len)
        out[i, :k] = row[:k]
    return out


def _logsum2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.maximum(a, b)
    with np.errstate(invalid='ignore'):
        out = m + np.log10(1.0 + np.power(10.0, np.minimum(a, b) - m))
    return np.where(np.asarray(m) <= -1e300, -np.inf, out)


def _logsumexp_np(vec: np.ndarray) -> float:
    """log10 de la suma de 10**vec (estabilidad numérica)."""
    m = float(np.max(vec))
    if not np.isfinite(m) or m <= -1e300:
        return -np.inf
    return m + float(np.log10(np.power(10.0, vec - m).sum()))


def _pairhmm_numpy_pair(rb: np.ndarray, rq: np.ndarray, R: int,
                        hb: np.ndarray, H: int, p_h: float,
                        go: float, ge: float, min_prob: float) -> float:
    if R <= 0 or H <= 0:
        return 0.0
    neg = -np.inf
    prvM = np.full(H + 1, neg)
    prvI = np.full(H + 1, neg)
    prvD = np.full(H + 1, neg)
    prvM[0] = 0.0
    prvD[1:] = go + np.arange(H) * ge

    acc_col = neg
    for r in range(1, R + 1):
        curM = np.full(H + 1, neg)
        curI = np.full(H + 1, neg)
        curD = np.full(H + 1, neg)
        curI[0] = go + (r - 1) * ge
        pr = np.power(10.0, -rq[r - 1] / 10.0)
        rbase = int(rb[r - 1])
        hcol = hb[:H]
        eq = (rbase < 4) & (hcol < 4) & (hcol == rbase)
        e = np.where(eq,
                     (1.0 - pr) * (1.0 - p_h) + pr * p_h,
                     pr * (1.0 - p_h) + (1.0 - pr) * p_h)
        np.maximum(e, min_prob, out=e)
        curM[1:] = (np.log10(e)
                    + _logsum2(_logsum2(prvM[:H], prvI[:H]), prvD[:H]))
        curI[1:] = _logsum2(prvM[1:] + go, prvI[1:] + ge)
        # la deleción es un recorrido secuencial por columnas:
        # curD[h] depende de curD[h-1] de la MISMA fila
        curD[1] = -np.inf
        for h in range(2, H + 1):
            curD[h] = _logsum2(curM[h - 1] + go, curD[h - 1] + ge)
        if r < R:
            acc_col = _logsum2(acc_col,
                               _logsum2(_logsum2(curM[H], curI[H]), curD[H]))
        prvM, prvI, prvD = curM, curI, curD

    # última fila (r = R): caminos con la lectura consumida en cualquier
    # columna del haplotipo (gaps de fin libres; incluye la esquina R,H)
    acc_row = _logsumexp_np(
        _logsum2(_logsum2(prvM[1:], prvI[1:]), prvD[1:]))
    total = _logsum2(np.float64(acc_col), np.float64(acc_row))
    return float(total)


def _pairhmm_numpy(reads: Sequence[ReadLike],
                   haps: Sequence[str],
                   *, hap_quality: float, gap_open: float, gap_extend: float,
                   min_prob: float) -> np.ndarray:
    seqs, quals = _split_reads(reads)
    n_r, n_h = len(seqs), len(haps)
    out = np.zeros((n_r, n_h), dtype=np.float64)
    if n_r == 0 or n_h == 0:
        return out
    p_h = 10.0 ** (-hap_quality / 10.0)
    go, ge = -gap_open / 10.0, -gap_extend / 10.0
    hap_codes = [encode_bases(h) for h in haps]
    hap_lens = [h.size for h in hap_codes]
    for i, seq in enumerate(seqs):
        rb = encode_bases(seq)
        rq = np.array([PHRED_MISSING] * len(seq) if not quals[i]
                      else [ord(c) - 33 for c in quals[i]],
                      dtype=np.float64)
        R = int(rb.size)
        for j, hc in enumerate(hap_codes):
            out[i, j] = _pairhmm_numpy_pair(rb, rq, R, hc, hap_lens[j],
                                            p_h, go, ge, min_prob)
    return out


def _pairhmm_torch(reads: Sequence[ReadLike],
                   haps: Sequence[str],
                   *, hap_quality: float, gap_open: float, gap_extend: float,
                   min_prob: float, device, chunk_pairs: int) -> np.ndarray:
    """DP affine lotada: se vectoriza sobre pares (lectura, haplotipo)."""
    import torch

    seqs, quals = _split_reads(reads)
    n_r, n_h = len(seqs), len(haps)
    out = np.zeros((n_r, n_h), dtype=np.float64)
    if n_r == 0 or n_h == 0:
        return out

    pairs = [(i, j) for i in range(n_r) for j in range(n_h)]
    p_h = 10.0 ** (-hap_quality / 10.0)
    go, ge = -gap_open / 10.0, -gap_extend / 10.0
    neg = float('-inf')
    min_prob_t = min_prob

    for start in range(0, len(pairs), chunk_pairs):
        chunk = pairs[start:start + chunk_pairs]
        b = len(chunk)
        max_r = max(len(seqs[i]) for i, _ in chunk)
        max_h = max(len(haps[j]) for _, j in chunk)

        codes = torch.zeros((b, max_r), dtype=torch.uint8)
        qmat = torch.full((b, max_r), float(PHRED_MISSING), dtype=torch.float64)
        hcodes = torch.zeros((b, max_h), dtype=torch.uint8)
        r_lens = torch.zeros(b, dtype=torch.long)
        h_lens = torch.zeros(b, dtype=torch.long)
        for k, (i, j) in enumerate(chunk):
            seq = seqs[i]
            rb = encode_bases(seq)
            codes[k, :rb.size] = torch.from_numpy(rb)
            r_lens[k] = rb.size
            if quals[i]:
                qrow = np.frombuffer(quals[i].encode('ascii'),
                                     dtype=np.uint8).astype(np.float64) - 33.0
                m = min(qrow.size, max_r)
                qmat[k, :m] = torch.from_numpy(qrow[:m])
            hc = encode_bases(haps[j])
            hcodes[k, :hc.size] = torch.from_numpy(hc)
            h_lens[k] = hc.size

        codes = codes.to(device)
        qmat = qmat.to(device)
        hcodes = hcodes.to(device)
        r_lens = r_lens.to(device)
        h_lens = h_lens.to(device)

        dev = device
        prvM = torch.full((b, max_h + 1), neg, dtype=torch.float64, device=dev)
        prvI = torch.full((b, max_h + 1), neg, dtype=torch.float64, device=dev)
        prvD = torch.full((b, max_h + 1), neg, dtype=torch.float64, device=dev)
        prvM[:, 0] = 0.0
        cols = torch.arange(max_h, device=dev, dtype=torch.float64)
        prvD[:, 1:] = go + cols * ge
        hidx = torch.arange(max_h + 1, device=dev)
        valid_h = hidx.unsqueeze(0) <= h_lens.unsqueeze(1)

        acc_col = torch.full((b,), neg, dtype=torch.float64, device=dev)
        acc_row = torch.full((b,), neg, dtype=torch.float64, device=dev)
        p_h_t = p_h
        go_t, ge_t = go, ge

        for r in range(1, max_r + 1):
            active = r <= r_lens
            if not bool(active.any()):
                break
            curM = torch.full_like(prvM, neg)
            curI = torch.full_like(prvI, neg)
            curD = torch.full_like(prvD, neg)
            curI[:, 0] = go_t + (r - 1) * ge_t

            rb = codes[:, r - 1].unsqueeze(1)
            qr = qmat[:, r - 1].unsqueeze(1)
            hb = hcodes
            eq = (rb < 4) & (hb < 4) & (hb == rb)
            pr = torch.pow(10.0, -qr / 10.0)
            e = torch.where(eq,
                            (1.0 - pr) * (1.0 - p_h_t) + pr * p_h_t,
                            pr * (1.0 - p_h_t) + (1.0 - pr) * p_h_t)
            e = torch.clamp(e, min=min_prob_t)

            diag = _logsum2_torch(_logsum2_torch(prvM[:, :max_h],
                                                 prvI[:, :max_h]),
                                  prvD[:, :max_h])
            curM[:, 1:] = torch.log10(e) + diag
            curI[:, 1:] = _logsum2_torch(prvM[:, 1:] + go_t,
                                         prvI[:, 1:] + ge_t)
            # la deleción es un recorrido secuencial por columnas:
            # curD[:, h] depende de curD[:, h-1] de la MISMA fila
            curD[:, 1] = _logsum2_torch(curM[:, 0] + go_t, curD[:, 0] + ge_t)
            for h in range(2, max_h + 1):
                curD[:, h] = _logsum2_torch(curM[:, h - 1] + go_t,
                                            curD[:, h - 1] + ge_t)

            curM = torch.where(valid_h, curM, torch.full_like(curM, neg))
            curI = torch.where(valid_h, curI, torch.full_like(curI, neg))
            curD = torch.where(valid_h, curD, torch.full_like(curD, neg))

            if r < int(max_r):
                last = _logsum2_torch(_logsum2_torch(curM[:, max_h],
                                                     curI[:, max_h]),
                                      curD[:, max_h])
                acc_col = torch.where(active & (r_lens > r), _logsum2_torch(acc_col, last), acc_col)

            snap = active & (r == r_lens)
            if bool(snap.any()):
                # última fila de la lectura: caminos con la lectura
                # consumida en cualquier columna (gaps de fin libres,
                # igual que el kernel C y el backend numpy)
                row_states = _logsum2_torch(
                    _logsum2_torch(curM[:, 1:], curI[:, 1:]), curD[:, 1:])
                acc_row = torch.where(
                    snap, _logsum2_torch(acc_row, _logsumexp_torch(row_states)),
                    acc_row)

            prvM, prvI, prvD = curM, curI, curD

        total = _logsum2_torch(acc_col, acc_row)
        degenerate = (r_lens == 0) | (h_lens == 0)
        total = torch.where(degenerate, torch.zeros_like(total), total)
        cpu = total.cpu().numpy().astype(np.float64)
        for k, (i, j) in enumerate(chunk):
            out[i, j] = float(cpu[k])

    return out


def _logsum2_torch(a, b):
    import torch
    m = torch.maximum(a, b)
    out = m + torch.log10(1.0 + torch.pow(10.0, torch.minimum(a, b) - m))
    return torch.where(m <= -1e300, torch.full_like(out, float('-inf')), out)


def _logsumexp_torch(x):
    """log10 de la suma por filas de 10**x (x: (b, H))."""
    import torch
    m = x.max(dim=1).values
    finite = torch.isfinite(m)
    # sin -inf/NaN intermedios: las filas degeneradas se sustituyen por 0
    safe_m = torch.where(finite, m, torch.zeros_like(m))
    safe_x = torch.where(finite.unsqueeze(1), x, torch.zeros_like(x))
    s = torch.pow(10.0, safe_x - safe_m.unsqueeze(1)).sum(dim=1)
    out = safe_m + torch.log10(s)
    return torch.where(finite & (m > -1e300), out,
                       torch.full_like(out, float('-inf')))


def _pairhmm_native(reads: Sequence[ReadLike],
                    haps: Sequence[str],
                    *, hap_quality: float, gap_open: float, gap_extend: float,
                    min_prob: float, chunk_pairs: int) -> np.ndarray:
    seqs, quals = _split_reads(reads)
    n_r, n_h = len(seqs), len(haps)
    out = np.zeros((n_r, n_h), dtype=np.float64)
    if n_r == 0 or n_h == 0:
        return out
    pairs = [(i, j) for i in range(n_r) for j in range(n_h)]
    for start in range(0, len(pairs), chunk_pairs):
        chunk = pairs[start:start + chunk_pairs]
        b = len(chunk)
        max_r = max(len(seqs[i]) for i, _ in chunk)
        max_h = max(len(haps[j]) for _, j in chunk)
        codes = np.zeros((b, max_r), dtype=np.uint8)
        qmat = np.full((b, max_r), float(PHRED_MISSING), dtype=np.float64)
        hcodes = np.zeros((b, max_h), dtype=np.uint8)
        r_lens = np.zeros(b, dtype=np.int32)
        h_lens = np.zeros(b, dtype=np.int32)
        dest = np.zeros(b, dtype=np.float64)
        for k, (i, j) in enumerate(chunk):
            rb = encode_bases(seqs[i])
            codes[k, :rb.size] = rb
            r_lens[k] = rb.size
            if quals[i]:
                qrow = np.frombuffer(quals[i].encode('ascii'),
                                     dtype=np.uint8).astype(np.float64) - 33.0
                m = min(qrow.size, max_r)
                qmat[k, :m] = qrow[:m]
            hc = encode_bases(haps[j])
            hcodes[k, :hc.size] = hc
            h_lens[k] = hc.size
        _native.pairhmm_batch(codes, qmat, r_lens, hcodes, h_lens,
                              max_r, max_h, hap_quality, gap_open,
                              gap_extend, min_prob, dest)
        for k, (i, j) in enumerate(chunk):
            out[i, j] = float(dest[k])
    return out


def pairhmm_log10(reads: Sequence[ReadLike],
                  haps: Sequence[str],
                  *,
                  hap_quality: float = HAP_QUALITY,
                  gap_open: float = DEFAULT_GAP_OPEN,
                  gap_extend: float = DEFAULT_GAP_EXTEND,
                  min_prob: float = MIN_PROB,
                  backend: str = 'auto',
                  device=None,
                  chunk_pairs: int = 4096) -> np.ndarray:
    """
    Matriz ``(n_lecturas, n_haplotipos)`` de log10 P(lectura | haplotipo).

    Args:
        reads: secuencias, o pares ``(secuencia, calidades Phred+33)``;
            sin calidades se usa :data:`PHRED_MISSING` (como en GATK,
            "calidad no disponible" != "calidad mala").
        haps: secuencias de haplotipo (en orientación de la referencia).
        hap_quality: calidad fija de las bases del haplotipo.
        gap_open / gap_extend: penalties Phred de hueco.
        min_prob: suelo de probabilidad de emisión.
        backend: ``'auto'``, ``'native'``, ``'torch'`` o ``'numpy'``.
        device: para el backend torch ('cuda', 'cpu', None = auto).
        chunk_pairs: pares por lote (acota la memoria de los backends
            lotados).

    Returns:
        Matriz float64 con log10 P(lectura_i | haplotipo_j).
    """
    if backend not in ('auto', 'native', 'torch', 'numpy'):
        raise ValueError(f"backend desconocido: {backend!r}")
    use_native = backend == 'native' or (backend == 'auto' and has_native())
    if backend == 'native' and not has_native():
        raise RuntimeError("el kernel nativo del PairHMM no está compilado")
    if use_native:
        return _pairhmm_native(reads, haps, hap_quality=hap_quality,
                               gap_open=gap_open, gap_extend=gap_extend,
                               min_prob=min_prob, chunk_pairs=chunk_pairs)
    if backend == 'torch' or backend == 'auto':
        from Genoly.core.device import DeviceManager
        dev = DeviceManager(device).device
        return _pairhmm_torch(reads, haps, hap_quality=hap_quality,
                              gap_open=gap_open, gap_extend=gap_extend,
                              min_prob=min_prob, device=dev,
                              chunk_pairs=chunk_pairs)
    return _pairhmm_numpy(reads, haps, hap_quality=hap_quality,
                          gap_open=gap_open, gap_extend=gap_extend,
                          min_prob=min_prob)

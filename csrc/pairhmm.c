/*
 * PairHMM de Genoly-GPU — port nativo del motor de likelihoods de
 * HaplotypeCaller / GenotypeGVCFs de GATK (que en Java delega en una
 * librería C++ nativa por JNI; aquí el kernel vive directamente en C y
 * se expone como extensión de Python).
 *
 * Calcula, para cada par (lectura, haplotipo), log10 P(lectura |
 * haplotipo) con un Gotoh affine en espacio log10 y sumatorias sobre
 * caminos (log-sum-exp):
 *
 *   - Gaps afines al inicio (la lectura puede empezar dentro del
 *     haplotipo pagando una sola apertura de gap) y gaps de fin libres
 *     (la lectura puede terminar antes que el haplotipo y viceversa),
 *     que es lo que HaplotypeCaller necesita al alinear lecturas con
 *     padding contra haplotipos ensamblados localmente.
 *   - Emisión de match/mismatch combinando las calidades de la lectura
 *     y la calidad fija del haplotipo (45, como en GATK), con suelo
 *     ``min_prob`` (1e-4, como el PairHMM de GATK).
 *
 * Las tres implementaciones (esta, NumPy y la de PyTorch) comparten
 * exactamente estas fórmulas; los tests verifican la paridad.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NEG_INF (-1.0e300)

static double logsum2(double a, double b)
{
    double t;
    if (a < b) { t = a; a = b; b = t; }
    if (a <= NEG_INF / 2.0) return NEG_INF;
    return a + log10(1.0 + pow(10.0, b - a));
}

static double logsum3(double a, double b, double c)
{
    return logsum2(logsum2(a, b), c);
}

/* Likelihood de un par (lectura, haplotipo). Devuelve 0 si ok. */
static int pairhmm_one(const unsigned char *rb, const double *rq, int R,
                       const unsigned char *hb, int H,
                       double p_h, double go, double ge,
                       double min_prob, double *out)
{
    double *buf, *curM, *curI, *curD, *prvM, *prvI, *prvD, *tmp;
    double pr, e, acc_col, acc_row;
    unsigned char rbase, hbase;
    int r, h, eq;

    if (R <= 0 || H <= 0) { *out = 0.0; return 0; }

    buf = (double *)malloc(6 * (size_t)(H + 1) * sizeof(double));
    if (buf == NULL) return -1;
    curM = buf;
    curI = curM + (H + 1);
    curD = curI + (H + 1);
    prvM = curD + (H + 1);
    prvI = prvM + (H + 1);
    prvD = prvI + (H + 1);

    /* fila r = 0: gaps afines al inicio sobre el haplotipo */
    prvM[0] = 0.0;
    prvI[0] = NEG_INF;
    prvD[0] = NEG_INF;
    for (h = 1; h <= H; h++) {
        prvM[h] = NEG_INF;
        prvI[h] = NEG_INF;
        prvD[h] = go + (h - 1) * ge;
    }

    acc_col = NEG_INF;

    for (r = 1; r <= R; r++) {
        curM[0] = NEG_INF;
        curD[0] = NEG_INF;
        curI[0] = go + (r - 1) * ge;
        pr = pow(10.0, -rq[r - 1] / 10.0);
        rbase = rb[r - 1];
        for (h = 1; h <= H; h++) {
            hbase = hb[h - 1];
            eq = (rbase < 4 && hbase < 4 && rbase == hbase);
            if (eq)
                e = (1.0 - pr) * (1.0 - p_h) + pr * p_h;
            else
                e = pr * (1.0 - p_h) + (1.0 - pr) * p_h;
            if (e < min_prob) e = min_prob;
            curM[h] = log10(e) + logsum3(prvM[h - 1], prvI[h - 1], prvD[h - 1]);
            curI[h] = logsum2(prvM[h] + go, prvI[h] + ge);
        }
        curD[1] = logsum2(curM[0] + go, curD[0] + ge);
        for (h = 2; h <= H; h++)
            curD[h] = logsum2(curM[h - 1] + go, curD[h - 1] + ge);

        /* suma de la columna h = H para filas r < R (gaps de fin libres
         * cuando el haplotipo se consume antes que la lectura) */
        if (r < R)
            acc_col = logsum2(acc_col,
                              logsum3(curM[H], curI[H], curD[H]));

        tmp = prvM; prvM = curM; curM = tmp;
        tmp = prvI; prvI = curI; curI = tmp;
        tmp = prvD; prvD = curD; curD = tmp;
    }

    /* Última fila (r = R): caminos con la lectura consumida; incluye la
     * esquina (R, H) exactamente una vez. */
    acc_row = NEG_INF;
    for (h = 1; h <= H; h++)
        acc_row = logsum2(acc_row, logsum3(prvM[h], prvI[h], prvD[h]));

    *out = logsum2(acc_col, acc_row);
    free(buf);
    return 0;
}

static PyObject *pairhmm_batch_py(PyObject *self, PyObject *args)
{
    PyObject *ro_codes_o, *ro_quals_o, *r_lens_o, *ho_codes_o, *h_lens_o;
    PyObject *out_o;
    Py_buffer rc, rq, rl, hc, hl, ob;
    int max_r, max_h;
    double hap_qual, gap_open, gap_extend, min_prob;
    Py_ssize_t n_pairs, i;
    double p_h, go, ge, *outp;
    int err = 0;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOOiiddddO:pairhmm_batch",
                          &ro_codes_o, &ro_quals_o, &r_lens_o,
                          &ho_codes_o, &h_lens_o,
                          &max_r, &max_h,
                          &hap_qual, &gap_open, &gap_extend, &min_prob,
                          &out_o))
        return NULL;

    if (max_r < 0 || max_h < 0) {
        PyErr_SetString(PyExc_ValueError, "longitudes máximas negativas");
        return NULL;
    }

    if (PyObject_GetBuffer(ro_codes_o, &rc, PyBUF_ND | PyBUF_FORMAT) != 0)
        return NULL;
    if (PyObject_GetBuffer(ro_quals_o, &rq, PyBUF_ND | PyBUF_FORMAT) != 0) {
        PyBuffer_Release(&rc);
        return NULL;
    }
    if (PyObject_GetBuffer(r_lens_o, &rl, PyBUF_ND | PyBUF_FORMAT) != 0) {
        PyBuffer_Release(&rc); PyBuffer_Release(&rq);
        return NULL;
    }
    if (PyObject_GetBuffer(ho_codes_o, &hc, PyBUF_ND | PyBUF_FORMAT) != 0) {
        PyBuffer_Release(&rc); PyBuffer_Release(&rq); PyBuffer_Release(&rl);
        return NULL;
    }
    if (PyObject_GetBuffer(h_lens_o, &hl, PyBUF_ND | PyBUF_FORMAT) != 0) {
        PyBuffer_Release(&rc); PyBuffer_Release(&rq); PyBuffer_Release(&rl);
        PyBuffer_Release(&hc);
        return NULL;
    }
    if (PyObject_GetBuffer(out_o, &ob, PyBUF_ND | PyBUF_FORMAT) != 0) {
        PyBuffer_Release(&rc); PyBuffer_Release(&rq); PyBuffer_Release(&rl);
        PyBuffer_Release(&hc); PyBuffer_Release(&hl);
        return NULL;
    }

    n_pairs = rl.len / (rl.itemsize > 0 ? rl.itemsize : 1);
    if (n_pairs < 0) n_pairs = 0;

    if (rc.len != n_pairs * (Py_ssize_t)max_r ||
        rq.len != n_pairs * (Py_ssize_t)max_r * (Py_ssize_t)sizeof(double) ||
        hc.len != n_pairs * (Py_ssize_t)max_h ||
        ob.len != n_pairs * (Py_ssize_t)sizeof(double) ||
        (rl.itemsize != 4 && rl.itemsize != 8) ||
        (hl.itemsize != 4 && hl.itemsize != 8)) {
        PyErr_SetString(PyExc_ValueError,
                        "formas incompatibles en pairhmm_batch");
        goto fail;
    }

    p_h = pow(10.0, -hap_qual / 10.0);
    go = -gap_open / 10.0;
    ge = -gap_extend / 10.0;
    outp = (double *)ob.buf;

    for (i = 0; i < n_pairs; i++) {
        int R, H;
        const unsigned char *rb = (const unsigned char *)rc.buf + i * max_r;
        const double *rqi = (const double *)rq.buf + i * max_r;
        const unsigned char *hb = (const unsigned char *)hc.buf + i * max_h;

        if (rl.itemsize == 4)
            R = ((const int *)rl.buf)[i];
        else
            R = (int)((const long long *)rl.buf)[i];
        if (hl.itemsize == 4)
            H = ((const int *)hl.buf)[i];
        else
            H = (int)((const long long *)hl.buf)[i];
        if (R > max_r) R = max_r;
        if (H > max_h) H = max_h;
        if (R < 0) R = 0;
        if (H < 0) H = 0;

        if (pairhmm_one(rb, rqi, R, hb, H, p_h, go, ge, min_prob,
                        &outp[i]) != 0) {
            err = 1;
            break;
        }
    }

    PyBuffer_Release(&rc); PyBuffer_Release(&rq); PyBuffer_Release(&rl);
    PyBuffer_Release(&hc); PyBuffer_Release(&hl); PyBuffer_Release(&ob);
    if (err)
        return PyErr_NoMemory();
    Py_RETURN_NONE;

fail:
    PyBuffer_Release(&rc); PyBuffer_Release(&rq); PyBuffer_Release(&rl);
    PyBuffer_Release(&hc); PyBuffer_Release(&hl); PyBuffer_Release(&ob);
    return NULL;
}

static PyMethodDef methods[] = {
    {"pairhmm_batch", pairhmm_batch_py, METH_VARARGS,
     "pairhmm_batch(read_codes, read_quals, read_lens, hap_codes,\n"
     "               hap_lens, max_read_len, max_hap_len,\n"
     "               hap_quality, gap_open, gap_extend, min_prob, out)\n"
     "\n"
     "Rellena ``out`` (float64[n]) con log10 P(read_i | hap_i). Los\n"
     "códigos de base son uint8 (A=0, C=1, G=2, T=3, N=4)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "Genoly.gatk._pairhmm_c",
    "PairHMM nativo (C) portado del motor de likelihoods de GATK.",
    -1,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__pairhmm_c(void)
{
    return PyModule_Create(&moduledef);
}

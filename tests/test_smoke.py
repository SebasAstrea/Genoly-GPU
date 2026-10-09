"""
Tests de humo para los modulos de Genoly-GPU.

Ejecutar desde la raiz del proyecto:
    python -m pytest tests -v
o directamente:
    python tests/test_smoke.py
"""

import os
import shutil
import unittest

import torch

from Genoly import (
    DeviceManager,
    FastaReader,
    FastaRecord,
    FastqReader,
    FastqRecord,
    GenomicBLUP,
    KmerCounter,
    LinearMixedModel,
    QualityAnalyzer,
    Read,
    SequenceEncoder,
    VariantCaller,
    build_kinship,
    prepare_quantitative_data,
    write_fasta,
    write_fastq,
)
from Genoly.core.gpu_setup import GpuSetup, recommend_cuda_tag

SEQ = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"


class TestDevice(unittest.TestCase):
    def test_device_manager(self):
        manager = DeviceManager()
        self.assertTrue(manager.device.type in ('cuda', 'cpu'))

    def test_get_device(self):
        d = DeviceManager('cpu').device
        self.assertEqual(d.type, 'cpu')

    def test_genoly_device_desde_entorno(self):
        """GENOLY_DEVICE fija el dispositivo por defecto del proceso."""
        original = os.environ.get('GENOLY_DEVICE')

        def restore():
            if original is None:
                os.environ.pop('GENOLY_DEVICE', None)
            else:
                os.environ['GENOLY_DEVICE'] = original

        self.addCleanup(restore)

        os.environ['GENOLY_DEVICE'] = 'cpu'
        self.assertEqual(DeviceManager().device.type, 'cpu')
        # el argumento explicito tiene prioridad sobre la variable
        os.environ['GENOLY_DEVICE'] = 'cpu'
        self.assertEqual(DeviceManager('cpu:0').device.type, 'cpu')
        # vacia o ausente = auto-deteccion
        os.environ['GENOLY_DEVICE'] = ''
        self.assertTrue(DeviceManager().device.type in ('cuda', 'cpu'))
        os.environ.pop('GENOLY_DEVICE', None)
        self.assertTrue(DeviceManager().device.type in ('cuda', 'cpu'))


class TestIO(unittest.TestCase):
    def setUp(self):
        self.tmpdir = os.path.join(os.path.dirname(__file__), '_tmp_io')
        os.makedirs(self.tmpdir, exist_ok=True)
        self.fasta_path = os.path.join(self.tmpdir, 'test.fasta')
        self.fastq_path = os.path.join(self.tmpdir, 'test.fastq')

    def tearDown(self):
        for p in (self.fasta_path, self.fastq_path):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(self.tmpdir)

    def test_fasta_roundtrip(self):
        records = [FastaRecord(id="seq1", sequence=SEQ, description="test")]
        write_fasta(self.fasta_path, records)
        loaded = FastaReader(self.fasta_path).read_all()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].sequence, SEQ)
        self.assertEqual(loaded[0].id, "seq1")
        self.assertEqual(loaded[0].header, "seq1 test")

    def test_fastq_roundtrip(self):
        records = [FastqRecord(id="read1", sequence=SEQ,
                               quality="I" * len(SEQ))]
        write_fastq(self.fastq_path, records)
        loaded = FastqReader(self.fastq_path).read_all()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].sequence, SEQ)
        self.assertEqual(loaded[0].scores, [40] * len(SEQ))

    def test_fastq_multiregistro(self):
        records = [
            FastqRecord(id="r1", sequence="ACGTAC", quality="IIIIII"),
            FastqRecord(id="r2", sequence="TTTTGG", quality="AAAAAA"),
            FastqRecord(id="r3", sequence=SEQ, quality="I" * len(SEQ)),
        ]
        write_fastq(self.fastq_path, records)
        loaded = FastqReader(self.fastq_path).read_all()
        self.assertEqual([r.id for r in loaded], ["r1", "r2", "r3"])
        self.assertEqual([r.sequence for r in loaded], ["ACGTAC", "TTTTGG", SEQ])
        self.assertEqual(loaded[1].quality, "AAAAAA")

    def test_fastq_multilinea_illumina(self):
        seq = "ACGTACGTAA"
        with open(self.fastq_path, "w") as fh:
            fh.write("@r1 desc\n")
            fh.write(seq + "\n")
            fh.write("+\n")
            fh.write("IIIII\n")
            fh.write("EEEEE\n")
            fh.write("@r2\n")
            fh.write("ACGT\n+\nIIII\n")
        loaded = FastqReader(self.fastq_path).read_all()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].id, "r1")
        self.assertEqual(loaded[0].quality, "IIIIIEEEEE")
        self.assertEqual(loaded[1].id, "r2")
        self.assertEqual(loaded[1].sequence, "ACGT")


class TestEncoding(unittest.TestCase):
    def test_roundtrip(self):
        enc = SequenceEncoder('cpu')
        tensor, lengths = enc.encode([SEQ, SEQ[:10]])
        decoded = enc.decode(tensor, lengths)
        self.assertEqual(decoded, [SEQ, SEQ[:10]])

    def test_one_hot_shape(self):
        enc = SequenceEncoder('cpu')
        onehot = enc.encode_one_hot([SEQ])
        self.assertEqual(tuple(onehot.shape), (1, len(SEQ), 5))

    def test_gc_reference(self):
        enc = SequenceEncoder('cpu')
        seq = "GCGCAT"
        tensor, _ = enc.encode([seq])
        # A=0,C=1,G=2,T=3
        valid = tensor < 4
        gc = ((tensor == 1) | (tensor == 2)) & valid
        self.assertEqual(gc.sum().item(), 4)


class TestQC(unittest.TestCase):
    def test_gc_content(self):
        qa = QualityAnalyzer('cpu')
        gc = qa.gc_content(["GCGCAT"]).item()
        self.assertAlmostEqual(gc, 4 / 6, places=5)

    def test_base_composition(self):
        qa = QualityAnalyzer('cpu')
        comp = qa.base_composition(["AAATTT"])
        self.assertEqual(comp['A'], 3)
        self.assertEqual(comp['T'], 3)
        self.assertEqual(comp['G'], 0)

    def test_trim_by_quality(self):
        qa = QualityAnalyzer('cpu')
        # Secuencia larga con calidad baja en la cola
        seq = "ACGTACGT"
        qual = "IIII" + "####"  # 4 bases buenas, 4 malas
        rec = FastqRecord(id="r1", sequence=seq, quality=qual)
        trimmed = qa.trim_by_quality([rec], min_quality=20, window_size=2)
        self.assertEqual(len(trimmed[0].sequence), 4)

    def test_filter_by_quality(self):
        qa = QualityAnalyzer('cpu')
        seq = "ACGTACGTACGT" * 3
        good = FastqRecord(id="g", sequence=seq, quality="I" * len(seq))
        bad = FastqRecord(id="b", sequence=seq, quality="!" * len(seq))
        kept = qa.filter_by_quality([good, bad], min_mean_quality=20,
                                    min_length=10)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].id, "g")


class TestKmer(unittest.TestCase):
    def test_count_known(self):
        kc = KmerCounter('cpu')
        values, counts = kc.count([SEQ], k=8)
        top = kc.decode_kmer(int(values[counts.argmax()].item()), 8)
        # Canonico correcto: 'CGTACGTA' agrupa su revcomp 'TACGTACG'
        # (8 + 8 ventanas) y supera a 'ACGTACGT' (9 ventanas, palindromico)
        self.assertEqual(top, "CGTACGTA")
        self.assertEqual(int(counts.max().item()), 16)

    def test_canonical(self):
        kc = KmerCounter('cpu')
        # 'ACGTACGT' y su reverse complement son iguales
        values, counts = kc.count(["ACGTACGT"], k=8, canonical=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(kc.decode_kmer(int(values[0].item()), 8), "ACGTACGT")

    def test_canonico_revcomp_referencia(self):
        kc = KmerCounter('cpu')
        seq = "GATTACAGATTACAGGATCCTAACGT"
        values, counts = kc.count([seq], k=5, canonical=True)
        obtenido = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
        idx = {b: i for i, b in enumerate('ACGT')}
        esperado = {}
        for i in range(len(seq) - 4):
            w = seq[i:i + 5]
            code = 0
            for ch in w:
                code = code * 4 + idx[ch]
            rc = 0
            for ch in reversed(w):
                rc = rc * 4 + (3 - idx[ch])
            clave = min(code, rc)
            esperado[clave] = esperado.get(clave, 0) + 1
        self.assertEqual(obtenido, esperado)

    def test_exactitud_k31(self):
        kc = KmerCounter('cpu')
        g = torch.Generator().manual_seed(1234)
        seq = ''.join('ACGT'[i] for i in torch.randint(0, 4, (4000,), generator=g).tolist())
        k = 31
        values, counts = kc.count([seq], k=k, canonical=False)
        obtenido = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
        idx = {b: i for i, b in enumerate('ACGT')}
        esperado = {}
        for i in range(len(seq) - k + 1):
            code = 0
            for ch in seq[i:i + k]:
                code = code * 4 + idx[ch]
            esperado[code] = esperado.get(code, 0) + 1
        self.assertEqual(obtenido, esperado)

    def test_spectrum(self):
        kc = KmerCounter('cpu')
        spec = kc.spectrum(["ACGT" * 25], k=4)
        self.assertTrue(len(spec) > 0)


class TestGpuSetup(unittest.TestCase):
    def test_recommend_cuda_tag(self):
        self.assertEqual(recommend_cuda_tag("12.4"), "cu124")
        self.assertEqual(recommend_cuda_tag("12.6"), "cu126")
        self.assertEqual(recommend_cuda_tag("13.1"), "cu130")
        self.assertIsNone(recommend_cuda_tag(None))

    def test_install_command(self):
        cmd = GpuSetup.install_command("cu126")
        self.assertIn("--index-url https://download.pytorch.org/whl/cu126", cmd)
        self.assertIn("torch", cmd)

    def test_detect_nvidia(self):
        info = GpuSetup.detect_nvidia()
        self.assertIsInstance(info.available, bool)
        # En máquinas sin GPU debe reportar no disponible sin fallar
        if info.available:
            self.assertIsNotNone(info.driver_version)
            self.assertIsNotNone(info.cuda_version)

    def test_torch_status_consistent(self):
        status = GpuSetup.torch_status()
        self.assertIn("installed", status)
        if status["installed"]:
            import torch
            self.assertEqual(status["cuda_available"], torch.cuda.is_available())


class TestVariantCaller(unittest.TestCase):
    def test_snv_detection(self):
        ref = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        mut_ref = list(ref)
        mut_ref[10] = 'T'  # referencia con SNV en pos 10
        mut_ref = "".join(mut_ref)

        reads = []
        for start in range(0, 20, 5):
            reads.append(Read(sequence=mut_ref[start:start + 15], start=start))
        # Duplicar cobertura
        reads = reads * 4

        vc = VariantCaller('cpu')
        variants = vc.call_variants(ref, reads, min_depth=3, min_alt_freq=0.8)
        snvs = [v for v in variants if v.type == 'SNV']
        self.assertEqual(len(snvs), 1)
        self.assertEqual(snvs[0].position, 11)  # 1-based
        self.assertEqual(snvs[0].ref, 'G')
        self.assertEqual(snvs[0].alt, 'T')


class TestQuantitativeLMM(unittest.TestCase):
    def _simulate(self, n=200, m=400, true_g=0.6, true_e=0.4, seed=42):
        torch.manual_seed(seed)
        genotypes = torch.randint(0, 3, (n, m)).float()
        K = build_kinship(genotypes)
        A = torch.linalg.cholesky(K + 1e-6 * torch.eye(n))
        u = (A @ torch.randn(n, dtype=torch.float64)) * true_g ** 0.5
        y = (u + torch.randn(n, dtype=torch.float64) * true_e ** 0.5).float()
        return y, u, K

    def test_kinship_symmetry(self):
        torch.manual_seed(0)
        grm = build_kinship(torch.randint(0, 3, (20, 60)).float())
        self.assertEqual(grm.shape, (20, 20))
        self.assertTrue(torch.allclose(grm, grm.T, atol=1e-8))

    def test_kinship_identical_individuals(self):
        gt = torch.tensor([[0., 2., 1., 2.],
                           [0., 2., 1., 2.],
                           [2., 0., 2., 0.]])
        grm = build_kinship(gt)
        self.assertAlmostEqual(grm[0, 1].item(), grm[0, 0].item(), places=6)

    def test_kinship_monomorphic_raises(self):
        with self.assertRaises(ValueError):
            build_kinship(torch.zeros(10, 5))

    def test_kinship_gcta_symmetry_and_identical(self):
        torch.manual_seed(0)
        gt = torch.randint(0, 3, (20, 60)).float()
        grm = build_kinship(gt, method="gcta")
        self.assertEqual(grm.shape, (20, 20))
        self.assertTrue(torch.allclose(grm, grm.T, atol=1e-8))
        dup = torch.cat([gt[:1], gt[:1], gt[2:]], dim=0)
        grm_dup = build_kinship(dup, method="gcta")
        self.assertAlmostEqual(
            grm_dup[0, 1].item(), grm_dup[0, 0].item(), places=6)

    def test_kinship_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            build_kinship(torch.randint(0, 3, (5, 10)).float(), method="noexiste")

    def test_fit_recovers_heritability_and_blup(self):
        y, u_true, K = self._simulate()
        model = LinearMixedModel('cpu')
        res = model.fit(y, torch.ones(y.shape[0], 1), K, max_iter=300)
        self.assertTrue(res.converged)
        self.assertLess(abs(res.heritability - 0.6), 0.15)
        corr = torch.corrcoef(torch.stack([u_true, model.blup()]))[0, 1].item()
        self.assertGreater(corr, 0.5)

    def test_blue_intercept_equals_mean(self):
        torch.manual_seed(7)
        y = 3.0 + torch.randn(150)
        model = LinearMixedModel('cpu')
        model.fit(y, torch.ones(150, 1), torch.eye(150))
        self.assertAlmostEqual(model.blue().item(), 3.0, delta=0.15)

    def test_predict_shape(self):
        y, _, K = self._simulate(n=80, m=100, seed=3)
        model = LinearMixedModel('cpu')
        model.fit(y, torch.ones(80, 1), K)
        self.assertEqual(tuple(model.predict().shape), (80,))

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            LinearMixedModel('cpu').fit(
                torch.randn(50), torch.ones(50, 1), torch.eye(50), method="noexiste")

    def test_unfitted_access_raises(self):
        with self.assertRaises(RuntimeError):
            LinearMixedModel('cpu').blup()


class TestGBLUP(unittest.TestCase):
    def _simulate(self, n=150, m=300, true_g=0.6, true_e=0.4, seed=11):
        torch.manual_seed(seed)
        genotypes = torch.randint(0, 3, (n, m)).float()
        K = build_kinship(genotypes)
        A = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
        u = (A @ torch.randn(n, dtype=torch.float64)) * true_g ** 0.5
        y = (u + torch.randn(n, dtype=torch.float64) * true_e ** 0.5).float()
        return y, u, K

    def test_known_variances_reliabilities_and_accuracy(self):
        y, u_true, K = self._simulate()
        model = GenomicBLUP('cpu')
        res = model.fit(y, torch.ones(y.shape[0], 1), K,
                        genetic_variance=0.6, residual_variance=0.4)
        self.assertEqual(res.variance_source, 'dadas')
        self.assertIsNone(res.log_likelihood)
        rel = model.reliabilities()
        self.assertTrue(bool(((rel >= 0) & (rel <= 1)).all()))
        acc = model.accuracies()
        self.assertTrue(torch.allclose(acc, rel.sqrt()))
        corr = torch.corrcoef(torch.stack([u_true, model.blup()]))[0, 1].item()
        self.assertGreater(corr, 0.5)

    def test_reliability_matches_mme_reference(self):
        """La fiabilidad debe usar el PEV con efectos fijos ESTIMADOS.

        Referencia canonica: bloque inferior derecho de la inversa de las
        ecuaciones del modelo mixto de Henderson.
        """
        g = torch.Generator().manual_seed(99)
        q = 30
        W = torch.randn(q, 50, generator=g, dtype=torch.float64) / 10.0
        K = W @ W.T + 0.2 * torch.eye(q)
        u = torch.linalg.cholesky(K) @ torch.randn(
            q, 1, generator=g, dtype=torch.float64)
        X = torch.cat([torch.ones(q, 1),
                       torch.randn(q, 1, generator=g, dtype=torch.float64)],
                      dim=1)
        Z = torch.eye(q, dtype=torch.float64)
        beta = torch.tensor([[1.0], [0.7]], dtype=torch.float64)
        y = (X @ beta + Z @ u + 0.8 * torch.randn(
            q, 1, generator=g, dtype=torch.float64)).reshape(-1)

        var_g, var_e = 1.2, 0.64
        model = GenomicBLUP('cpu')
        res = model.fit(y, X, K, genetic_variance=var_g,
                        residual_variance=var_e)

        Rinv = torch.eye(q, dtype=torch.float64) / var_e
        Ginv = torch.linalg.inv(var_g * K)
        mme = torch.cat([
            torch.cat([X.T @ Rinv @ X, X.T @ Rinv @ Z], dim=1),
            torch.cat([Z.T @ Rinv @ X, Z.T @ Rinv @ Z + Ginv], dim=1),
        ], dim=0)
        Cinv = torch.linalg.inv(mme)
        pev = torch.diagonal(Cinv[X.shape[1]:, X.shape[1]:])
        rel_ref = (1.0 - pev / (var_g * torch.diagonal(K))).clamp(0.0, 1.0)
        self.assertTrue(torch.allclose(res.reliabilities, rel_ref, atol=1e-8))

    def test_estimates_variances_when_missing(self):
        y, _, K = self._simulate(n=200, m=400, seed=12)
        model = GenomicBLUP('cpu')
        res = model.fit(y, torch.ones(y.shape[0], 1), K)
        self.assertEqual(res.variance_source, 'reml')
        self.assertIsNotNone(res.log_likelihood)
        h2 = res.genetic_variance / (res.genetic_variance + res.residual_variance)
        self.assertLess(abs(h2 - 0.6), 0.15)

    def test_single_variance_raises(self):
        y, _, K = self._simulate(n=40, m=50, seed=5)
        with self.assertRaises(ValueError):
            GenomicBLUP('cpu').fit(y, torch.ones(40, 1), K,
                                   genetic_variance=0.5)

    def test_predict_shape(self):
        y, _, K = self._simulate(n=40, m=50, seed=6)
        model = GenomicBLUP('cpu')
        model.fit(y, torch.ones(40, 1), K)
        self.assertEqual(tuple(model.predict().shape), (40,))

    def test_unfitted_access_raises(self):
        with self.assertRaises(RuntimeError):
            GenomicBLUP('cpu').blup()


class TestDataset(unittest.TestCase):
    """Descubrimiento y registro de datasets NCBI (múltiples FASTA)."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        from ui.backend import datasets as ds
        self.ds = ds
        self.tmp = tempfile.mkdtemp(prefix='ds_test_')
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._orig_dir = ds._DATASETS_DIR
        ds._DATASETS_DIR = Path(self.tmp)
        ds._REGISTRY_PATH = Path(self.tmp) / 'registry.json'

    def tearDown(self):
        import shutil
        self.ds._DATASETS_DIR = self._orig_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mkfasta(self, path, seqs):
        from Genoly import FastaRecord, write_fasta
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_fasta(path, [FastaRecord(id=f's{i}', sequence=s)
                           for i, s in enumerate(seqs)], line_width=20)

    def test_discover_fastas_recursivo(self):
        base = os.path.join(self.tmp, 'ncbi_dataset', 'data')
        self._mkfasta(os.path.join(base, 'GCA_X', 'a.fna'), ['ACGT' * 5])
        self._mkfasta(os.path.join(base, 'GCF_Y', 'b.fasta'), ['TTTT' * 5])
        with open(os.path.join(base, 'nota.txt'), 'w') as fh:
            fh.write('no fasta')
        fastas = self.ds.discover_fastas(os.path.join(self.tmp, 'ncbi_dataset'))
        self.assertEqual(len(fastas), 2)
        self.assertTrue(all(p.suffix.lower() in ('.fna', '.fasta')
                            for p in fastas))

    def test_register_dataset_escanea_en_segundo_plano(self):
        import time
        d = os.path.join(self.tmp, 'data')
        self._mkfasta(os.path.join(d, 'GCA_X', 'a.fna'),
                      ['ACGT' * 5, 'TTTT' * 4])
        self._mkfasta(os.path.join(d, 'GCF_Y', 'b.fna'), ['GGGG' * 5])
        meta = self.ds.register_dataset(d)
        self.assertEqual(len(meta['files']), 2)
        for _ in range(100):
            cur = self.ds.get_dataset(meta['dataset_id'])
            if cur['status'] != 'pending':
                break
            time.sleep(0.05)
        self.assertEqual(cur['status'], 'ready')
        self.assertEqual([f['records'] for f in cur['files']], [2, 1])

    def test_zip_slip_rechazado(self):
        import zipfile
        zip_path = os.path.join(self.tmp, 'evil.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('../escapado.fna', '>x\nACGT\n')
        with self.assertRaises(Exception):
            self.ds.register_dataset(zip_path)


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.tmpdir = os.path.join(os.path.dirname(__file__), '_tmp_prep')
        os.makedirs(self.tmpdir, exist_ok=True)

    def tearDown(self):
        for name in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, name))
        os.rmdir(self.tmpdir)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        return path

    def test_csv_header_drops_column_and_imputes_mean(self):
        path = self._write(
            'datos.csv',
            "feno,m1,m2,sexo\n"
            "1.0,0,2,M\n"
            "2.0,1,,F\n"
            "0.5,2,1,M\n"
            "1.5,,0,F\n"
            "2.5,1,2,\n"
            "0.8,0,1,F\n",
        )
        pheno, geno, rep = prepare_quantitative_data(path)
        self.assertTrue(rep.header_detected)
        self.assertEqual(rep.dropped_columns, ['sexo'])
        self.assertEqual(rep.imputed_cells, 2)
        self.assertEqual(rep.final_rows, 6)
        self.assertEqual(rep.final_markers, 2)
        self.assertEqual(len(pheno), 6)
        self.assertAlmostEqual(geno[1][1], 1.2)
        self.assertAlmostEqual(geno[3][0], 0.8)

    def test_mode_imputation(self):
        path = self._write(
            'moda.csv',
            "feno,m1,m2\n"
            "1.0,,2\n"
            "2.0,1,0\n"
            "0.5,0,2\n"
            "1.5,1,1\n"
            "2.5,0,2\n"
            "0.8,1,0\n",
        )
        pheno, geno, rep = prepare_quantitative_data(path, impute_method='moda')
        self.assertEqual(rep.imputed_cells, 1)
        self.assertEqual(geno[0][0], 1)

    def test_semicolon_delimiter_and_decimal_comma(self):
        path = self._write(
            'es.csv',
            "feno;m1;m2\n"
            "2,5;0;2\n"
            "1,0;1;1\n"
            "0,5;2;0\n"
            "1,5;1;2\n"
            "2,5;0;1\n",
        )
        pheno, geno, _ = prepare_quantitative_data(path)
        self.assertAlmostEqual(pheno[0], 2.5)
        self.assertAlmostEqual(pheno[-1], 2.5)

    def test_excel_file(self):
        openpyxl = __import__('openpyxl')
        wb = openpyxl.Workbook()
        ws = wb.active
        rows = [
            ['feno', 'm1', 'm2'],
            [1.0, 0, 2], [2.0, 1, 0], [0.5, 2, 1],
            [1.5, '', 0], [2.5, 1, 2], [0.8, 0, 1],
        ]
        for row in rows:
            ws.append(row)
        path = os.path.join(self.tmpdir, 'datos.xlsx')
        wb.save(path)

        pheno, geno, rep = prepare_quantitative_data(path)
        self.assertEqual(rep.final_rows, 6)
        self.assertEqual(rep.final_markers, 2)
        self.assertAlmostEqual(geno[3][0], (0 + 1 + 2 + 1 + 0) / 5)

    def test_too_few_individuals_raises(self):
        path = self._write('corto.csv', "feno,m1\n1.0,0\n2.0,1\n0.5,2\n")
        with self.assertRaises(ValueError):
            prepare_quantitative_data(path)

    def test_clean_grid_robusto(self):
        from Genoly.quantitative.preprocess import (
            clean_grid,
            load_grid,
            profile_grid,
            scan_grid_stats,
        )
        path = os.path.join(self.tmpdir, 'sucio.csv')
        with open(path, 'w') as fh:
            fh.write('id;feno;m1;m2;m3;nota\n'
                     'A1;12,5;1;0;2;buena\n'
                     'A2;11;0;;1;regular\n'
                     'A3;;2;1;;mala\n'
                     'A4;13,2;;1;0;\n'
                     'A5;10,8;1;2;1;\n'
                     'A6;14;0;1;2;\n')
        stats = scan_grid_stats(path)
        self.assertEqual(stats['rows'], 6)
        self.assertTrue(stats['header_detected'])
        grid = load_grid(path)
        prof = profile_grid(grid)
        names = {c['name']: c for c in prof['columns']}
        self.assertEqual(names['m2']['type'], 'mixed')
        self.assertGreater(names['m2']['missing_pct'], 0)
        self.assertEqual(names['nota']['type'], 'text')

        pheno, geno, markers, pedigree, rep = clean_grid(
            grid, phenotype_col=1, impute_method='media',
            max_column_missingness=0.5, min_individuals=5, min_markers=2)
        self.assertIsNone(pedigree)
        self.assertEqual(rep['final_rows'], 5)
        self.assertEqual(rep['final_markers'], 3)
        self.assertEqual(rep['phenotype_column'], 'feno')
        self.assertEqual(rep['imputed_cells'], 2)
        self.assertIn('nota', [dc['name'] for dc in rep['dropped_columns']])
        self.assertEqual(markers, ['m1', 'm2', 'm3'])

    def test_clean_grid_validationes(self):
        from Genoly.quantitative.preprocess import clean_grid
        grid = [['id', 'feno', 'm1'], ['a', '1', '0'], ['b', '2', '1']]
        with self.assertRaises(ValueError):
            clean_grid(grid, phenotype_col=0, min_individuals=5)
        with self.assertRaises(ValueError):
            clean_grid(grid, phenotype_col=99)
        with self.assertRaises(ValueError):
            clean_grid(grid, phenotype_col=1, impute_method='xxx')


class TestAlignment(unittest.TestCase):
    """Alineador Smith-Waterman (motor nativo parasail si está disponible)."""

    def setUp(self):
        from Genoly.alignment.alignment import GPUSequenceAligner
        self.a = GPUSequenceAligner(device='cpu')

    def test_secuencias_identicas(self):
        r = self.a.align_pair('ACGTACGT', 'ACGTACGT')
        self.assertEqual(r.score, 16.0)
        self.assertEqual(r.identity_percent, 100.0)
        self.assertEqual(r.mismatches, 0)
        self.assertEqual(r.gaps, 0)

    def test_detecta_variante(self):
        r = self.a.align_pair('ACGTACGT', 'ACGTTCGT')
        self.assertEqual(r.mismatches, 1)
        self.assertGreater(r.identity_percent, 80.0)
        variants = self.a.find_variants(r.aligned_query, r.aligned_target)
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]['type'], 'SNV')

    def test_score_paridad_con_python(self):
        # el motor nativo (parasail) y el escalar Python deben coincidir
        if self.a.engine != 'parasail':
            self.skipTest('parasail no disponible')
        import random
        rng = random.Random(42)
        bases = 'ACGT'
        n = 30
        q = ''.join(rng.choice(bases) for _ in range(n))
        t = ''.join(rng.choice(bases) for _ in range(n))
        enc_q, enc_t = self.a.encode_sequence(q), self.a.encode_sequence(t)
        py_score, _, _ = self.a._smith_waterman(enc_q, enc_t)
        nat = self.a._align_native(q, t)
        self.assertAlmostEqual(py_score, nat[0], delta=1e-6)

    def test_secuencia_vacia(self):
        r = self.a.align_pair('', 'ACGT')
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.alignment_length, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

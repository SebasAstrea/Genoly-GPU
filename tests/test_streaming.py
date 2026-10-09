"""
Tests del pipeline de streaming (RAM/VRAM acotadas).

Ejecutar desde la raiz del proyecto:
    python -m pytest tests -v
o directamente:
    python tests/test_streaming.py
"""

import os
import random
import shutil
import unittest

import torch

from Genoly.core.vram import (
    VRAMManager,
    estimate_encode_bytes_per_base,
    estimate_kmer_bytes_per_base,
)
from Genoly.encoding.encoder import SequenceEncoder
from Genoly.io.fasta import (
    FastaReader,
    FastaRecord,
    iter_fasta_batches,
    write_fasta,
)
from Genoly.kmer import kmers as kmers_module
from Genoly.kmer.kmers import KmerCounter

B4 = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def random_seq(rng, n):
    return ''.join(rng.choice('ACGT') for _ in range(n))


def kmer_codes(seq, k):
    out = set()
    for i in range(len(seq) - k + 1):
        code = 0
        for ch in seq[i:i + k]:
            code = code * 4 + B4[ch]
        out.add(code)
    return out


class _TmpDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = os.path.join(os.path.dirname(__file__), '_tmp_stream')
        os.makedirs(self.tmpdir, exist_ok=True)

    def tearDown(self):
        for name in os.listdir(self.tmpdir):
            full = os.path.join(self.tmpdir, name)
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                os.remove(full)
        os.rmdir(self.tmpdir)

    def path(self, name):
        return os.path.join(self.tmpdir, name)


class TestBlockReader(_TmpDirTestCase):
    def test_multiregistro_crlf_y_sin_salto_final(self):
        path = self.path('mix.fasta')
        with open(path, 'w', newline='') as fh:
            fh.write('>s1 desc uno\r\nACGTACGTAA\r\n\r\nTTTTGGGGCC\r\n')
            fh.write('>s2\r\n' + 'ACGT' * 30 + '\r\n')
            fh.write('>s3\r\nACGTACGT')  # sin salto final

        loaded = FastaReader(path).read_all()
        self.assertEqual([r.id for r in loaded], ['s1', 's2', 's3'])
        self.assertEqual(loaded[0].sequence, 'ACGTACGTAA' + 'TTTTGGGGCC')
        self.assertEqual(loaded[0].description, 'desc uno')
        self.assertEqual(loaded[1].sequence, 'ACGT' * 30)
        self.assertEqual(loaded[2].sequence, 'ACGTACGT')

    def test_bloques_pequenos_reconstruyen_lineas(self):
        # block_size de 7 bytes fuerza bordes de bloque dentro de líneas
        rng = random.Random(3)
        seq = random_seq(rng, 5000)
        path = self.path('blocks.fasta')
        write_fasta(path, [FastaRecord(id='b1', sequence=seq)], line_width=80)

        loaded = FastaReader(path, block_size=7, max_line=13).read_all()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].sequence, seq)

    def test_sin_registros(self):
        path = self.path('vacio.fasta')
        with open(path, 'w') as fh:
            fh.write('ACGTACGT\nACGT\n')  # sin cabeceras
        self.assertEqual(FastaReader(path).read_all(), [])
        stats = FastaReader(path).scan_stats()
        self.assertEqual(stats.records, 0)

    def test_scan_stats(self):
        path = self.path('stats.fasta')
        write_fasta(path, [
            FastaRecord(id='s1', sequence='ACGT' * 25, description='d'),
            FastaRecord(id='s2', sequence='TTTT' * 10),
        ], line_width=30)
        stats = FastaReader(path).scan_stats()
        self.assertEqual(stats.records, 2)
        self.assertEqual(stats.total_bases, 100 + 40)
        self.assertEqual(stats.first_id, 's1')
        self.assertEqual(stats.first_description, 'd')
        self.assertEqual(stats.first_length, 100)

    def test_iter_batches(self):
        path = self.path('batches.fasta')
        write_fasta(path, [FastaRecord(id=f's{i}', sequence='ACGT')
                           for i in range(7)])
        sizes = [len(b) for b in FastaReader(path).iter_batches(3)]
        self.assertEqual(sizes, [3, 3, 1])

        batches = list(iter_fasta_batches(path, batch_size=2))
        self.assertEqual([len(b) for b in batches], [2, 2, 2, 1])
        with self.assertRaises(ValueError):
            list(FastaReader(path).iter_batches(0))

    def test_iter_windows_tiling(self):
        # Los inicios de k-mer cubiertos por las ventanas son exactamente
        # los de la secuencia completa (solape k-1, sin huecos ni dobles)
        rng = random.Random(5)
        seq = random_seq(rng, 800)
        path = self.path('win.fasta')
        write_fasta(path, [FastaRecord(id='w', sequence=seq)], line_width=57)

        for k, window in ((5, 40), (8, 100), (3, 7)):
            wins = list(FastaReader(path).iter_windows(window, overlap=k - 1))
            covered = set()
            for w in wins:
                covered |= kmer_codes(w, k)
            self.assertEqual(covered, kmer_codes(seq, k), (k, window))

    def test_iter_windows_no_cruza_registros(self):
        path = self.path('multi.fasta')
        write_fasta(path, [
            FastaRecord(id='a', sequence='A' * 100),
            FastaRecord(id='c', sequence='C' * 100),
        ], line_width=10)
        for w in FastaReader(path).iter_windows(30, overlap=5):
            self.assertTrue(set(w) <= {'A'} or set(w) <= {'C'})

    def test_iter_windows_codes_equivalente_a_iter_windows(self):
        # decodificar los códigos debe reproducir las ventanas de texto
        # (incluye minúsculas, CRLF, sin salto final y Ns)
        rng = random.Random(11)
        seqs = [random_seq(rng, rng.randint(30, 300)) for _ in range(6)]
        seqs = ['N' + s[:len(s) // 3].lower() + s[len(s) // 3:]
                for s in seqs]
        path = self.path('codes.fasta')
        with open(path, 'w', newline='\r\n') as fh:
            for i, s in enumerate(seqs):
                fh.write(f'>s{i} desc\n')
                for j in range(0, len(s), 57):
                    fh.write(s[j:j + 57] + '\n')
            fh.write('>cola\nACGT')  # sin salto final

        inv = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 255: 'N'}
        for k, window in ((5, 40), (8, 100)):
            texto = list(FastaReader(path).iter_windows(window, overlap=k - 1))
            codigos = list(FastaReader(path).iter_windows_codes(
                window, overlap=k - 1))
            self.assertEqual(len(texto), len(codigos), (k, window))
            for t, c in zip(texto, codigos):
                self.assertEqual(c.size, len(t))
                self.assertEqual(
                    ''.join(inv[v] for v in c.tolist()), t.upper())

    def test_iter_windows_codes_n_no_se_salta(self):
        # N permanece en el stream como dígito inválido: las ventanas no
        # se compactan (un k-mer que cruza una N debe seguir inválido)
        path = self.path('ns.fasta')
        with open(path, 'w') as fh:
            fh.write('>n\nACGTNACGTNACGT\n')
        wins = list(FastaReader(path).iter_windows_codes(6, overlap=0))
        self.assertEqual([w.size for w in wins], [6, 6, 2])
        # ACGTNA: la N ocupa su posición (255) y no se salta
        self.assertEqual(wins[0].tolist(), [0, 1, 2, 3, 255, 0])

    def test_iter_windows_codes_validaciones(self):
        path = self.path('v.fasta')
        write_fasta(path, [FastaRecord(id='a', sequence='ACGT' * 10)])
        with self.assertRaises(ValueError):
            list(FastaReader(path).iter_windows_codes(0))
        with self.assertRaises(ValueError):
            list(FastaReader(path).iter_windows_codes(10, overlap=10))

    def test_scan_composition(self):
        path = self.path('comp.fasta')
        write_fasta(path, [
            FastaRecord(id='s1', sequence='ACGT' * 10, description='d'),
            FastaRecord(id='s2', sequence='TTTT' * 5 + 'NNNN' * 2),
        ], line_width=30)
        comp = FastaReader(path).scan_composition()
        self.assertEqual(comp.records, 2)
        self.assertEqual(comp.total_bases, 40 + 20 + 8)
        self.assertEqual(comp.composition(),
                         {'A': 10, 'C': 10, 'G': 10, 'T': 10 + 20, 'N': 8})
        self.assertEqual(comp.gc_bases, 20)
        # sin cabeceras: 0 registros, bases contadas igualmente
        empty = self.path('vacio_comp.fasta')
        with open(empty, 'w') as fh:
            fh.write('ACGTACGT\n')
        st = FastaReader(empty).scan_composition()
        self.assertEqual(st.records, 0)
        self.assertEqual(st.total_bases, 8)
        # N minúscula y caracteres IUPAC también se agregan como N
        iupac = self.path('iupac.fasta')
        with open(iupac, 'w') as fh:
            fh.write('>x\nACGTNRYKMWS\n')
        c = FastaReader(iupac).scan_composition()
        self.assertEqual(c.composition(),
                         {'A': 1, 'C': 1, 'G': 1, 'T': 1, 'N': 7})

    def test_scan_stats_casos_limite(self):
        casos = {
            'multi': '>s1 d1\n' + 'ACGT' * 30 + '\n>s2\n' + 'TTTT' * 40,
            'sin_salto': '>a\nACGTACGTACGT',
            'crlf': '>x desc\r\nACGT\r\nTTTT\r\n',
            'borde': '>h1\n' + 'A' * 63 + '>h2\n' + 'C' * 63,
            'vacio': '',
            'solo_headers': '>a\n>b\n>c\n',
        }
        for name, content in casos.items():
            path = self.path(f'{name}.fasta')
            with open(path, 'w', newline='') as fh:
                fh.write(content)
            for bs in (64, 7):
                st = FastaReader(path, block_size=bs,
                                 max_line=11).scan_stats()
                self.assertIsInstance(st.records, int, (name, bs))
                self.assertGreaterEqual(st.total_bases, 0, (name, bs))
        # caso detallado multi-registro
        path = self.path('multi2.fasta')
        with open(path, 'w', newline='') as fh:
            fh.write(casos['multi'])
        for bs in (64, 7):
            st = FastaReader(path, block_size=bs, max_line=11).scan_stats()
            self.assertEqual(st.records, 2, bs)
            self.assertEqual(st.total_bases, 120 + 160, bs)
            self.assertEqual(st.first_id, 's1', bs)
            self.assertEqual(st.first_length, 120, bs)


class _FixedVRAM(VRAMManager):
    """VRAMManager con memoria libre inyectada para tests deterministas."""

    def __init__(self, free_bytes, **kwargs):
        super().__init__('cpu', **kwargs)
        self._free = free_bytes

    def free_bytes(self):
        return self._free


class TestVRAMPlanner(unittest.TestCase):
    def test_grupos_exactos(self):
        # free=1e6, safety=0.5 -> presupuesto 500k; bytes/base=1
        vram = _FixedVRAM(1_000_000, safety_fraction=0.5, min_budget_bytes=1)
        # 50 * 10_000 = 500_000 <= presupuesto: un solo grupo exacto
        groups = list(vram.plan_micro_batches([10_000] * 50, bytes_per_base=1))
        self.assertEqual([len(g) for g in groups], [50])
        # la 51a daría 510_000 > 500_000: cierra en 50 y abre otro grupo
        groups = list(vram.plan_micro_batches([10_000] * 51, bytes_per_base=1))
        self.assertEqual([len(g) for g in groups], [50, 1])
        for group in groups:
            used = len(group) * 10_000
            self.assertLessEqual(used, 500_000)

    def test_presupuesto_con_safety(self):
        vram = _FixedVRAM(1_000_000, safety_fraction=0.25, min_budget_bytes=1)
        self.assertEqual(vram.budget_bytes(), 250_000)

    def test_item_unico_oversized_viaja_solo(self):
        vram = _FixedVRAM(1_000_000, safety_fraction=0.5, min_budget_bytes=1)
        lengths = [900_000, 100, 100]  # el primero excede 500k por sí solo
        groups = list(vram.plan_micro_batches(lengths, bytes_per_base=1))
        self.assertEqual(groups, [[0], [1, 2]])

    def test_max_items(self):
        vram = _FixedVRAM(10 ** 12, max_items=3)
        groups = list(vram.plan_micro_batches([1] * 10, bytes_per_base=1))
        self.assertEqual([len(g) for g in groups], [3, 3, 3, 1])

    def test_padding_cuenta_con_maximo(self):
        # n * max(L) * b <= presupuesto: una secuencia larga encarece al grupo
        vram = _FixedVRAM(1_000_000, safety_fraction=0.5, min_budget_bytes=1)
        lengths = [100_000, 100_000, 400_000, 1]
        groups = list(vram.plan_micro_batches(lengths, bytes_per_base=1))
        for group in groups:
            used = len(group) * max(lengths[i] for i in group)
            self.assertLessEqual(used, 500_000)
        # 100k+100k+400k = 600k > 500k -> el tercero abre grupo nuevo
        self.assertEqual([len(g) for g in groups], [2, 1, 1])

    def test_validaciones(self):
        with self.assertRaises(ValueError):
            VRAMManager('cpu', safety_fraction=0)
        with self.assertRaises(ValueError):
            list(VRAMManager('cpu').plan_micro_batches([10], bytes_per_base=0))

    def test_suggest_window_bases(self):
        vram = _FixedVRAM(10 ** 12)
        # target 128MB / 89 B -> ~1.5M -> potencia de 2 por debajo
        window = vram.suggest_window_bases(estimate_kmer_bytes_per_base(31))
        self.assertGreaterEqual(window, 65_536)
        self.assertLessEqual(window, 8 * 1024 * 1024)
        self.assertEqual(window & (window - 1), 0)  # potencia de 2

    def test_estimaciones_coste(self):
        self.assertEqual(estimate_kmer_bytes_per_base(31, canonical=True),
                         8 * 11 + 1)
        self.assertEqual(estimate_kmer_bytes_per_base(31, canonical=False),
                         8 * 7 + 1)
        self.assertEqual(estimate_encode_bytes_per_base(one_hot=False), 24)
        self.assertEqual(estimate_encode_bytes_per_base(one_hot=True), 68)


class TestEncodeStream(_TmpDirTestCase):
    def test_equivalente_a_encode(self):
        enc = SequenceEncoder('cpu')
        rng = random.Random(9)
        seqs = [random_seq(rng, rng.randint(5, 200)) for _ in range(30)]
        batches = [seqs[:7], seqs[7:20], seqs[20:]]

        ref, ref_lens = enc.encode(seqs)
        ref_lens = ref_lens.cpu().tolist()

        pos = 0
        for chunk in enc.encode_stream(batches):
            for r in range(chunk.tensor.shape[0]):
                length = chunk.lengths[r]
                gi = pos + r
                self.assertEqual(chunk.lengths[r], ref_lens[gi])
                self.assertTrue(torch.equal(
                    chunk.tensor[r, :length], ref[gi, :length]))
            pos += chunk.tensor.shape[0]
        self.assertEqual(pos, len(seqs))

    def test_one_hot_equivalente(self):
        enc = SequenceEncoder('cpu')
        seqs = ['ACGTACGT', 'TTTTGGGGCC', 'AAAANNNN']
        ref = enc.encode_one_hot(seqs)
        off = 0
        for chunk in enc.encode_stream([seqs[:2], seqs[2:]], one_hot=True):
            for r in range(chunk.tensor.shape[0]):
                length = chunk.lengths[r]
                self.assertTrue(torch.allclose(
                    chunk.tensor[r, :length], ref[off + r, :length]))
            off += chunk.tensor.shape[0]

    def test_lote_vacio(self):
        enc = SequenceEncoder('cpu')
        chunks = list(enc.encode_stream([[]]))
        self.assertEqual(chunks, [])


class TestCountStream(_TmpDirTestCase):
    def setUp(self):
        super().setUp()
        self.kc = KmerCounter('cpu')
        rng = random.Random(21)
        self.seqs = [random_seq(rng, rng.randint(50, 600)) for _ in range(12)]
        self.k = 6

    def assertSameCounts(self, a, b):
        va, ca = a
        vb, cb = b
        self.assertEqual(set(zip(va.tolist(), ca.tolist())),
                         set(zip(vb.tolist(), cb.tolist())))
        self.assertEqual(int(ca.sum()), int(cb.sum()))

    def test_count_stream_igual_a_count(self):
        ref = self.kc.count(self.seqs, k=self.k, canonical=True)
        stream = self.kc.count_stream(iter(self.seqs), k=self.k,
                                      canonical=True, ram_batch_size=3)
        self.assertSameCounts(ref, stream)

    def test_count_stream_no_canonico(self):
        ref = self.kc.count(self.seqs, k=self.k, canonical=False)
        stream = self.kc.count_stream(iter(self.seqs), k=self.k,
                                      canonical=False, ram_batch_size=2)
        self.assertSameCounts(ref, stream)

    def test_count_stream_ventanas_iguales_a_completo(self):
        ref = self.kc.count(self.seqs, k=self.k, canonical=True)
        stream = self.kc.count_stream(iter(self.seqs), k=self.k,
                                      canonical=True, window_size=97,
                                      ram_batch_size=2)
        self.assertSameCounts(ref, stream)

    def test_count_records_progreso(self):
        path = self.path('recs.fasta')
        write_fasta(path, [FastaRecord(id=f's{i}', sequence=s)
                           for i, s in enumerate(self.seqs)])
        events = []
        got = self.kc.count_records(
            FastaReader(path).records(), k=self.k,
            on_progress=lambda info: events.append(dict(info)))
        self.assertSameCounts(
            self.kc.count(self.seqs, k=self.k, canonical=True), got)
        self.assertTrue(events)
        self.assertEqual(events[-1]['records'], len(self.seqs))
        self.assertEqual(events[-1]['bases'],
                         sum(len(s) for s in self.seqs))
        self.assertGreater(events[-1]['micro_batches'], 0)

    def test_count_fasta_igual_a_count(self):
        path = self.path('file.fasta')
        write_fasta(path, [FastaRecord(id=f's{i}', sequence=s)
                           for i, s in enumerate(self.seqs)], line_width=60)
        ref = self.kc.count(self.seqs, k=self.k, canonical=True)
        got = self.kc.count_fasta(path, k=self.k, canonical=True)
        self.assertSameCounts(ref, got)

    def test_count_fasta_ventana_fija(self):
        path = self.path('file2.fasta')
        write_fasta(path, [FastaRecord(id=f's{i}', sequence=s)
                           for i, s in enumerate(self.seqs)], line_width=60)
        ref = self.kc.count(self.seqs, k=self.k, canonical=True)
        got = self.kc.count_fasta(path, k=self.k, canonical=True,
                                  window_size=150, ram_batch_size=4)
        self.assertSameCounts(ref, got)

    def test_compaction_intermedia_no_altera_resultado(self):
        path = self.path('merge.fasta')
        write_fasta(path, [FastaRecord(id=f's{i}', sequence=s)
                           for i, s in enumerate(self.seqs)])
        original = kmers_module.MERGE_THRESHOLD_ELEMENTS
        kmers_module.MERGE_THRESHOLD_ELEMENTS = 10  # fuerza merges constantes
        try:
            got = self.kc.count_fasta(path, k=self.k, canonical=True)
        finally:
            kmers_module.MERGE_THRESHOLD_ELEMENTS = original
        self.assertSameCounts(
            self.kc.count(self.seqs, k=self.k, canonical=True), got)

    def test_k31_stream_igual_a_count(self):
        g = torch.Generator().manual_seed(1234)
        seq = ''.join('ACGT'[i] for i in torch.randint(
            0, 4, (2500,), generator=g).tolist())
        ref = self.kc.count([seq], k=31, canonical=False)
        stream = self.kc.count_stream(iter([seq]), k=31, canonical=False,
                                      window_size=0, ram_batch_size=1)
        self.assertSameCounts(ref, stream)

    def test_secuencias_cortas_ignoradas(self):
        got = self.kc.count_stream(iter(['AC', 'ACGTA']), k=6)
        self.assertEqual(int(got[1].numel()), 0)

    def test_validaciones(self):
        with self.assertRaises(ValueError):
            self.kc.count_stream(iter(['ACGT']), k=32)
        with self.assertRaises(ValueError):
            self.kc.count_stream(iter(['ACGT']), k=3, ram_batch_size=0)

    def test_rutas_int32_e_int64_contra_referencia(self):
        # k <= 15 (1 palabra int32), 16 <= k <= 30 (2 palabras int32) y
        # k = 31 (int64): todos deben coincidir con la referencia
        # brute-force, en modo canónico, con Ns en la secuencia
        rng = random.Random(77)
        seq = ''.join(rng.choice('ACGTN') for _ in range(1500))
        for k in (1, 2, 14, 15, 16, 17, 21, 29, 30, 31):
            esperado = {}
            for i in range(len(seq) - k + 1):
                w = seq[i:i + k]
                if any(ch not in B4 for ch in w):
                    continue
                code = 0
                for ch in w:
                    code = code * 4 + B4[ch]
                rc = 0
                for ch in reversed(w):
                    rc = rc * 4 + (3 - B4[ch])
                code = min(code, rc)
                esperado[code] = esperado.get(code, 0) + 1

            values, counts = self.kc.count([seq], k=k, canonical=True)
            got = dict(zip(values.tolist(), counts.tolist()))
            self.assertEqual(got, esperado, f'k={k}')


class TestAggregated(_TmpDirTestCase):
    """count_fasta_aggregated: acumulador particionado con derrame."""

    def setUp(self):
        super().setUp()
        self.kc = KmerCounter('cpu')
        rng = random.Random(33)
        self.seqs = [random_seq(rng, rng.randint(50, 500)) for _ in range(14)]
        self.k = 6
        self.fasta = self.path('agg.fasta')
        write_fasta(self.fasta, [FastaRecord(id=f's{i}', sequence=s)
                                 for i, s in enumerate(self.seqs)],
                    line_width=60)

    def _reference(self, min_abundance=1):
        values, counts = self.kc.count(self.seqs, k=self.k, canonical=True,
                                       min_abundance=min_abundance)
        return values, counts

    def test_equivalente_a_count_con_spill_forzado(self):
        # particiones y spill diminutos: fuerza derrames y merges por disco
        agg = self.kc.count_fasta_aggregated(
            self.fasta, k=self.k, canonical=True, top=10,
            n_partitions=8, spill_rows=5)
        ref_v, ref_c = self._reference()

        self.assertEqual(agg['total_unique'], len(ref_v))
        self.assertEqual(agg['total_kmers'], int(ref_c.sum()))
        self.assertEqual(agg['spectrum'],
                         KmerCounter.spectrum_from_counts(ref_c))

        esperado = sorted(zip(ref_c.tolist(), ref_v.tolist()),
                          key=lambda x: (-x[0], x[1]))
        top_ref = [{'kmer': self.kc.decode_kmer(v, self.k), 'count': c}
                   for c, v in esperado[:10]]
        self.assertEqual(agg['top_kmers'], top_ref)

    def test_min_abundance(self):
        agg = self.kc.count_fasta_aggregated(
            self.fasta, k=self.k, canonical=True, min_abundance=3, top=5,
            n_partitions=8, spill_rows=5)
        ref_v, ref_c = self._reference(min_abundance=3)
        self.assertEqual(agg['total_unique'], len(ref_v))
        self.assertEqual(agg['total_kmers'], int(ref_c.sum()))

    def test_sin_ventanas_registros_completos(self):
        agg = self.kc.count_fasta_aggregated(
            self.fasta, k=self.k, canonical=True, top=3, window_size=0,
            n_partitions=8, spill_rows=5)
        ref_v, ref_c = self._reference()
        self.assertEqual(agg['total_unique'], len(ref_v))
        self.assertEqual(agg['total_kmers'], int(ref_c.sum()))

    def test_ventana_igual_a_k(self):
        agg = self.kc.count_fasta_aggregated(
            self.fasta, k=self.k, canonical=True, top=3, window_size=self.k,
            n_partitions=4, spill_rows=5)
        ref_v, ref_c = self._reference()
        self.assertEqual(agg['total_unique'], len(ref_v))
        self.assertEqual(agg['total_kmers'], int(ref_c.sum()))

    def test_ventana_menor_que_k_falla(self):
        with self.assertRaises(ValueError):
            self.kc.count_fasta_aggregated(
                self.fasta, k=10, window_size=5)

    def test_progreso_y_limpieza_de_derrames(self):
        eventos = []
        spill = self.path('spill_dir')
        os.makedirs(spill, exist_ok=True)
        agg = self.kc.count_fasta_aggregated(
            self.fasta, k=self.k, canonical=True, top=2,
            n_partitions=4, spill_rows=10, spill_dir=spill,
            on_progress=lambda info: eventos.append(dict(info)))
        self.assertTrue(eventos)
        self.assertGreater(eventos[-1]['window_size'], 0)  # ventana auto
        self.assertEqual(agg['total_kmers'],
                         int(self._reference()[1].sum()))
        leftovers = list(os.listdir(spill))
        self.assertEqual(leftovers, [], leftovers)

    def test_temporales_eliminados_al_fallar(self):
        spill = self.path('spill_err')
        os.makedirs(spill, exist_ok=True)
        with self.assertRaises(ValueError):
            self.kc.count_fasta_aggregated(
                self.fasta, k=99, n_partitions=4, spill_rows=5,
                spill_dir=spill)
        self.assertEqual(list(os.listdir(spill)), [])

    def test_count_fastas_agregadas_equivale_a_concatenado(self):
        # dos archivos FASTA separados == el mismo contenido concatenado:
        # el resultado combinado es exacto (los k-mers compartidos no se
        # duplican), igual que el pipeline de un solo archivo
        half = len(self.seqs) // 2
        f1 = self.path('agg1.fasta')
        f2 = self.path('agg2.fasta')
        fc = self.path('agg_c.fasta')
        write_fasta(f1, [FastaRecord(id=f's{i}', sequence=s)
                         for i, s in enumerate(self.seqs[:half])], line_width=60)
        write_fasta(f2, [FastaRecord(id=f's{i}', sequence=s)
                         for i, s in enumerate(self.seqs[half:])], line_width=60)
        with open(fc, 'w') as fh:
            fh.write(open(f1).read())
            fh.write(open(f2).read())

        kw = dict(k=self.k, canonical=True, top=10,
                  n_partitions=4, spill_rows=5)
        combined = self.kc.count_fastas_aggregated([f1, f2], **kw)
        single = self.kc.count_fasta_aggregated(fc, **kw)
        self.assertEqual(combined, single)
        self.assertEqual(combined['total_kmers'],
                         int(self._reference()[1].sum()))

    def test_count_fastas_agregadas_progreso_por_archivo(self):
        eventos = []
        self.kc.count_fastas_aggregated(
            [self.fasta, self.fasta], k=self.k, canonical=True, top=2,
            n_partitions=4, spill_rows=10,
            on_progress=lambda info: eventos.append(dict(info)))
        # el progreso debe indicar el archivo (file/file_index/files)
        self.assertTrue(any('file' in e for e in eventos))
        self.assertEqual(eventos[-1]['files'], 2)

    def test_validaciones(self):
        with self.assertRaises(ValueError):
            self.kc.count_fasta_aggregated(self.fasta, k=5, top=-1)
        with self.assertRaises(ValueError):
            self.kc.count_fasta_aggregated(self.fasta, k=5, n_partitions=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)

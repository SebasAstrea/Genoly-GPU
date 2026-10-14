"""Tests de la canalización estilo GATK: lecturas, PairHMM, ensamblaje,
regiones activas, genotipado, HaplotypeCaller end-to-end, BQSR,
MarkDuplicates, VCF y superficie API."""

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from Genoly.gatk import (  # noqa: E402
    BaseRecalibrator,
    FilterStats,
    GATKRead,
    GATKVariant,
    HaplotypeCaller,
    LocalAssembler,
    RecalTable,
    active_regions,
    allele_log_likelihoods,
    apply_duplicate_flags,
    apply_recalibration,
    duplicate_keys,
    filter_reads,
    genotype_likelihoods,
    haplotype_sites,
    has_native,
    mark_duplicates,
    pairhmm_log10,
    read_score,
    write_gatk_vcf,
    write_recal_sam,
)

# Referencia pseudoaleatoria (determinista): una tanda repetitiva como
# "ACGT"*n rompe el grafo de k-mers (los k-mers rotan) y a difflib.
_rng = random.Random(7)
FASTA = "".join(_rng.choice("ACGT") for _ in range(120))
VAR_POS = 52         # 0-based
REF_BASE = FASTA[VAR_POS]
ALT_BASE = "ACGT"[("ACGT".index(REF_BASE) + 1) % 4]


def ref_seq():
    return FASTA


def alt_seq():
    return FASTA[:VAR_POS] + ALT_BASE + FASTA[VAR_POS + 1:]


def del_seq():
    return FASTA[:VAR_POS] + FASTA[VAR_POS + 3:]


def make_reads(hap=None, n=20, start=0, qual="I", length=90, tag="r"):
    """lecturas sintéticas forward de longitud `length` con `qual` uniforme"""
    s_seq = ref_seq() if hap is None else hap
    reads = []
    for i in range(n):
        s = start + (i % (VAR_POS + 1))
        seq = s_seq[s:s + length]
        reads.append(GATKRead(
            qname=f"{tag}{i}", sequence=seq, qualities=qual * len(seq),
            flag=0, rname="chr1", start=s, mapq=60,
            cigar=f"{len(seq)}M"))
    return reads


class TestReads(unittest.TestCase):
    def test_filter_stats(self):
        reads = [
            GATKRead(qname="ok", sequence="ACGT", qualities="IIII",
                     flag=0, rname="chr1", start=0, mapq=60),
            GATKRead(qname="unmapped", sequence="ACGT", qualities="IIII",
                     flag=4, rname="*", start=-1, mapq=0),
            GATKRead(qname="dup", sequence="ACGT", qualities="IIII",
                     flag=512, rname="chr1", start=0, mapq=60),
            GATKRead(qname="secondary", sequence="ACGT", qualities="IIII",
                     flag=256, rname="chr1", start=0, mapq=60),
            GATKRead(qname="lowq", sequence="ACGT", qualities="IIII",
                     flag=0, rname="chr1", start=0, mapq=5),
        ]
        stats = filter_reads(reads, min_mapq=20)
        self.assertIsInstance(stats, FilterStats)
        self.assertEqual(stats.total, 5)
        self.assertEqual([r.qname for r in stats.reads], ["ok"])
        self.assertEqual(stats.removed["unmapped"], 1)
        self.assertEqual(stats.removed["duplicate"], 1)
        self.assertEqual(stats.removed["secondary"], 1)
        self.assertEqual(stats.removed["low_mapq"], 1)

    def test_cigar_geometry(self):
        r = GATKRead(qname="x", sequence="ACGT", qualities="IIII",
                     flag=0, rname="chr1", start=10, mapq=60,
                     cigar="2S6M1H")
        self.assertEqual(r.ref_span(), 6)
        self.assertEqual(r.unclipped_start(), 8)
        self.assertEqual(r.unclipped_end(), 17)
        self.assertEqual(r.five_prime(), 8)
        self.assertTrue(r.overlaps(12, 20))
        self.assertFalse(r.overlaps(14, 30))
        self.assertEqual(r.phred_scores(), [40, 40, 40, 40])
        self.assertFalse(r.is_duplicate)


class TestPairHMM(unittest.TestCase):
    def _out(self, backend):
        a = ref_seq()[:50]
        b = "A" * 50
        return pairhmm_log10([(a, "I" * 50), (b, "I" * 50)], [a],
                             backend=backend)

    def test_sanity(self):
        a = ref_seq()[:50]
        one_mm = a[:25] + ("T" if a[25] != "T" else "A") + a[26:]
        rnd = "T" * 50
        out = self._out("numpy")
        self.assertEqual(out.shape, (2, 1))
        # lectura idéntica al haplotipo: likelihood altísima
        self.assertGreater(out[0][0], -0.1)
        # un solo mismatch (Q40) empeora la likelihood
        mid = pairhmm_log10([(one_mm, "I" * 50)], [a], backend="numpy")
        self.assertLess(mid[0][0], out[0][0])
        # lectura sin ninguna coincidencia: la peor de las tres
        out2 = pairhmm_log10([(rnd, "I" * 50)], [a], backend="numpy")
        self.assertLess(out2[0][0], mid[0][0])

    def test_numpy_torch_parity(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch no disponible")
        np_out = self._out("numpy")
        torch_out = self._out("torch")
        for v_np, v_t in zip(np_out.ravel(), torch_out.ravel()):
            self.assertAlmostEqual(float(v_np), float(v_t), places=9)

    def test_native_parity(self):
        if not has_native():
            self.skipTest("extensión nativa no compilada")
        np_out = self._out("numpy")
        nat_out = self._out("native")
        for v_np, v_n in zip(np_out.ravel(), nat_out.ravel()):
            self.assertAlmostEqual(float(v_np), float(v_n), places=6)


class TestAssembly(unittest.TestCase):
    def test_choose_k(self):
        assembler = LocalAssembler()
        long_reads = [r.sequence for r in make_reads(n=5, length=80)]
        self.assertEqual(assembler.choose_k(ref_seq(), long_reads), 25)
        short = ["ACGTACGTAC"]
        self.assertEqual(assembler.choose_k("ACGT" * 25, short), 9)

    def test_bubble_ref_and_alt(self):
        assembler = LocalAssembler(k=25, max_haplotypes=16)
        reads = make_reads(n=10) + make_reads(hap=alt_seq(), n=10, tag="a")
        haps = assembler.assemble(ref_seq(), [r.sequence for r in reads])
        self.assertEqual(len(haps), 2)
        self.assertEqual(haps[0], ref_seq())
        self.assertIn(alt_seq(), haps)

    def test_deletion_path(self):
        assembler = LocalAssembler(k=25, max_haplotypes=16)
        reads = ([GATKRead(qname=f"r{i}", sequence=ref_seq(),
                           qualities="I" * 120, flag=0, rname="chr1",
                           start=0, mapq=60, cigar="120M")
                  for i in range(5)] +
                 [GATKRead(qname=f"d{i}", sequence=del_seq(),
                           qualities="I" * 117, flag=0, rname="chr1",
                           start=0, mapq=60, cigar="117M")
                  for i in range(5)])
        haps = assembler.assemble(ref_seq(), [r.sequence for r in reads])
        self.assertIn(ref_seq(), haps)
        self.assertIn(del_seq(), haps)


class TestActiveRegions(unittest.TestCase):
    def test_detection_and_merge(self):
        reads = make_reads(n=12) + make_reads(hap=alt_seq(), n=8, tag="a")
        regions = active_regions(reads, ref_seq(), (0, 120),
                                 window=50, extension=50)
        self.assertTrue(regions)
        start, end = regions[0]
        self.assertLessEqual(start, VAR_POS)
        self.assertGreater(end, VAR_POS)

    def test_no_signal_no_regions(self):
        reads = make_reads(n=5, start=0)
        regions = active_regions(reads, ref_seq(), (0, 120), window=50,
                                 extension=10, min_af=0.3, min_alt_count=4)
        self.assertEqual(regions, [])


class TestLikelihoods(unittest.TestCase):
    def test_haplotype_sites_snv(self):
        sites, per_hap = haplotype_sites(ref_seq(),
                                         [ref_seq(), alt_seq()])
        self.assertEqual(len(sites), 1)
        s = sites[0]
        self.assertEqual(s.key, ("snv", VAR_POS))
        self.assertEqual(s.position, VAR_POS + 1)
        self.assertEqual(s.ref, REF_BASE)
        self.assertEqual(s.alts, [ALT_BASE])
        self.assertEqual(s.types, ["SNV"])
        # el haplotipo REF no toca el sitio (alelo 0 implícito)
        self.assertIn(per_hap[0].get(s.key), (None, 0))
        self.assertEqual(per_hap[1].get(s.key), 1)

    def test_haplotype_sites_del(self):
        sites, per_hap = haplotype_sites(ref_seq(),
                                         [ref_seq(), del_seq()])
        s = sites[0]
        self.assertEqual(s.key, ("del", VAR_POS, VAR_POS + 3))
        self.assertEqual(s.position, VAR_POS)
        self.assertEqual(s.ref, ref_seq()[VAR_POS - 1:VAR_POS + 3])
        self.assertEqual(s.alts[0], ref_seq()[VAR_POS - 1])
        self.assertEqual(s.types, ["DEL"])
        self.assertIn(per_hap[0].get(s.key), (None, 0))
        self.assertEqual(per_hap[1].get(s.key), 1)

    def test_haplotype_sites_ins(self):
        ins_hap = ref_seq()[:VAR_POS] + "AG" + ref_seq()[VAR_POS:]
        sites, _ = haplotype_sites(ref_seq(), [ref_seq(), ins_hap])
        self.assertEqual(len(sites), 1)
        s = sites[0]
        self.assertEqual(s.types, ["INS"])
        # difflib puede anclar la inserción a izquierda o a derecha
        valid = {(ref_seq()[VAR_POS - 1], ref_seq()[VAR_POS - 1] + "AG"),
                 (ref_seq()[VAR_POS], ref_seq()[VAR_POS] + "G")}
        self.assertIn((s.ref, s.alts[0]), valid)
        self.assertIn(s.position, (VAR_POS, VAR_POS + 1))

    def test_genotype_likelihoods_diploid(self):
        import numpy as np
        ll = np.array([[0.0, -20.0], [0.0, -20.0],
                       [-20.0, 0.0], [-20.0, 0.0]])
        res = genotype_likelihoods(ll)
        self.assertEqual(res.gt, "0/1")
        self.assertEqual(res.pl[1], 0)
        self.assertEqual(res.pl[0], res.pl[2])
        self.assertGreater(res.pl[0], 300)
        self.assertEqual(res.gq, 99)
        self.assertEqual(res.ad, [2, 2])
        self.assertEqual(res.dp, 4)
        self.assertGreater(res.qual, 300)

    def test_allele_log_likelihoods(self):
        import numpy as np
        haps = [ref_seq(), alt_seq()]
        reads = make_reads(n=10) + make_reads(hap=alt_seq(), n=10, tag="a")
        read_hap = pairhmm_log10(
            [(r.sequence, r.qualities) for r in reads], haps,
            backend="numpy")
        sites, per_hap = haplotype_sites(ref_seq(), haps)
        ll = allele_log_likelihoods(read_hap, per_hap, sites[0], 2)
        self.assertEqual(ll.shape, (20, 2))
        self.assertTrue(np.isfinite(ll).all())
        # cada lectura favorece claramente su alelo
        self.assertTrue((np.argmax(ll, axis=1)
                         == np.array([0] * 10 + [1] * 10)).all())


class TestHaplotypeCaller(unittest.TestCase):
    def _caller(self):
        return HaplotypeCaller(min_mapq=20, min_depth=5, min_qual=10.0,
                               min_af=0.15, window=50, extension=50,
                               max_haplotypes=16, backend="numpy")

    def test_het_snv(self):
        reads = make_reads(n=10) + make_reads(hap=alt_seq(), n=10, tag="a")
        variants = self._caller().call_region(ref_seq(), reads)
        self.assertEqual(len(variants), 1)
        v = variants[0]
        self.assertIsInstance(v, GATKVariant)
        self.assertEqual(v.type, "SNV")
        self.assertEqual(v.position, VAR_POS + 1)
        self.assertEqual(v.ref, REF_BASE)
        self.assertEqual(v.alt, ALT_BASE)
        self.assertEqual(v.gt, "0/1")
        self.assertEqual(v.dp, 20)
        self.assertEqual(v.ad, [10, 10])
        self.assertEqual(v.filter, "PASS")
        self.assertGreater(v.qual, 30)

    def test_hom_alt(self):
        reads = make_reads(hap=alt_seq(), n=20)
        variants = self._caller().call_region(ref_seq(), reads)
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].gt, "1/1")
        self.assertEqual(variants[0].ad, [0, 20])

    def test_reference_only_no_calls(self):
        result = self._caller().call_region(ref_seq(), make_reads(n=15))
        self.assertEqual(result, [])

    def test_deletion(self):
        reads = ([GATKRead(qname=f"r{i}", sequence=ref_seq()[:90],
                           qualities="I" * 90, flag=0, rname="chr1",
                           start=0, mapq=60, cigar="90M")
                  for i in range(8)] +
                 [GATKRead(qname=f"d{i}", sequence=del_seq()[:90],
                           qualities="I" * 90, flag=0, rname="chr1",
                           start=0, mapq=60, cigar="90M")
                  for i in range(8)])
        variants = self._caller().call_region(ref_seq(), reads)
        dels = [v for v in variants if v.type == "DEL"]
        self.assertEqual(len(dels), 1)
        v = dels[0]
        self.assertEqual(v.position, VAR_POS)
        self.assertEqual(v.ref, ref_seq()[VAR_POS - 1:VAR_POS + 3])
        self.assertEqual(v.alt, ref_seq()[VAR_POS - 1])
        self.assertIn(v.gt, ("0/1", "1/1"))


class TestVcfOutput(unittest.TestCase):
    def test_write_vcf(self):
        v = GATKVariant(position=VAR_POS + 1, ref="A", alts=["G"],
                        type="SNV", qual=99.0, filter="PASS", gt="0/1",
                        dp=20, ad=[10, 10], gq=60, pl=[0, 100, 100],
                        af=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.vcf")
            n = write_gatk_vcf(path, {"chr1": [v]},
                               [{"name": "chr1", "length": 120}],
                               sample="S1")
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        self.assertEqual(n, 1)
        self.assertIn("##FORMAT=<ID=GT", text)
        self.assertIn("##FILTER=<ID=LowQual", text)
        self.assertIn("##contig=<ID=chr1,length=120>", text)
        self.assertIn("GT:AD:DP:GQ:PL", text)
        line = [row for row in text.splitlines()
                if not row.startswith("#")][0].split("\t")
        self.assertEqual(line[0], "chr1")
        self.assertEqual(line[1], str(VAR_POS + 1))
        self.assertEqual(line[3], "A")
        self.assertEqual(line[4], "G")
        self.assertEqual(line[9], "0/1:10,10:20:60:0,100,100")
        self.assertIn("\tS1\n", text)


class TestBqsr(unittest.TestCase):
    def _setup(self):
        ref = "ACGT" * 25
        reads = []
        for i in range(20):
            seq = ("T" if ref[0] != "T" else "A") + ref[1:]
            reads.append(GATKRead(qname=f"r{i}", sequence=seq,
                                  qualities="I" * len(seq), flag=0,
                                  rname="chr1", start=0, mapq=60,
                                  cigar=f"{len(seq)}M"))
        return ref, reads

    def test_estimate_and_apply(self):
        ref, reads = self._setup()
        table = BaseRecalibrator().estimate(ref, reads)
        self.assertIsInstance(table, RecalTable)
        self.assertEqual(table.n_bases, 2000)
        self.assertEqual(table.n_errors, 20)
        q_cycle1 = table.empirical_quality(40, 1, 0)
        q_cycle2 = table.empirical_quality(40, 2, 0)
        self.assertLess(q_cycle1, 10)
        self.assertGreater(q_cycle2, q_cycle1)
        recal = apply_recalibration(reads, table)
        self.assertLess(recal[0].qualities[0], reads[0].qualities[0])
        self.assertEqual(len(recal[0].qualities), len(reads[0].qualities))

    def test_to_from_dict_roundtrip(self):
        table = RecalTable()
        table.counts[(40, 1, 0)] = [10, 2]
        table.n_bases = 100
        table.n_errors = 5
        other = RecalTable.from_dict(table.to_dict())
        self.assertEqual(other.counts[(40, 1, 0)], [10, 2])
        self.assertEqual(other.n_bases, 100)
        self.assertEqual(other.n_errors, 5)


class TestMarkDuplicates(unittest.TestCase):
    def _reads(self):
        return [
            GATKRead(qname="r1", sequence="A" * 60, qualities="I" * 60,
                     flag=0, rname="chr1", start=100, mapq=60,
                     cigar="60M"),
            GATKRead(qname="r2", sequence="A" * 60, qualities="5" * 60,
                     flag=0, rname="chr1", start=100, mapq=60,
                     cigar="60M"),
            GATKRead(qname="r3", sequence="A" * 60, qualities="#" * 60,
                     flag=0, rname="chr1", start=100, mapq=60,
                     cigar="60M"),
            GATKRead(qname="r4", sequence="A" * 60, qualities="I" * 60,
                     flag=0, rname="chr1", start=300, mapq=60,
                     cigar="60M"),
        ]

    def test_keys_and_score(self):
        reads = self._reads()
        keys = {duplicate_keys(r) for r in reads}
        self.assertEqual(len(keys), 2)
        self.assertGreater(read_score(reads[0]), read_score(reads[2]))
        self.assertEqual(read_score(reads[2]), 0)
        self.assertEqual(read_score(reads[0]), 60)

    def test_mark_and_apply(self):
        reads = self._reads()
        stats = mark_duplicates(reads)
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["duplicates"], 2)
        self.assertEqual(stats["libraries"], 2)
        self.assertEqual(set(stats["qnames"]), {"r2", "r3"})
        flagged = apply_duplicate_flags(reads, set(stats["qnames"]))
        self.assertFalse(flagged[0].is_duplicate)
        self.assertTrue(flagged[1].is_duplicate)
        self.assertTrue(flagged[2].is_duplicate)
        self.assertFalse(flagged[3].is_duplicate)

    def test_write_recal_sam(self):
        reads = self._reads()
        flagged = apply_duplicate_flags(reads, {"r2", "r3"})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.sam")
            write_recal_sam(path, flagged, {"chr1": 1000}, sample="S1")
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        self.assertIn("@SQ\tSN:chr1\tLN:1000", text)
        self.assertIn("@RG\tID:1\tSM:S1", text)
        self.assertIn("\t512\t", text)


class TestCallFromSam(unittest.TestCase):
    def test_end_to_end_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            fai = os.path.join(tmp, "ref.fa")
            with open(fai, "w", encoding="utf-8") as fh:
                fh.write(">chr1\n" + ref_seq() + "\n")
            sam = os.path.join(tmp, "aln.sam")
            lines = ["@HD\tVN:1.6\tSO:coordinate",
                     "@SQ\tSN:chr1\tLN:120"]
            for i in range(20):
                hap = ref_seq() if i % 2 == 0 else alt_seq()
                start = i % (VAR_POS + 1)
                seq = hap[start:start + 90]
                lines.append(
                    f"r{i}\t0\tchr1\t{start + 1}\t60\t90M\t*\t0\t0\t"
                    f"{seq}\t{'I' * 90}")
            with open(sam, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

            seen = []
            result = HaplotypeCaller(backend="numpy").call_from_sam(
                sam, fai, region_size=50_000_000, sample_name="TEST",
                on_progress=lambda p: seen.append(p))
            self.assertEqual(result["total_variants"], 1)
            self.assertEqual(result["snvs"], 1)
            self.assertEqual(result["sample_name"], "TEST")
            self.assertTrue(seen)
            self.assertEqual(seen[0]["sequence"], "chr1")
            v = result["by_sequence"]["chr1"][0]
            self.assertEqual(v.gt, "0/1")
            self.assertEqual(v.position, VAR_POS + 1)
            self.assertEqual(result["sequences"][0]["name"], "chr1")

            out = os.path.join(tmp, "out.vcf")
            n = write_gatk_vcf(out, result["by_sequence"],
                               result["sequences"],
                               sample=result["sample_name"])
            self.assertEqual(n, 1)
            self.assertTrue(os.path.getsize(out) > 0)


class TestGatkApi(unittest.TestCase):
    """Superficie de la API: auth y validación de entradas."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from ui.backend.main import app
        cls.client = TestClient(app)
        cls._key = os.environ.get("GENOLY_API_KEY")
        os.environ["GENOLY_API_KEY"] = "secreta-prueba"

    @classmethod
    def tearDownClass(cls):
        if cls._key is None:
            os.environ.pop("GENOLY_API_KEY", None)
        else:
            os.environ["GENOLY_API_KEY"] = cls._key

    def test_requires_auth(self):
        schema = self.client.get("/openapi.json").json()
        paths = {p: ops for p, ops in schema["paths"].items()
                 if p.startswith("/api/gatk")}
        self.assertTrue(paths)
        for p, ops in paths.items():
            if "post" not in ops:
                continue
            resp = self.client.post(p, json={})
            self.assertEqual(resp.status_code, 401, p)

    def test_unknown_upload_404(self):
        headers = {"X-API-Key": "secreta-prueba"}
        resp = self.client.post(
            "/api/gatk/markdups",
            json={"sam_upload_id": "00000000000000000000000000000000"},
            headers=headers)
        self.assertEqual(resp.status_code, 404)

    def test_missing_field_422(self):
        headers = {"X-API-Key": "secreta-prueba"}
        resp = self.client.post("/api/gatk/bqsr", json={},
                                headers=headers)
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()

"""
Proceso trabajador GPU aislado.

Los trabajos de análisis pesado (conteo de k-mers sobre GPU, genomas de
varios GB) se ejecutan en un PROCESO aparte. Si el trabajo revienta —
OOM del kernel, segfault de una librería nativa, un ``os._exit`` — el
proceso del servidor sobrevive y el trabajo se marca como error en lugar
de tumbar la API.

Protocolo sobre la conexión (Pipe multiprocessing):
    ("progress", info)    progreso parcial (el padre lo fusiona con los
                          totales y lo publica como SSE, con throttling)
    ("result",   result)  resultado final (dict JSON-serializable)
    ("error",    detail)  error controlado

Si el proceso muere sin enviar mensaje, el lector del padre recibe EOF y
marca el trabajo como "error (crash)".

Este módulo no importa la app ni torch en el arranque: las dependencias
pesadas se cargan dentro de la ejecución del trabajo, para que el
proceso hijo arranque ligero.
"""

import logging
import multiprocessing as mp

logger = logging.getLogger("genoly.worker")


def _progress(conn: "mp.connection.Connection"):
    """Callback de progreso que reenvía los mensajes por la Pipe."""
    def cb(info: dict) -> None:
        conn.send(("progress", dict(info)))
    return cb


def _finalize_estimate(payload: dict, min_abundance: int) -> dict:
    """Añade la estimación de genoma al payload agregado (Lander-Waterman)."""
    from Genoly.kmer.kmers import KmerCounter
    estimate = None
    if min_abundance <= 1 and payload["total_unique"] > 0:
        size, _ = KmerCounter.estimate_from_spectrum(payload["spectrum"])
        estimate = round(size, 0) if size > 0 else None
    payload["genome_estimate"] = estimate
    return payload


def _build_payload(kc, spec: dict, values, counts) -> dict:
    """Payload de la ruta por secuencias (en memoria, tamaños pequeños)."""
    from Genoly.kmer.kmers import KmerCounter
    k = spec["k"]
    top = spec["top"]
    top_kmers = [
        {"kmer": kc.decode_kmer(int(v), k), "count": int(c)}
        for v, c in zip(values[:top].tolist(), counts[:top].tolist())
    ]
    spectrum = KmerCounter.spectrum_from_counts(counts)
    estimate = None
    if spec["min_abundance"] <= 1 and values.numel() > 0:
        size, _ = KmerCounter.estimate_from_counts(counts)
        estimate = round(size, 0) if size > 0 else None
    return {
        "k": k,
        "total_unique": len(values),
        "total_kmers": int(counts.sum().item()) if len(counts) else 0,
        "top_kmers": top_kmers,
        "spectrum": spectrum,
        "genome_estimate": estimate,
    }


def _run_kmer(spec: dict, progress) -> dict:
    """Conteo de k-mers sobre GPU (dataset, archivo o secuencias)."""
    from Genoly.kmer.kmers import KmerCounter
    kc = KmerCounter()
    k = spec["k"]
    canonical = spec["canonical"]
    min_abundance = spec["min_abundance"]
    top = spec["top"]

    if spec.get("paths"):
        payload = kc.count_fastas_aggregated(
            spec["paths"], k=k, canonical=canonical,
            min_abundance=min_abundance, top=top, on_progress=progress)
    elif spec.get("path"):
        payload = kc.count_fasta_aggregated(
            spec["path"], k=k, canonical=canonical,
            min_abundance=min_abundance, top=top, on_progress=progress)
    else:
        values, counts = kc.count(
            spec["sequences"], k=k, canonical=canonical,
            min_abundance=min_abundance)
        payload = _build_payload(kc, spec, values, counts)
    return _finalize_estimate(payload, min_abundance)


def _run_fastq_qc(spec: dict, progress) -> dict:
    """Control de calidad FastQC-like de un FASTQ en streaming (CPU/numpy)."""
    from Genoly.io.fastq import FastqReader
    from Genoly.qc.quality import QualityAnalyzer
    qa = QualityAnalyzer("cpu")
    return qa.analyze_stream(
        FastqReader(spec["path"]).records(),
        batch_size=spec.get("batch_size", 4096),
        max_position=spec.get("max_position", 300),
        on_progress=progress)


def _run_fastq_process(spec: dict, progress) -> dict:
    """Trim por calidad + filtro de un FASTQ en streaming; actualiza el
    registro de la subida de salida."""
    import os

    from Genoly.io.fastq import FastqReader
    from Genoly.qc.quality import QualityAnalyzer
    from ui.backend.uploads import set_bytes, set_stats

    qa = QualityAnalyzer("cpu")
    out_path = spec["out_path"]
    reads_in, reads_out, bases_in, bases_out = qa.process_stream(
        FastqReader(spec["path"]).records(),
        open(out_path, "w", encoding="utf-8"),
        min_quality=spec["min_quality"],
        window_size=spec["window_size"],
        min_length=spec["min_length"],
        min_mean_quality=spec["min_mean_quality"],
        max_n_ratio=spec["max_n_ratio"],
        on_progress=progress)

    out_upload_id = spec.get("out_upload_id")
    if out_upload_id:
        try:
            set_bytes(out_upload_id, int(os.path.getsize(out_path)))
            set_stats(out_upload_id, reads_out, bases_out, None)
        except Exception as exc:
            # el registro lo puede rellenar el frontend vía GET
            logger.warning("No se pudieron guardar las métricas de %s: %s",
                           out_upload_id, exc)

    return {
        "reads_in": reads_in,
        "reads_out": reads_out,
        "bases_in": bases_in,
        "bases_out": bases_out,
        "out_upload_id": out_upload_id,
        "out_path": out_path,
    }


def _run_map(spec: dict, progress) -> dict:
    """Mapea lecturas FASTQ contra una referencia FASTA (seed-and-extend)
    y, si ``sam_path`` está definido, escribe el SAM en streaming."""
    import os

    from Genoly.alignment.mapper import ReadMapper
    from Genoly.io.fastq import FastqReader
    from Genoly.io.sam import write_sam
    from ui.backend.uploads import set_bytes, set_stats

    mapper = ReadMapper(k=spec["k"], stride=spec["stride"],
                        min_seeds=spec.get("min_seeds", 2))

    def on_index(info: dict) -> None:
        progress({"stage": "index", **info})
    index = mapper.build_index(spec["ref_path"], on_progress=on_index)

    def on_map(info: dict) -> None:
        progress({"stage": "map", **info})
    reads = FastqReader(spec["reads_path"]).records()
    hits = []
    sam_path = spec.get("sam_path")
    if sam_path:
        pairs = mapper.iter_mapped(reads, index, on_progress=on_map)
        def collect():
            for read, hit in pairs:
                hits.append(hit)
                yield read, hit
        write_sam(sam_path, index.sequences, collect())
    else:
        for _, hit in mapper.iter_mapped(reads, index, on_progress=on_map):
            hits.append(hit)

    total = len(hits)
    mapped = [h for h in hits if h.mapped]
    sample = mapped[:spec.get("sample_limit", 500)]
    mean_id = sum(h.identity_percent for h in mapped) / len(mapped) \
        if mapped else 0.0
    mean_score = sum(h.score for h in mapped) / len(mapped) \
        if mapped else 0.0

    sam_upload_id = spec.get("sam_upload_id")
    if sam_upload_id and sam_path:
        try:
            set_bytes(sam_upload_id, int(os.path.getsize(sam_path)))
            set_stats(sam_upload_id, len(mapped), 0, None)
        except Exception as exc:
            logger.warning("No se pudieron guardar las métricas: %s", exc)

    def to_dict(h):
        return {
            "read_id": h.read_id, "mapped": h.mapped,
            "ref_name": h.ref_name, "ref_start": h.ref_start,
            "strand": h.strand, "score": h.score,
            "identity_percent": h.identity_percent,
            "cigar": h.cigar, "n_seeds": h.n_seeds,
        }

    return {
        "k": spec["k"], "stride": spec["stride"],
        "ref_sequences": len(index.sequences),
        "ref_bases": sum(len(s.seq) for s in index.sequences),
        "total_reads": total,
        "mapped": len(mapped),
        "unmapped": total - len(mapped),
        "mapping_rate": round(len(mapped) / total * 100, 2) if total else 0.0,
        "mean_identity": round(mean_id, 2),
        "mean_score": round(mean_score, 1),
        "hits": [to_dict(h) for h in sample],
        "sam_upload_id": sam_upload_id,
    }


def _run_variant_call(spec: dict, progress) -> dict:
    """Llamada de variantes de un SAM/BAM contra una referencia (por
    regiones, vectorizada) + exportación VCF."""
    import os

    from Genoly.variants.caller import VariantCaller
    from Genoly.variants.vcf import write_vcf
    from ui.backend.uploads import set_bytes, set_stats

    vc = VariantCaller()
    result = vc.call_variants_from_sam(
        spec["sam_path"], spec["ref_path"],
        min_depth=spec["min_depth"],
        min_alt_freq=spec["min_alt_freq"],
        region_size=spec.get("region_size", 50_000_000),
        on_progress=progress)

    out_vcf = spec["out_vcf_path"]
    write_vcf(out_vcf, result["by_sequence"], result["sequences"])
    out_id = spec.get("out_vcf_upload_id")
    if out_id:
        try:
            set_bytes(out_id, int(os.path.getsize(out_vcf)))
            set_stats(out_id, result["total_variants"], 0, None)
        except Exception as exc:
            logger.warning("No se pudieron guardar las métricas: %s", exc)

    sample = []
    for name, variants in result["by_sequence"].items():
        for v in variants[:200]:
            sample.append({
                "sequence": name, "position": v.position, "ref": v.ref,
                "alt": v.alt, "type": v.type, "depth": v.depth,
                "alt_count": v.alt_count, "freq": round(v.freq, 4),
            })
        if len(sample) >= 500:
            break

    return {
        "total_variants": result["total_variants"],
        "snvs": result["snvs"],
        "deletions": result["deletions"],
        "sequences": result["sequences"],
        "vcf_upload_id": out_id,
        "sample": sample[:500],
    }


def _run_haplotype(spec: dict, progress) -> dict:
    """Llamada de variantes estilo GATK HaplotypeCaller (regiones
    activas + ensamblaje local + PairHMM + genotipado PL/GQ)."""
    import os

    from Genoly.gatk.genotyper import HaplotypeCaller
    from Genoly.gatk.vcf import write_gatk_vcf
    from ui.backend.uploads import set_bytes, set_stats

    caller = HaplotypeCaller(
        min_mapq=spec.get("min_mapq", 20),
        min_depth=spec.get("min_depth", 5),
        min_qual=spec.get("min_qual", 10.0),
        min_af=spec.get("min_af", 0.15),
        window=spec.get("window", 50),
        extension=spec.get("extension", 50),
        max_haplotypes=spec.get("max_haplotypes", 32),
        ploidy=spec.get("ploidy", 2),
    )
    result = caller.call_from_sam(
        spec["sam_path"], spec["ref_path"],
        region_size=spec.get("region_size", 50_000_000),
        sample_name=spec.get("sample_name", "SAMPLE"),
        on_progress=progress)

    out_vcf = spec.get("out_vcf_path")
    out_id = spec.get("out_vcf_upload_id")
    if out_vcf:
        write_gatk_vcf(out_vcf, result["by_sequence"],
                       result["sequences"],
                       sample=result["sample_name"])
        if out_id:
            try:
                set_bytes(out_id, int(os.path.getsize(out_vcf)))
                set_stats(out_id, result["total_variants"], 0, None)
            except Exception as exc:
                logger.warning("No se pudieron guardar las métricas: %s",
                               exc)

    sample = []
    for name, variants in result["by_sequence"].items():
        for v in variants[:200]:
            sample.append({
                "sequence": name, "position": v.position, "ref": v.ref,
                "alt": v.alt, "type": v.type, "qual": v.qual,
                "filter": v.filter, "gt": v.gt, "dp": v.dp,
                "ad": v.ad, "gq": v.gq, "pl": v.pl,
                "af": v.af,
            })
        if len(sample) >= 500:
            break

    return {
        "total_variants": result["total_variants"],
        "snvs": result["snvs"],
        "indels": result["indels"],
        "pass_variants": result["pass_variants"],
        "sequences": result["sequences"],
        "sample_name": result["sample_name"],
        "vcf_upload_id": out_id,
        "sample": sample[:500],
        "pairhmm_backend": _pairhmm_backend(),
    }


def _pairhmm_backend() -> str:
    """Backend real del PairHMM en este proceso (para telemetría)."""
    try:
        from Genoly.gatk.pairhmm import has_native
        return "native (C)" if has_native() else "torch (GPU/CPU)"
    except Exception:
        return "desconocido"


def _run_bqsr(spec: dict, progress) -> dict:
    """Base Quality Score Recalibration (GATK BaseRecalibrator + apply)."""
    import json
    import os

    from Genoly.gatk.bqsr import (
        BaseRecalibrator,
        RecalTable,
        apply_recalibration,
        write_recal_sam,
    )
    from Genoly.gatk.reads import reads_from_sam
    from Genoly.io.fasta import FastaReader
    from ui.backend.uploads import set_bytes, set_stats

    refs = {rec.id: rec.sequence.upper()
            for rec in FastaReader(spec["ref_path"]).records()}
    reads = reads_from_sam(spec["sam_path"])
    if not reads:
        raise ValueError("el SAM/BAM no contiene lecturas mapeadas")

    progress({"stage": "bqsr_estimate", "reads": len(reads)})
    recal = BaseRecalibrator(
        min_base_quality=spec.get("min_base_quality", 0),
        min_mapq=spec.get("min_mapq", 0))

    by_ref: dict = {}
    for r in reads:
        by_ref.setdefault(r.rname, []).append(r)

    global_table = RecalTable()

    # los eventos del estimador se reetiquetan para la UI (estimación)
    def est_progress(p: dict) -> None:
        progress({"stage": "bqsr_estimate", **p})

    for name, subset in by_ref.items():
        ref = refs.get(name)
        if ref is None:
            continue
        table = recal.estimate(ref, subset, on_progress=est_progress)
        for key, (obs, err) in table.counts.items():
            slot = global_table.counts.setdefault(key, [0, 0])
            slot[0] += obs
            slot[1] += err
        global_table.n_bases += table.n_bases
        global_table.n_errors += table.n_errors
        global_table.max_cycle = max(global_table.max_cycle,
                                     table.max_cycle)

    deltas = [global_table.empirical_quality(*key) - key[0]
              for key in global_table.counts]

    out_sam = spec.get("out_sam_path")
    out_sam_id = spec.get("out_sam_upload_id")
    if out_sam:
        recal_reads = apply_recalibration(reads, global_table)
        ref_lengths = {name: len(refs[name]) for name in by_ref
                       if name in refs}
        write_recal_sam(out_sam, recal_reads, ref_lengths,
                        sample=spec.get("sample_name", "SAMPLE"))
        if out_sam_id:
            try:
                set_bytes(out_sam_id, int(os.path.getsize(out_sam)))
                set_stats(out_sam_id, len(recal_reads), 0, None)
            except Exception as exc:
                logger.warning("No se pudieron guardar las métricas: %s",
                               exc)

    payload = {
        "reads": len(reads),
        "bases": global_table.n_bases,
        "errors": global_table.n_errors,
        "global_rate": round(global_table.global_rate, 6),
        "covariates": len(global_table.counts),
        "mean_delta": round(sum(deltas) / len(deltas), 3) if deltas else 0.0,
        "table": global_table.to_dict(),
    }
    out_table = spec.get("out_table_path")
    table_id = spec.get("out_table_upload_id")
    if out_table:
        with open(out_table, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        if table_id:
            try:
                set_bytes(table_id, int(os.path.getsize(out_table)))
                set_stats(table_id, 1, 0, None)
            except Exception as exc:
                logger.warning("No se pudieron guardar las métricas: %s",
                               exc)

    return {
        "reads": payload["reads"],
        "bases": payload["bases"],
        "errors": payload["errors"],
        "global_rate": payload["global_rate"],
        "covariates": payload["covariates"],
        "mean_delta": payload["mean_delta"],
        "table_upload_id": table_id,
        "sam_upload_id": out_sam_id,
    }


def _run_markdups(spec: dict, progress) -> dict:
    """Marcado de duplicados estilo GATK MarkDuplicates."""
    import os

    from Genoly.gatk.bqsr import write_recal_sam
    from Genoly.gatk.markdups import apply_duplicate_flags, mark_duplicates
    from Genoly.gatk.reads import reads_from_sam
    from ui.backend.uploads import set_bytes, set_stats

    reads = reads_from_sam(spec["sam_path"])
    progress({"stage": "markdups", "reads": len(reads)})
    stats = mark_duplicates(reads)
    marked = set(stats["qnames"])

    out_sam = spec.get("out_sam_path")
    out_id = spec.get("out_sam_upload_id")
    if out_sam:
        flagged = apply_duplicate_flags(reads, marked)
        refs: dict = {}
        for r in reads:
            if r.rname:
                refs[r.rname] = max(refs.get(r.rname, 0),
                                    r.start + len(r.sequence))
        write_recal_sam(out_sam, flagged, refs,
                        sample=spec.get("sample_name", "SAMPLE"))
        if out_id:
            try:
                set_bytes(out_id, int(os.path.getsize(out_sam)))
                set_stats(out_id, len(flagged), 0, None)
            except Exception as exc:
                logger.warning("No se pudieron guardar las métricas: %s",
                               exc)

    return {
        "total_reads": stats["total"],
        "duplicates": stats["duplicates"],
        "duplicate_rate": round(stats["duplicate_rate"], 4),
        "keys": stats["libraries"],
        "sam_upload_id": out_id,
    }


def _run_downstream(spec: dict, progress) -> dict:
    """PCA / t-SNE / K-Means sobre una matriz de rasgos (genotipos, etc.)."""
    from Genoly.downstream.reduction import DownstreamAnalyzer
    analyzer = DownstreamAnalyzer()
    return analyzer.run(
        spec["matrix"],
        pca_components=spec.get("pca_components", 2),
        run_tsne=spec.get("run_tsne", True),
        tsne_perplexity=spec.get("tsne_perplexity", 30.0),
        tsne_iter=spec.get("tsne_iter", 500),
        k=spec.get("k", 3),
        standardize=spec.get("standardize", True),
        on_progress=progress)


def _run_annotation(spec: dict, progress) -> dict:
    """Anota variantes con genes/regiones/GO desde un GFF/GTF."""
    from Genoly.annotation.annotator import Annotator, summarize_go
    an = Annotator()
    n_features = an.load(spec["gff_path"], on_progress=progress)
    res = an.annotate_variants(spec["variants"], on_progress=progress)
    res["features_loaded"] = n_features
    res["go_summary"] = summarize_go(res["go_terms"])
    return res


def _run_report(spec: dict, progress) -> dict:
    """Genera heatmap y/o volcano SVG + reporte Markdown/JSON."""
    import os

    from Genoly.report.plots import build_report, heatmap_svg, volcano_svg
    from ui.backend.uploads import set_bytes, set_stats

    out: dict = {}
    if spec.get("matrix"):
        progress({"stage": "heatmap"})
        out["heatmap_svg"] = heatmap_svg(
            spec["matrix"], spec.get("row_labels"), spec.get("col_labels"),
            title=spec.get("title", "Heatmap"))
    if spec.get("fold_changes") and spec.get("p_values"):
        progress({"stage": "volcano"})
        out["volcano_svg"] = volcano_svg(
            spec["fold_changes"], spec["p_values"],
            fc_threshold=spec.get("fc_threshold", 1.0),
            p_threshold=spec.get("p_threshold", 0.05),
            title=spec.get("title", "Volcano plot"))

    report = build_report(spec.get("title", "Reporte de análisis"),
                          spec.get("sections", {}))
    out["markdown"] = report.markdown
    out["report_json"] = report.json
    out["sections"] = report.sections

    # guarda el markdown como subida descargable
    md_id = spec.get("md_upload_id")
    if spec.get("md_path"):
        with open(spec["md_path"], "w", encoding="utf-8") as fh:
            fh.write(report.markdown)
        if md_id:
            try:
                set_bytes(md_id, int(os.path.getsize(spec["md_path"])))
                set_stats(md_id, 1, 0, None)
            except Exception as exc:
                logger.warning("No se pudieron guardar las métricas: %s",
                               exc)
            out["md_upload_id"] = md_id
    return out


def _clean_options(spec: dict) -> dict:
    """Opciones comunes de clean_grid del spec del trabajo."""
    return {
        "phenotype_col": spec["phenotype_col"],
        "impute_method": spec["impute_method"],
        "max_column_missingness": spec["max_column_missingness"],
        "min_individuals": spec["min_individuals"],
        "min_markers": spec["min_markers"],
        "min_maf": spec.get("min_maf", 0.0),
        "hwe_p": spec.get("hwe_p"),
        "id_col": spec.get("id_col"),
        "sire_col": spec.get("sire_col"),
        "dam_col": spec.get("dam_col"),
    }


def _write_clean_payload(phenotypes, genotypes, markers, pedigree, report,
                         spec) -> dict:
    """Serializa el resultado de clean_grid y actualiza las métricas."""
    import json
    import os

    from ui.backend.uploads import set_bytes, set_stats

    payload = {
        "phenotypes": phenotypes,
        "genotypes": genotypes,
        "markers": markers,
        "pedigree": pedigree,
        "report": report,
    }
    with open(spec["out_path"], "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    out_id = spec.get("out_upload_id")
    if out_id:
        try:
            set_bytes(out_id, int(os.path.getsize(spec["out_path"])))
            set_stats(out_id, len(phenotypes),
                      len(phenotypes) * len(markers), None)
        except Exception as exc:
            logger.warning("No se pudieron guardar las métricas: %s", exc)

    return {"report": report, "clean_id": out_id,
            "n_individuals": len(phenotypes), "n_markers": len(markers)}


def _run_qdata_clean(spec: dict, progress) -> dict:
    """Limpia/imputa/filtra una matriz CSV/Excel sucia y guarda el JSON."""
    from Genoly.quantitative.preprocess import clean_grid, load_grid

    progress({"stage": "load"})
    grid = load_grid(spec["path"])
    progress({"stage": "load", "rows": len(grid)})

    phenotypes, genotypes, markers, pedigree, report = clean_grid(
        grid, **_clean_options(spec))
    progress({"stage": "clean", "rows": len(phenotypes),
              "markers": len(markers)})

    return _write_clean_payload(phenotypes, genotypes, markers, pedigree,
                                report, spec)


_MISSING_CELL = {"", "na", "nan", "null", "none", "-", "nd"}


def _cell_missing(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _MISSING_CELL


def _passes_filters(row, filters: dict) -> bool:
    """Semántica del filtrado por definición (equivalente a FiltableGrid)."""
    for key, f in filters.items():
        col = int(key)
        raw = row[col] if col < len(row) else None
        if _cell_missing(raw):
            if not f.get("vacios", True):
                return False
        else:
            allowed = f.get("vals")
            if allowed is not None and str(raw).strip() not in set(allowed):
                return False
    return True


def _run_apply_filters(spec: dict, progress) -> dict:
    """Filtra por definición/ediciones y limpia la grid en el servidor."""
    from Genoly.quantitative.preprocess import clean_grid, load_grid

    progress({"stage": "load"})
    full = load_grid(spec["path"])
    header = full[0] if (spec.get("header") and full) else None
    grid = full[1:] if header is not None else full
    progress({"stage": "load", "rows": len(grid)})

    filters = spec.get("filters") or {}
    if filters:
        grid = [row for row in grid if _passes_filters(row, filters)]
    if not grid:
        raise ValueError("La grid está vacía — ninguna fila sobrevivió al filtro")

    offset = 1 if header is not None else 0
    for edit in spec.get("edits") or []:
        r, c = int(edit["row"]) + offset, int(edit["col"])
        if 0 <= r < len(grid):
            while len(grid[r]) <= c:
                grid[r].append("")
            grid[r][c] = str(edit.get("value", "")).strip()

    if header is not None:
        grid = [header] + grid
    progress({"stage": "filter", "rows": len(grid)})
    phenotypes, genotypes, markers, pedigree, report = clean_grid(
        grid, **_clean_options(spec))
    progress({"stage": "clean", "rows": len(phenotypes),
              "markers": len(markers)})

    return _write_clean_payload(phenotypes, genotypes, markers, pedigree,
                                report, spec)


def _run_crossval(spec: dict, progress) -> dict:
    from Genoly.quantitative.crossval import crossval
    return crossval(
        spec["phenotypes"], spec["genotypes"],
        kinship=spec.get("kinship", "vanraden"),
        n_folds=spec.get("n_folds", 5),
        n_repeats=spec.get("n_repeats", 1),
        seed=spec.get("seed", 0),
        on_progress=progress)


def _run_gwas(spec: dict, progress) -> dict:
    from Genoly.quantitative.gwas import gwas
    return gwas(
        spec["phenotypes"], spec["genotypes"],
        kinship=spec.get("kinship", "vanraden"),
        min_maf=spec.get("min_maf", 0.01),
        on_progress=progress)


def _run_kinship(spec: dict, progress) -> dict:
    from Genoly.quantitative.popgen import grm_analysis
    progress({"stage": "grm"})
    return grm_analysis(spec["genotypes"],
                        kinship=spec.get("kinship", "vanraden"),
                        top_relatives=spec.get("top_relatives", 10))


def _crash_test(spec: dict) -> None:
    """Simula distintos tipos de muerte del trabajador (solo tests)."""
    mode = spec.get("mode", "raise")
    if mode == "exit":
        import os
        os._exit(1)  # muere sin mensaje (como un OOM)
    if mode == "segv":
        import ctypes
        ctypes.string_at(0)  # segfault: el proceso hijo muere, no el padre
    raise RuntimeError("crash simulado (error controlado)")


def run_job(spec: dict, conn: "mp.connection.Connection") -> None:
    """
    Entrypoint del proceso hijo: despacha el trabajo y reporta por ``conn``.

    Args:
        spec: Especificación picklable del trabajo
            (``{"kind": "kmer", ...}`` o ``{"kind": "crash_test", ...}``).
        conn: Conexión de escritura hacia el padre.
    """
    try:
        kind = spec["kind"]
        if kind == "kmer":
            result = _run_kmer(spec, _progress(conn))
        elif kind == "fastq_qc":
            result = _run_fastq_qc(spec, _progress(conn))
        elif kind == "fastq_process":
            result = _run_fastq_process(spec, _progress(conn))
        elif kind == "map":
            result = _run_map(spec, _progress(conn))
        elif kind == "variant_call":
            result = _run_variant_call(spec, _progress(conn))
        elif kind == "haplotype":
            result = _run_haplotype(spec, _progress(conn))
        elif kind == "bqsr":
            result = _run_bqsr(spec, _progress(conn))
        elif kind == "markdups":
            result = _run_markdups(spec, _progress(conn))
        elif kind == "downstream":
            result = _run_downstream(spec, _progress(conn))
        elif kind == "annotation":
            result = _run_annotation(spec, _progress(conn))
        elif kind == "report":
            result = _run_report(spec, _progress(conn))
        elif kind == "qdata_clean":
            result = _run_qdata_clean(spec, _progress(conn))
        elif kind == "apply_filters":
            result = _run_apply_filters(spec, _progress(conn))
        elif kind == "crossval":
            result = _run_crossval(spec, _progress(conn))
        elif kind == "gwas":
            result = _run_gwas(spec, _progress(conn))
        elif kind == "kinship":
            result = _run_kinship(spec, _progress(conn))
        elif kind == "crash_test":
            _crash_test(spec)
            result = {"ok": True}
        else:
            raise ValueError(f"tipo de trabajo desconocido: {kind!r}")
        conn.send(("result", result))
    except Exception as exc:
        try:
            conn.send(("error", f"{exc.__class__.__name__}: {exc}"))
        except Exception:
            pass
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


def _run_qdata_clean(spec: dict, progress) -> dict:
    """Limpia/imputa/filtra una matriz CSV/Excel sucia y guarda el JSON."""
    import json
    import os

    from Genoly.quantitative.preprocess import clean_grid, load_grid
    from ui.backend.uploads import set_bytes, set_stats

    progress({"stage": "load"})
    grid = load_grid(spec["path"])
    progress({"stage": "load", "rows": len(grid)})

    phenotypes, genotypes, markers, report = clean_grid(
        grid,
        phenotype_col=spec["phenotype_col"],
        impute_method=spec["impute_method"],
        max_column_missingness=spec["max_column_missingness"],
        min_individuals=spec["min_individuals"],
        min_markers=spec["min_markers"],
        min_maf=spec.get("min_maf", 0.0),
        hwe_p=spec.get("hwe_p"),
    )
    progress({"stage": "clean", "rows": len(phenotypes),
              "markers": len(markers)})

    payload = {
        "phenotypes": phenotypes,
        "genotypes": genotypes,
        "markers": markers,
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


def _run_apply_filters(spec: dict, progress) -> dict:
    """Limpia/imputa/filtra una grid ya filtrada por el cliente (frontend)."""
    import json
    import os

    from Genoly.quantitative.preprocess import clean_grid
    from ui.backend.uploads import set_bytes, set_stats

    progress({"stage": "load"})
    # spec["grid_rows"] viene del frontend: lista de listas tras filtrado interactivo
    grid = [list(row) for row in spec.get("grid_rows", [])]
    if not grid:
        raise ValueError("La grid está vacía — ninguna fila sobrevivió al filtro")

    phenotypes, genotypes, markers, report = clean_grid(
        grid,
        phenotype_col=spec["phenotype_col"],
        impute_method=spec.get("impute_method", "media"),
        max_column_missingness=spec.get("max_column_missingness", 1.0),
        min_individuals=spec.get("min_individuals", 5),
        min_markers=spec.get("min_markers", 2),
        min_maf=spec.get("min_maf", 0.0),
        hwe_p=spec.get("hwe_p"),
    )
    progress({"stage": "clean", "rows": len(phenotypes),
              "markers": len(markers)})

    payload = {
        "phenotypes": phenotypes,
        "genotypes": genotypes,
        "markers": markers,
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
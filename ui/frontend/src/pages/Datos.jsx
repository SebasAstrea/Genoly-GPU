import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api.js'
import { Card, StatCard, Badge, Bar, PageHeader } from '../components/ui.jsx'
import { useQData } from '../qdata.jsx'
import FiltableGrid from './FiltableGrid.jsx'
import {
  IconServer,
  IconList,
  IconCog,
  IconPercent,
  IconStack,
  IconPlay,
  IconTarget,
  IconRefresh,
  IconPulse,
  IconDna,
} from '../components/icons.jsx'

const CHUNK_SIZE = 32 * 1024 * 1024
const CHUNK_THRESHOLD = 1024 * 1024 * 1024
// por encima de este tamaño la tabla interactiva deja de cargar la grid
const GRID_MAX_ROWS = 20000

const DEFAULT_OPTIONS = {
  phenotype_col: 0,
  impute_method: 'media',
  max_column_missingness: 1.0,
  min_individuals: 5,
  min_markers: 2,
  min_maf: 0.0,
  hwe_p: '',
  id_col: null,
  sire_col: null,
  dam_col: null,
  dataset_kind: 'mixto',
}

// autodetección de columnas de pedigrí por nombre de cabecera
const ID_HINTS = ['id', 'animal', 'individual', 'individuo', 'head', 'cabeza']
const SIRE_HINTS = ['sire', 'padre', 'father', 'toro', 'p']
const DAM_HINTS = ['dam', 'madre', 'mother', 'vaca', 'm']

function detectColumn(names, hints) {
  const lower = (names || []).map((n) => String(n ?? '').trim().toLowerCase())
  for (const hint of hints) {
    const i = lower.indexOf(hint)
    if (i >= 0) return i
  }
  // coincidencia laxa sólo con pistas de 3+ letras (evita 'p'/'m' solos)
  for (let i = 0; i < lower.length; i++) {
    if (hints.filter((h) => h.length >= 3).some((h) => lower[i].includes(h))) return i
  }
  return null
}

function fmt(n) {
  return n?.toLocaleString()
}

const TYPE_TONE = { numeric: 'ok', text: 'slate', mixed: 'warn', empty: 'bad' }

function Scatter({ points, labels, xLabel, yLabel, title, identity = false }) {
  if (!points?.length) return null
  const W = 520
  const H = 300
  const P = 36
  const xs = points.map((p) => p[0])
  const ys = points.map((p) => p[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys, minX)
  const maxY = Math.max(...ys, maxX)
  const lo = Math.min(minX, minY)
  const hi = Math.max(maxX, maxY)
  const span = hi - lo || 1
  const sx = (v) => P + ((v - lo) / span) * (W - 2 * P)
  const sy = (v) => H - P - ((v - lo) / span) * (H - 2 * P)
  const c = sx(lo)
  const d = sy(hi)
  return (
    <div>
      <p className="mb-2 text-xs text-ink-faint">{title}</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <line x1={P} x2={W - P} y1={H - P} y2={H - P} stroke="rgb(var(--line) / 0.4)" />
        <line x1={P} x2={P} y1={P} y2={H - P} stroke="rgb(var(--line) / 0.4)" />
        {identity && (
          <line x1={c} y1={d} x2={sx(hi)} y2={sy(lo)} stroke="rgb(var(--line) / 0.5)" strokeDasharray="4 4" />
        )}
        {points.map((p, i) => (
          <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={3.4}
            fill="rgb(var(--accent))" fillOpacity={0.8}
            style={{ filter: 'drop-shadow(0 0 3px rgb(var(--accent) / 0.5))' }} />
        ))}
        <text x={W / 2} y={H - 4} textAnchor="middle" className="font-mono"
          style={{ fill: 'rgb(var(--ink-faint))', fontSize: 10 }}>{xLabel}</text>
        <text x={12} y={H / 2} textAnchor="middle" transform={`rotate(-90 12 ${H / 2})`}
          className="font-mono" style={{ fill: 'rgb(var(--ink-faint))', fontSize: 10 }}>{yLabel}</text>
      </svg>
    </div>
  )
}

export default function Datos() {
  const inputRef = useRef(null)
  const { qdata, setQdata } = useQData()
  const navigate = useNavigate()
  const [upload, setUpload] = useState(null)
  const [serverPath, setServerPath] = useState('')
  const [preview, setPreview] = useState(null)
  const [options, setOptions] = useState(DEFAULT_OPTIONS)
  const [running, setRunning] = useState(false)
  const [cleanResult, setCleanResult] = useState(null)
  const [progress, setProgress] = useState(null)
  const [error, setError] = useState(null)
  const [dryRun, setDryRun] = useState({ loading: false, result: null, error: null })
  const [cv, setCv] = useState({ running: false, result: null, error: null })
  const [cvParams, setCvParams] = useState({ n_folds: 5, n_repeats: 1 })
  const [gwas, setGwas] = useState({ running: false, result: null, error: null })
  const [gwasMaf, setGwasMaf] = useState(0.05)
  const [kin, setKin] = useState({ running: false, result: null, error: null })
  const recipeRef = useRef(null)

  const uploadFile = async (file) => {
    if (file.size > CHUNK_THRESHOLD) {
      const init = await api.uploadChunkedInit(file.name, file.size)
      const uploadId = init.upload_id
      let offset = init.received
      try {
        const st = await api.uploadChunkedGet(uploadId)
        offset = Math.min(st.received, file.size)
      } catch {
        // sin estado previo
      }
      while (offset < file.size) {
        const end = Math.min(offset + CHUNK_SIZE, file.size)
        const res = await api.uploadChunkedPut(uploadId, offset, file.slice(offset, end))
        offset = res.received
      }
      return api.uploadChunkedComplete(uploadId, file.size)
    }
    return api.upload(file)
  }

  const handleFile = async (file) => {
    if (!file) return
    setError(null)
    setPreview(null)
    setCleanResult(null)
    try {
      const res = await uploadFile(file)
      setUpload(res)
      if (res.stats_status === 'pending') pollStats(res.upload_id)
      else loadPreview(res.upload_id)
    } catch (e) {
      setError(e.message)
    }
  }

  const pollStats = async (id) => {
    const st = await api.uploadStats(id)
    if (st.stats_status === 'pending') {
      setTimeout(() => pollStats(id), 2000)
      return
    }
    setUpload(st)
    loadPreview(id)
  }

  const registerServer = async (path) => {
    setError(null)
    setPreview(null)
    setCleanResult(null)
    try {
      const res = await api.registerUpload(path)
      setUpload(res)
      if (res.stats_status === 'pending') pollStats(res.upload_id)
      else loadPreview(res.upload_id)
    } catch (e) {
      setError(e.message)
    }
  }

  const loadPreview = async (id) => {
    try {
      setPreview(await api.qdataPreview(id))
    } catch (e) {
      setError(e.message)
    }
  }

  // mantener la columna de fenotipo válida y autodetectar pedigrí al perfilar
  useEffect(() => {
    if (!preview?.columns?.length) return
    const max = preview.columns.length - 1
    const names = preview.column_names || []
    setOptions((o) => {
      const next = { ...o, phenotype_col: Math.min(o.phenotype_col, max) }
      if (next.id_col == null || next.id_col > max) next.id_col = detectColumn(names, ID_HINTS)
      if (next.sire_col == null || next.sire_col > max) next.sire_col = detectColumn(names, SIRE_HINTS)
      if (next.dam_col == null || next.dam_col > max) next.dam_col = detectColumn(names, DAM_HINTS)
      if (next.sire_col != null || next.dam_col != null) next.dataset_kind = next.dataset_kind || 'mixto'
      return next
    })
  }, [preview])

  const cleanPayload = (opts = options) => ({
    upload_id: upload.upload_id,
    phenotype_col: opts.phenotype_col,
    impute_method: opts.impute_method,
    max_column_missingness: opts.max_column_missingness,
    min_individuals: opts.min_individuals,
    min_markers: opts.dataset_kind === 'pedigree' ? 0 : opts.min_markers,
    min_maf: opts.min_maf,
    hwe_p: opts.hwe_p ? Number(opts.hwe_p) : null,
    id_col: opts.id_col,
    sire_col: opts.sire_col,
    dam_col: opts.dam_col,
  })

  const runDryRun = async () => {
    if (!upload) return
    setDryRun((s) => ({ ...s, loading: true, error: null }))
    try {
      const result = await api.qdataPreviewClean(cleanPayload())
      setDryRun({ loading: false, result, error: null })
    } catch (e) {
      setDryRun({ loading: false, result: null, error: e.message })
    }
  }

  // vista previa de limpieza automática al cargar el archivo o cambiar opciones
  useEffect(() => {
    if (!upload?.upload_id || !preview) return
    const t = setTimeout(() => { runDryRun() }, 300)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upload?.upload_id, preview, options])

  const exportRecipe = () => {
    const blob = new Blob([JSON.stringify(options, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'genoly-receta-limpieza.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const importRecipe = (file) => {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result))
        setOptions((o) => ({ ...o, ...parsed }))
      } catch {
        setError('La receta no es un JSON válido')
      }
    }
    reader.readAsText(file)
  }

  const run = async () => {
    if (!upload) return
    setRunning(true)
    setError(null)
    setCleanResult(null)
    try {
      const job = await api.qdataClean(cleanPayload())
      const res = await api.jobEvents(job.job_id, { onProgress: (p) => setProgress(p) })
      const matrix = await api.qdataGet(res.clean_id)
      setQdata({ ...matrix, source: upload.filename })
      setCleanResult({ ...res, n_individuals: matrix.phenotypes.length })
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
      setProgress(null)
    }
  }

  // Interactive grid "Aplicar y continuar": filtros por definición + ediciones
  const handleGridCleaned = async ({ filters = {}, edits = [] } = {}) => {
    if (!upload) return
    setRunning(true)
    setError(null)
    setCleanResult(null)
    try {
      const filtersObj = {}
      for (const [col, f] of Object.entries(filters)) {
        filtersObj[col] = { vals: [...f.vals], vacios: !!f.vacios }
      }
      const job = await api.qdataApplyFilters({
        ...cleanPayload(),
        upload_id: upload.upload_id,
        header: !!preview?.header_detected,
        filters: filtersObj,
        edits,
      })
      const res = await api.jobEvents(job.job_id, { onProgress: (p) => setProgress(p) })
      const matrix = await api.qdataGet(res.clean_id)
      setQdata({ ...matrix, source: upload.filename + ' [filtrado]' })
      setCleanResult({ ...res, n_individuals: matrix.phenotypes.length })
    } catch (e) {
      setError(e.message || 'Error al aplicar limpieza')
    } finally {
      setRunning(false)
      setProgress(null)
    }
  }

  const sendTo = (path) => navigate(path)

  const runAnalysis = async (fn, payload, setState, setProgressFn) => {
    setState((s) => ({ ...s, running: true, error: null, result: null }))
    try {
      const job = await fn(payload)
      const res = await api.jobEvents(job.job_id, {
        onProgress: (p) => setProgressFn?.(p),
      })
      setState({ running: false, result: res, error: null })
    } catch (e) {
      setState((s) => ({ ...s, running: false, error: e.message }))
    }
  }

  const runCv = () => {
    if (!qdata) return
    runAnalysis(api.qdataCrossval, {
      phenotypes: qdata.phenotypes,
      genotypes: qdata.genotypes,
      kinship: 'vanraden',
      n_folds: cvParams.n_folds,
      n_repeats: cvParams.n_repeats,
      seed: 0,
    }, setCv)
  }

  const runGwas = () => {
    if (!qdata) return
    runAnalysis(api.qdataGwas, {
      phenotypes: qdata.phenotypes,
      genotypes: qdata.genotypes,
      kinship: 'vanraden',
      min_maf: gwasMaf,
    }, setGwas)
  }

  const runKin = () => {
    if (!qdata) return
    runAnalysis(api.qdataKinship, {
      genotypes: qdata.genotypes,
      kinship: 'vanraden',
    }, setKin)
  }

  const columns = preview?.columns ?? []
  const maxMissing = Math.max(...columns.map((c) => c.missing_pct), 0)
  const report = cleanResult?.report
  // CV/GWAS/parentesco necesitan marcadores genómicos (no dataset solo pedigrí)
  const hasMarkers = (qdata?.genotypes?.[0]?.length ?? 0) > 0

  return (
    <div className="space-y-8">
      <PageHeader
        index="Datos · Genética cuantitativa"
        title="Carga y limpieza"
        subtitle="CSV/Excel robusto: perfilado, limpieza, imputación y filtrado antes de modelar"
        icon={<IconServer className="h-6 w-6" />}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="space-y-4">
          <Card title="Archivo" subtitle="CSV/TSV/Excel (los grandes se suben por partes)"
            icon={<IconServer className="h-5 w-5" />}>
            <div className="flex items-center gap-2">
              <button className="btn-primary" onClick={() => inputRef.current?.click()}>
                Cargar CSV / Excel
              </button>
              <span className="truncate font-mono text-xs text-ink-faint">
                {upload?.filename ?? '—'}
              </span>
            </div>
            <input ref={inputRef} type="file" accept=".csv,.tsv,.xlsx,.xls" className="hidden"
              onChange={(e) => { handleFile(e.target.files?.[0]); e.target.value = '' }} />
            <div className="mt-2 flex items-center gap-2">
              <input className="input flex-1 font-mono text-xs" placeholder="/ruta/datos.csv"
                value={serverPath} onChange={(e) => setServerPath(e.target.value)} spellCheck={false} />
              <button className="btn-ghost" onClick={() => registerServer(serverPath.trim())}
                disabled={!serverPath.trim()}>Registrar</button>
            </div>
            {upload?.records != null && (
              <p className="mt-1 text-[10px] text-ink-faint">
                {fmt(upload.records)} filas · {upload.total_bases != null && upload.records
                  ? `${Math.round(upload.total_bases / Math.max(1, upload.records))} columnas`
                  : ''}
              </p>
            )}
          </Card>

          <Card title="Limpieza" subtitle="Opciones aplicadas al limpiar"
            icon={<IconCog className="h-5 w-5" />}>
            <label className="label">Tipo de dataset</label>
            <select className="input" value={options.dataset_kind}
              onChange={(e) => setOptions({ ...options, dataset_kind: e.target.value })}>
              <option value="mixto">Mixto (fenotipo + marcadores [+ pedigrí])</option>
              <option value="pedigree">Solo pedigrí (sin marcadores)</option>
            </select>
            <label className="label mt-3">Columna de fenotipo</label>
            <select className="input" value={options.phenotype_col}
              onChange={(e) => setOptions({ ...options, phenotype_col: Number(e.target.value) })}>
              {columns.map((c) => (
                <option key={c.index} value={c.index}>
                  {c.index}: {c.name} ({c.type})
                </option>
              ))}
            </select>
            <div className="mt-3 grid grid-cols-3 gap-2">
              <div>
                <label className="label">ID</label>
                <select className="input" value={options.id_col ?? ''}
                  onChange={(e) => setOptions({ ...options, id_col: e.target.value === '' ? null : Number(e.target.value) })}>
                  <option value="">— ninguno —</option>
                  {columns.map((c) => (
                    <option key={c.index} value={c.index}>{c.index}: {c.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="label">Padre</label>
                <select className="input" value={options.sire_col ?? ''}
                  onChange={(e) => setOptions({ ...options, sire_col: e.target.value === '' ? null : Number(e.target.value) })}>
                  <option value="">— ninguno —</option>
                  {columns.map((c) => (
                    <option key={c.index} value={c.index}>{c.index}: {c.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="label">Madre</label>
                <select className="input" value={options.dam_col ?? ''}
                  onChange={(e) => setOptions({ ...options, dam_col: e.target.value === '' ? null : Number(e.target.value) })}>
                  <option value="">— ninguno —</option>
                  {columns.map((c) => (
                    <option key={c.index} value={c.index}>{c.index}: {c.name}</option>
                  ))}
                </select>
              </div>
            </div>
            <p className="mt-1 text-[10px] text-ink-faint">
              El pedigrí (padre/madre) se conserva aunque no sea numérico y se usa para PBLUP.
            </p>

            {options.dataset_kind === 'mixto' && (
              <>
                <label className="label mt-3">Imputación de dosis perdidas</label>
                <select className="input" value={options.impute_method}
                  onChange={(e) => setOptions({ ...options, impute_method: e.target.value })}>
                  <option value="media">Media</option>
                  <option value="moda">Moda</option>
                </select>
                <label className="label mt-3">Máx. datos perdidos por marcador ({Math.round(options.max_column_missingness * 100)}%)</label>
                <input type="range" min={0} max={1} step={0.05} className="w-full accent-accent"
                  value={options.max_column_missingness}
                  onChange={(e) => setOptions({ ...options, max_column_missingness: Number(e.target.value) })} />
              </>
            )}
            <div className="mt-3 grid grid-cols-2 gap-3">
              <div>
                <label className="label">Mín. individuos</label>
                <input type="number" min={3} className="input" value={options.min_individuals}
                  onChange={(e) => setOptions({ ...options, min_individuals: Number(e.target.value) })} />
              </div>
              <div>
                <label className="label">Mín. marcadores</label>
                <input type="number" min={0} className="input" disabled={options.dataset_kind === 'pedigree'}
                  value={options.dataset_kind === 'pedigree' ? 0 : options.min_markers}
                  onChange={(e) => setOptions({ ...options, min_markers: Number(e.target.value) })} />
              </div>
            </div>
            {options.dataset_kind === 'mixto' && (
              <div className="mt-3 grid grid-cols-2 gap-3">
                <div>
                  <label className="label">MAF mín. (0–0.5)</label>
                  <input type="number" min={0} max={0.5} step={0.01} className="input"
                    value={options.min_maf}
                    onChange={(e) => setOptions({ ...options, min_maf: Number(e.target.value) })} />
                </div>
                <div>
                  <label className="label">HWE (p mín.)</label>
                  <select className="input" value={options.hwe_p}
                    onChange={(e) => setOptions({ ...options, hwe_p: e.target.value })}>
                    <option value="">Ninguno</option>
                    <option value="0.05">0.05</option>
                    <option value="0.01">0.01</option>
                    <option value="0.001">0.001</option>
                  </select>
                </div>
              </div>
            )}
            {options.dataset_kind === 'mixto' && (
              <p className="mt-1 text-[10px] text-ink-faint">
                MAF = frecuencia alélica menor; HWE = equilibrio de Hardy-Weinberg. Descartan marcadores de baja calidad.
              </p>
            )}

            <div className="mt-3 flex gap-2">
              <button className="btn-ghost flex-1 text-xs" onClick={exportRecipe} disabled={!upload}>
                Exportar receta
              </button>
              <button className="btn-ghost flex-1 text-xs" onClick={() => recipeRef.current?.click()}>
                Importar receta
              </button>
              <input ref={recipeRef} type="file" accept="application/json,.json" className="hidden"
                onChange={(e) => { importRecipe(e.target.files?.[0]); e.target.value = '' }} />
            </div>

            {/* Vista previa de la limpieza (dry-run) */}
            {upload && (
              <div className="mt-3 rounded-lg border border-line/30 bg-panel-2/40 p-3">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold text-ink">Impacto de la limpieza (vista previa)</p>
                  <button className="text-[11px] text-accent hover:underline" onClick={runDryRun}
                    disabled={dryRun.loading}>
                    {dryRun.loading ? 'Calculando…' : 'Recalcular'}
                  </button>
                </div>
                {dryRun.error && (
                  <p className="mt-2 rounded border border-bad/40 bg-bad/10 px-2 py-1 text-[11px] text-bad">{dryRun.error}</p>
                )}
                {dryRun.result && (
                  <div className="mt-2 space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <p className="text-ink-faint">
                        Filas: <span className="font-mono text-ink">{fmt(dryRun.result.report.rows_read)}</span>
                        {' → '}
                        <span className="font-mono text-accent-glow">{fmt(dryRun.result.projected.rows)}</span>
                      </p>
                      <p className="text-ink-faint">
                        Marcadores: <span className="font-mono text-ink">{fmt(dryRun.result.report.columns_total)}</span>
                        {' → '}
                        <span className="font-mono text-accent-glow">{fmt(dryRun.result.projected.markers)}</span>
                      </p>
                    </div>
                    {dryRun.result.blocking_errors?.length > 0 && (
                      <ul className="space-y-1">
                        {dryRun.result.blocking_errors.map((msg, i) => (
                          <li key={i} className="rounded border border-bad/40 bg-bad/10 px-2 py-1 text-[11px] text-bad">
                            {msg}
                          </li>
                        ))}
                      </ul>
                    )}
                    {dryRun.result.qc_flags?.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {dryRun.result.qc_flags.map((f) => (
                          <span key={f.code} title={f.message}
                            className="rounded-full bg-warn/20 px-2 py-0.5 text-[10px] font-semibold text-warn">
                            {f.message}
                          </span>
                        ))}
                      </div>
                    )}
                    {dryRun.result.report.dropped_columns?.length > 0 && (
                      <p className="text-[11px] text-ink-faint">
                        {dryRun.result.report.dropped_columns.length} columna(s) descartada(s) por tipo/perdidos.
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}

            <button className="btn-primary mt-3 w-full" onClick={run} disabled={running || !upload}>
              <IconPlay className="h-4 w-4" />
              {running ? 'Limpiando…' : 'Limpiar y preparar'}
            </button>
            {running && progress?.stage === 'clean' && (
              <p className="mt-2 text-xs text-ink-faint">
                {fmt(progress.rows)} individuos · {progress.markers} marcadores
              </p>
            )}
            {error && (
              <p className="mt-3 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
                {error}
              </p>
            )}
          </Card>

          {qdata && (
            <>
              <Card title="Enviar al análisis" subtitle="La matriz limpia está lista"
                icon={<IconTarget className="h-5 w-5" />}>
                <button className="btn-primary w-full" onClick={() => sendTo('/quantitative')}>
                  Enviar a Modelos mixtos
                </button>
                <button className="btn-ghost mt-2 w-full" onClick={() => sendTo('/gblup')}>
                  Enviar a GBLUP
                </button>
                <button className="btn-ghost mt-2 w-full" onClick={() => sendTo('/pblup')}>
                  Enviar a PBLUP
                </button>
                <p className="mt-2 text-[11px] text-ink-faint">
                  En sesión: <span className="font-mono text-accent-glow">{qdata.source}</span>
                </p>
              </Card>

              <Card title="Pedigrí" subtitle="Resumen de las columnas de parentesco"
                icon={<IconDna className="h-5 w-5" />}>
                {qdata.pedigree ? (
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <dt className="text-ink-faint">Individuos</dt>
                      <dd className="font-mono text-accent-glow">
                        {qdata.pedigree.individuals?.length ?? 0}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-ink-faint">Padres listados</dt>
                      <dd className="font-mono text-accent-glow">
                        {(qdata.pedigree.sires ?? []).filter((s) => s != null).length}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-ink-faint">Madres listadas</dt>
                      <dd className="font-mono text-accent-glow">
                        {(qdata.pedigree.dams ?? []).filter((d) => d != null).length}
                      </dd>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-ink-faint">
                    Sin columnas de pedigrí: PBLUP no estará disponible hasta
                    que selecciones ID/Padre/Madre y vuelvas a limpiar.
                  </p>
                )}
              </Card>

              {hasMarkers && (
                <>
              <Card title="Validación cruzada" subtitle="Exactitud de la predicción GBLUP (K-fold)"
                icon={<IconTarget className="h-5 w-5" />}>
                <div className="flex items-end gap-3">
                  <div>
                    <label className="label">Pliegues</label>
                    <input type="number" min={2} className="input w-24" value={cvParams.n_folds}
                      onChange={(e) => setCvParams({ ...cvParams, n_folds: Number(e.target.value) })} />
                  </div>
                  <div>
                    <label className="label">Repeticiones</label>
                    <input type="number" min={1} className="input w-24" value={cvParams.n_repeats}
                      onChange={(e) => setCvParams({ ...cvParams, n_repeats: Number(e.target.value) })} />
                  </div>
                  <button className="btn-primary" onClick={runCv} disabled={cv.running}>
                    <IconPlay className="h-4 w-4" />
                    {cv.running ? 'Validando…' : 'Validar'}
                  </button>
                </div>
                {cv.error && (
                  <p className="mt-2 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{cv.error}</p>
                )}
                {cv.result && (
                  <div className="mt-3">
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <dt className="text-ink-faint">Correlación r</dt>
                        <dd className="font-mono text-accent-glow">{cv.result.mean_r} ± {cv.result.r_sd}</dd>
                      </div>
                      <div>
                        <dt className="text-ink-faint">RMSE</dt>
                        <dd className="font-mono">{cv.result.mean_rmse}</dd>
                      </div>
                      <div>
                        <dt className="text-ink-faint">Exactitud</dt>
                        <dd className="font-mono">{cv.result.mean_accuracy ?? '—'}</dd>
                      </div>
                    </div>
                    <div className="mt-2">
                      <Scatter points={cv.result.per_individual.map((p) => [p.obs, p.pred])}
                        xLabel="observado" yLabel="predicho" title="Predicho vs observado" identity />
                    </div>
                    <div className="mt-2 overflow-x-auto">
                      <table className="table-base min-w-[22rem]">
                        <thead>
                          <tr><th>Pliegue</th><th>Rep.</th><th className="text-right">r</th><th className="text-right">RMSE</th><th className="text-right">Exactitud</th><th className="text-right">n val</th></tr>
                        </thead>
                        <tbody>
                          {cv.result.per_fold.map((f, i) => (
                            <tr key={i}>
                              <td className="font-mono">{f.fold}</td>
                              <td className="font-mono">{f.repeat}</td>
                              <td className="text-right font-mono">{f.r}</td>
                              <td className="text-right font-mono">{f.rmse}</td>
                              <td className="text-right font-mono">{f.accuracy ?? '—'}</td>
                              <td className="text-right font-mono">{f.n_val}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </Card>

              <Card title="GWAS" subtitle="Asociación de marcador único (EMMAX)"
                icon={<IconPulse className="h-5 w-5" />}>
                <div className="flex items-end gap-3">
                  <div>
                    <label className="label">MAF mín.</label>
                    <input type="number" min={0} max={0.5} step={0.01} className="input w-24"
                      value={gwasMaf} onChange={(e) => setGwasMaf(Number(e.target.value))} />
                  </div>
                  <button className="btn-primary" onClick={runGwas} disabled={gwas.running}>
                    <IconPlay className="h-4 w-4" />
                    {gwas.running ? 'Analizando…' : 'Ejecutar GWAS'}
                  </button>
                </div>
                {gwas.error && (
                  <p className="mt-2 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{gwas.error}</p>
                )}
                {gwas.result && (
                  <div className="mt-3">
                    <div className="mb-2 grid grid-cols-3 gap-2 text-xs">
                      <div><dt className="text-ink-faint">Marcadores</dt><dd className="font-mono">{fmt(gwas.result.n_tested)}</dd></div>
                      <div><dt className="text-ink-faint">h² nulo</dt><dd className="font-mono">{gwas.result.heritability}</dd></div>
                      <div><dt className="text-ink-faint">Asociados</dt><dd className="font-mono">{gwas.result.top_hits.length}</dd></div>
                    </div>
                    {gwas.result.manhattan_svg && (
                      <div className="rounded-lg border border-line/30 overflow-x-auto"
                        dangerouslySetInnerHTML={{ __html: gwas.result.manhattan_svg }} />
                    )}
                    {gwas.result.top_hits.length > 0 && (
                      <div className="mt-2 overflow-x-auto">
                        <table className="table-base min-w-[24rem]">
                          <thead>
                            <tr><th>Marcador</th><th className="text-right">MAF</th><th className="text-right">Efecto</th><th className="text-right">p</th><th className="text-right">-log10</th></tr>
                          </thead>
                          <tbody>
                            {gwas.result.top_hits.slice(0, 20).map((m, i) => (
                              <tr key={i}>
                                <td className="font-mono text-accent-glow">{m.index}</td>
                                <td className="text-right font-mono">{m.maf}</td>
                                <td className="text-right font-mono">{m.beta}</td>
                                <td className="text-right font-mono">{m.p.toExponential(2)}</td>
                                <td className="text-right font-mono">{m.neglogp}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </Card>

              <Card title="Parentesco y estructura" subtitle="GRM, PCA poblacional y pares relacionados"
                icon={<IconStack className="h-5 w-5" />}>
                <button className="btn-primary w-full" onClick={runKin} disabled={kin.running}>
                  <IconPlay className="h-4 w-4" />
                  {kin.running ? 'Calculando…' : 'Analizar parentesco'}
                </button>
                {kin.error && (
                  <p className="mt-2 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{kin.error}</p>
                )}
                {kin.result && (
                  <div className="mt-3 space-y-3">
                    {kin.result.heatmap_svg && (
                      <div className="rounded-lg border border-line/30 overflow-x-auto"
                        dangerouslySetInnerHTML={{ __html: kin.result.heatmap_svg }} />
                    )}
                    <Scatter points={kin.result.pca} xLabel="PC1" yLabel="PC2"
                      title="Estructura poblacional (PCA de la GRM)" />
                    {kin.result.top_pairs.length > 0 && (
                      <div>
                        <p className="mb-1 text-xs text-ink-faint">Pares más relacionados</p>
                        <table className="table-base min-w-[18rem]">
                          <thead><tr><th>#</th><th className="text-right">i</th><th className="text-right">j</th><th className="text-right">Parentesco</th></tr></thead>
                          <tbody>
                            {kin.result.top_pairs.slice(0, 10).map((p, i) => (
                              <tr key={i}>
                                <td className="font-mono text-ink-faint">{i + 1}</td>
                                <td className="text-right font-mono">{p.i + 1}</td>
                                <td className="text-right font-mono">{p.j + 1}</td>
                                <td className="text-right font-mono text-accent-glow">{p.value}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </Card>
                </>
              )}
            </>
          )}
        </div>

        <div className="space-y-4 lg:col-span-2">
          {cleanResult && report && (
            <>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <StatCard label="Individuos" value={fmt(report.final_rows)} icon={<IconList className="h-4 w-4" />} />
                <StatCard label="Marcadores" value={fmt(report.final_markers)} icon={<IconStack className="h-4 w-4" />} />
                <StatCard label="Celdas imputadas" value={fmt(report.imputed_cells)} accent icon={<IconPercent className="h-4 w-4" />} />
                <StatCard label="Filas sin fenotipo" value={fmt(report.dropped_rows_no_phenotype)} icon={<IconList className="h-4 w-4" />} />
              </div>

              <Card title="Reporte de limpieza" icon={<IconRefresh className="h-5 w-5" />}
                subtitle={`Fenotipo: ${report.phenotype_column} · imputación: ${report.impute_method}`}>
                <p className="mb-1 text-xs text-ink-faint">Columnas descartadas</p>
                {report.dropped_columns?.length ? (
                  <ul className="mb-3 space-y-1">
                    {report.dropped_columns.map((dc, i) => (
                      <li key={i} className="flex justify-between rounded border border-line/30 bg-panel-2/40 px-3 py-1 text-xs">
                        <span className="font-mono">{dc.name}</span>
                        <span className="text-ink-faint">{dc.reason}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mb-3 text-xs text-ink-faint">Ninguna.</p>
                )}
                <div className="space-y-1">
                  {columns.map((c) => (
                    <div key={c.index} className="flex items-center gap-3">
                      <span className="w-28 truncate font-mono text-xs">{c.name}</span>
                      <Badge tone={TYPE_TONE[c.type] || 'slate'}>{c.type}</Badge>
                      <div className="flex-1">
                        <Bar value={c.missing_pct / Math.max(maxMissing, 1)}
                          color={c.missing_pct > 50 ? 'bg-bad/70' : 'bg-accent/70'} glow />
                      </div>
                      <span className="w-14 text-right font-mono text-[10px] text-ink-faint">
                        {c.missing_pct}%
                      </span>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          )}

          {upload && !cleanResult && (
            <Card title="Vista interactiva" subtitle="Filtra, edita y limpia fila a fila antes de enviar al análisis"
              icon={<IconList className="h-5 w-5" />}
              actions={running ? <Badge tone="warn">limpiando…</Badge> : null}>
              {(upload.records ?? 0) > GRID_MAX_ROWS ? (
                <div className="rounded-lg border border-warn/40 bg-warn/10 px-3 py-3 text-xs text-warn">
                  El archivo tiene {fmt(upload.records)} filas; la vista interactiva se oculta por encima de
                  {' '}{fmt(GRID_MAX_ROWS)} filas. Usa «Limpiar y preparar» con las opciones de la izquierda
                  (incluye la vista previa de impacto).
                </div>
              ) : (
                <FiltableGrid uploadId={upload.upload_id} onCleaned={handleGridCleaned} />
              )}
            </Card>
          )}

          {!preview && !cleanResult && (
            <Card>
              <p className="text-sm text-ink-faint">
                Sube o registra un <span className="font-semibold text-accent-glow">.csv / .tsv / .xlsx</span> sucio
                (cabeceras raras, decimales con coma, celdas vacías, columnas de texto…). Genoly lo perfilará,
                te dejará elegir la columna de fenotipo y los criterios, y lo limpiará/imputará para enviarlo a
                Modelos mixtos o GBLUP.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
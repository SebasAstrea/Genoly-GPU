import { useRef, useState } from 'react'
import { api } from '../api.js'
import { Card, StatCard, Badge, PageHeader } from '../components/ui.jsx'
import {
  IconDna,
  IconList,
  IconPlay,
  IconServer,
  IconStack,
  IconScissors,
  IconFlask,
  IconTarget,
  IconPercent,
} from '../components/icons.jsx'

const CHUNK_SIZE = 32 * 1024 * 1024
const CHUNK_THRESHOLD = 1024 * 1024 * 1024

function fmt(n) {
  return n?.toLocaleString?.() ?? n
}

// Sube un archivo (multipart o por partes si es > 1 GB) y devuelve su upload_id
async function uploadFile(file, onProgress) {
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
      onProgress?.(offset / file.size)
    }
    return api.uploadChunkedComplete(uploadId, file.size)
  }
  return api.upload(file)
}

function UploadRow({ label, accept, onUploaded, color }) {
  const ref = useRef(null)
  const [info, setInfo] = useState(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(null)

  const poll = async (uploadId) => {
    const st = await api.uploadStats(uploadId)
    if (st.stats_status === 'pending') {
      setTimeout(() => poll(uploadId), 2000)
      return
    }
    setInfo(st)
    onUploaded?.(st)
  }

  const handle = async (file) => {
    if (!file) return
    setBusy(true)
    setProgress(null)
    try {
      const res = await uploadFile(file, setProgress)
      setInfo(res)
      onUploaded?.(res)
      if (res.stats_status === 'pending') poll(res.upload_id)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="label">{label}</span>
        {info && <Badge tone={color}>{info.stats_status === 'ready' ? 'listo' : '…'}</Badge>}
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          className={`btn ${color === 'ok' ? 'btn-ghost' : 'btn-primary'}`}
          onClick={() => ref.current?.click()}
          disabled={busy}
        >
          {busy ? 'Subiendo…' : 'Cargar'}
        </button>
        <span className="truncate font-mono text-xs text-ink-faint">{info?.filename ?? '—'}</span>
      </div>
      <input
        ref={ref}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          handle(e.target.files?.[0])
          e.target.value = ''
        }}
      />
      {progress != null && progress < 1 && (
        <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-panel-2">
          <div className="h-full bg-accent shadow-glow" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
      )}
      {info?.records != null && info.records > 0 && (
        <p className="mt-1 text-[10px] text-ink-faint">
          {fmt(info.records)} registros · {fmt(info.total_bases)} pb
        </p>
      )}
    </div>
  )
}

function ErrorBox({ error }) {
  if (!error) return null
  return (
    <p className="mt-3 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">{error}</p>
  )
}

function NumberField({ label, value, onChange, min, max, step }) {
  return (
    <div>
      <label className="label">{label}</label>
      <input
        type="number"
        className="input w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

const FILTER_TONE = { PASS: 'ok', LowQual: 'warn', LowDepth: 'bad' }
const TYPE_TONE = { SNV: 'accent', DEL: 'warn', INS: 'ok' }

export default function GatK() {
  const [refUpload, setRefUpload] = useState(null)
  const [samUpload, setSamUpload] = useState(null)

  // HaplotypeCaller
  const [params, setParams] = useState({
    min_mapq: 20,
    min_depth: 5,
    min_qual: 10,
    min_af: 0.15,
    max_haplotypes: 32,
    ploidy: 2,
  })
  const [calling, setCalling] = useState(false)
  const [callProgress, setCallProgress] = useState(null)
  const [result, setResult] = useState(null)
  const [callError, setCallError] = useState(null)

  // MarkDuplicates
  const [dupSam, setDupSam] = useState(null)
  const [dupRunning, setDupRunning] = useState(false)
  const [dupResult, setDupResult] = useState(null)
  const [dupError, setDupError] = useState(null)

  // BQSR
  const [bqsrRunning, setBqsrRunning] = useState(false)
  const [bqsrProgress, setBqsrProgress] = useState(null)
  const [bqsrResult, setBqsrResult] = useState(null)
  const [bqsrError, setBqsrError] = useState(null)

  const runHaplotype = async () => {
    if (!refUpload || !samUpload) return
    setCalling(true)
    setCallError(null)
    setResult(null)
    try {
      const job = await api.gatkCallFile({
        ref_upload_id: refUpload.upload_id,
        sam_upload_id: samUpload.upload_id,
        ...params,
      })
      const res = await api.jobEvents(job.job_id, {
        onProgress: (p) => setCallProgress(p),
      })
      setResult(res)
    } catch (e) {
      setCallError(e.message)
    } finally {
      setCalling(false)
      setCallProgress(null)
    }
  }

  const runMarkdups = async () => {
    const sam = dupSam || samUpload
    if (!sam) return
    setDupRunning(true)
    setDupError(null)
    setDupResult(null)
    try {
      const job = await api.gatkMarkdups({ sam_upload_id: sam.upload_id })
      const res = await api.jobEvents(job.job_id)
      setDupResult(res)
    } catch (e) {
      setDupError(e.message)
    } finally {
      setDupRunning(false)
    }
  }

  const runBqsr = async () => {
    if (!refUpload || !samUpload) return
    setBqsrRunning(true)
    setBqsrError(null)
    setBqsrResult(null)
    try {
      const job = await api.gatkBqsr({
        ref_upload_id: refUpload.upload_id,
        sam_upload_id: samUpload.upload_id,
      })
      const res = await api.jobEvents(job.job_id, {
        onProgress: (p) => setBqsrProgress(p),
      })
      setBqsrResult(res)
    } catch (e) {
      setBqsrError(e.message)
    } finally {
      setBqsrRunning(false)
      setBqsrProgress(null)
    }
  }

  return (
    <div className="space-y-8">
      <PageHeader
        index="03 · Variantes GATK"
        title="HaplotypeCaller"
        subtitle="Algoritmos GATK (regiones activas, ensamblaje local, PairHMM, BQSR, MarkDuplicates) acelerados por GPU"
        icon={<IconDna className="h-6 w-6" />}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="space-y-4">
          <Card title="Entrada" subtitle="Referencia FASTA y lecturas SAM/BAM mapeadas" icon={<IconServer className="h-5 w-5" />}>
            <div className="space-y-4">
              <UploadRow label="Referencia (.fasta/.fna)" accept=".fasta,.fa,.fna,.txt" color="accent" onUploaded={setRefUpload} />
              <UploadRow label="Lecturas mapeadas (.sam)" accept=".sam,.bam,.txt" color="ok" onUploaded={setSamUpload} />
            </div>
            <p className="mt-3 text-[10px] text-ink-faint">
              El SAM puede venir de la página Mapeo u otro pipeline (BWA, etc.). Las lecturas se
              procesan en un proceso aislado; la llamada usa el PairHMM nativo (C) o GPU según el
              dispositivo.
            </p>
          </Card>

          <Card title="Parámetros" icon={<IconTarget className="h-5 w-5" />}>
            <div className="grid grid-cols-2 gap-3">
              <NumberField label="MAPQ mín." min={0} max={60} value={params.min_mapq} onChange={(v) => setParams({ ...params, min_mapq: v })} />
              <NumberField label="Profundidad mín." min={1} value={params.min_depth} onChange={(v) => setParams({ ...params, min_depth: v })} />
              <NumberField label="QUAL mín." min={0} value={params.min_qual} onChange={(v) => setParams({ ...params, min_qual: v })} />
              <NumberField label="AF mín." min={0} max={1} step={0.05} value={params.min_af} onChange={(v) => setParams({ ...params, min_af: v })} />
              <NumberField label="Máx. haplotipos" min={2} max={256} value={params.max_haplotypes} onChange={(v) => setParams({ ...params, max_haplotypes: v })} />
              <NumberField label="Ploidía" min={1} max={4} value={params.ploidy} onChange={(v) => setParams({ ...params, ploidy: v })} />
            </div>
            <button className="btn-primary mt-3 w-full" onClick={runHaplotype} disabled={calling || !refUpload || !samUpload}>
              <IconPlay className="h-4 w-4" />
              {calling ? 'Genotipando…' : 'Llamar variantes (HaplotypeCaller)'}
            </button>
            {calling && callProgress && (
              <p className="mt-2 text-xs text-ink-faint">
                {callProgress.sequence} · región {callProgress.region}/{callProgress.regions} · {fmt(callProgress.variants)} variantes
              </p>
            )}
            <ErrorBox error={callError} />
          </Card>
        </div>

        <div className="space-y-4 lg:col-span-2">
          {result ? (
            <>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
                <StatCard label="Variantes" value={fmt(result.total_variants)} accent icon={<IconStack className="h-4 w-4" />} />
                <StatCard label="SNVs" value={fmt(result.snvs)} icon={<IconDna className="h-4 w-4" />} />
                <StatCard label="Indels" value={fmt(result.indels)} icon={<IconScissors className="h-4 w-4" />} />
                <StatCard label="PASS" value={fmt(result.pass_variants)} icon={<IconPercent className="h-4 w-4" />} />
                <StatCard label="Backend" value={result.pairhmm_backend?.includes('native') ? 'C nativo' : 'GPU/CPU'} icon={<IconFlask className="h-4 w-4" />} />
              </div>

              <Card title="Genotipos (muestra)" subtitle={`${result.sample_name} · PL/GQ/AD al estilo GATK`} icon={<IconList className="h-5 w-5" />}>
                {result.vcf_upload_id && (
                  <a className="btn-ghost mb-3 inline-flex" href={api.downloadUrl(result.vcf_upload_id)} download>
                    Descargar VCF
                  </a>
                )}
                <div className="overflow-x-auto">
                  <table className="table-base min-w-[42rem]">
                    <thead>
                      <tr>
                        <th>Secuencia</th>
                        <th className="text-right">POS</th>
                        <th>REF</th>
                        <th>ALT</th>
                        <th>Tipo</th>
                        <th>GT</th>
                        <th className="text-right">DP</th>
                        <th>AD</th>
                        <th className="text-right">GQ</th>
                        <th className="text-right">QUAL</th>
                        <th>Filtro</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.sample.slice(0, 100).map((v, i) => (
                        <tr key={i}>
                          <td className="font-mono text-ink-faint">{v.sequence}</td>
                          <td className="text-right font-mono">{fmt(v.position)}</td>
                          <td className="font-mono">{v.ref}</td>
                          <td className="font-mono text-accent-glow">{v.alt}</td>
                          <td>
                            <Badge tone={TYPE_TONE[v.type] || 'slate'}>{v.type}</Badge>
                          </td>
                          <td className="font-mono">{v.gt}</td>
                          <td className="text-right font-mono">{v.dp}</td>
                          <td className="font-mono text-ink-faint">{v.ad?.join('/')}</td>
                          <td className="text-right font-mono">{v.gq}</td>
                          <td className="text-right font-mono">{v.qual?.toFixed?.(1) ?? v.qual}</td>
                          <td>
                            <Badge tone={FILTER_TONE[v.filter] || 'slate'}>{v.filter}</Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {result.sample.length === 0 && (
                  <p className="mt-2 text-xs text-ink-faint">Ninguna variante superó los filtros con los parámetros dados.</p>
                )}
              </Card>
            </>
          ) : (
            <Card>
              <p className="text-sm text-ink-faint">
                Sube una <span className="font-semibold text-accent-glow">referencia</span> y las{' '}
                <span className="font-semibold text-accent-glow">lecturas mapeadas</span> (SAM), ajusta
                los umbrales y pulsa <span className="font-semibold text-accent-glow">Llamar variantes</span>.
                El pipeline detecta regiones activas en GPU, ensambla haplotipos locales (de Bruijn),
                calcula likelihoods con PairHMM y genotipa con PL/GQ/AD como HaplotypeCaller de GATK.
              </p>
            </Card>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card title="MarkDuplicates" subtitle="Duplicados de PCR/ópticos por posición 5' (estilo GATK)" icon={<IconScissors className="h-5 w-5" />}>
          <div className="space-y-3">
            <UploadRow label="SAM (opcional si ya está arriba)" accept=".sam,.bam,.txt" color="ok" onUploaded={setDupSam} />
            <button
              className="btn-primary w-full"
              onClick={runMarkdups}
              disabled={dupRunning || !(dupSam || samUpload)}
            >
              <IconPlay className="h-4 w-4" />
              {dupRunning ? 'Marcando…' : 'Marcar duplicados'}
            </button>
            <ErrorBox error={dupError} />
            {dupResult && (
              <>
                <div className="grid grid-cols-3 gap-3">
                  <StatCard label="Lecturas" value={fmt(dupResult.total_reads)} icon={<IconList className="h-4 w-4" />} />
                  <StatCard label="Duplicados" value={fmt(dupResult.duplicates)} accent icon={<IconScissors className="h-4 w-4" />} />
                  <StatCard label="Tasa" value={`${(dupResult.duplicate_rate * 100).toFixed(1)}%`} icon={<IconPercent className="h-4 w-4" />} />
                </div>
                {dupResult.sam_upload_id && (
                  <a className="btn-ghost inline-flex" href={api.downloadUrl(dupResult.sam_upload_id)} download>
                    Descargar SAM con flags 0x400
                  </a>
                )}
              </>
            )}
          </div>
        </Card>

        <Card title="BQSR" subtitle="Recalibración de calidades de base (BaseRecalibrator + ApplyBQSR)" icon={<IconFlask className="h-5 w-5" />}>
          <div className="space-y-3">
            <p className="text-[10px] text-ink-faint">
              Usa la referencia y el SAM de la columna de Entrada. Estima la tasa de error por
              covariable (calidad × ciclo × hebra) y recalibra las lecturas en el SAM de salida.
            </p>
            <button className="btn-primary w-full" onClick={runBqsr} disabled={bqsrRunning || !refUpload || !samUpload}>
              <IconPlay className="h-4 w-4" />
              {bqsrRunning ? 'Recalibrando…' : 'Recalibrar calidades (BQSR)'}
            </button>
            {bqsrRunning && bqsrProgress && (
              <p className="text-xs text-ink-faint">
                {bqsrProgress.stage === 'bqsr_estimate'
                  ? `Estimando covariables… ${fmt(bqsrProgress.bases)} bases`
                  : 'Aplicando recalibración…'}
              </p>
            )}
            <ErrorBox error={bqsrError} />
            {bqsrResult && (
              <>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <StatCard label="Bases" value={fmt(bqsrResult.bases)} icon={<IconList className="h-4 w-4" />} />
                  <StatCard label="Errores" value={fmt(bqsrResult.errors)} icon={<IconScissors className="h-4 w-4" />} />
                  <StatCard label="Tasa global" value={`${(bqsrResult.global_rate * 100).toFixed(3)}%`} accent icon={<IconPercent className="h-4 w-4" />} />
                  <StatCard label="Δ medio" value={bqsrResult.mean_delta} icon={<IconFlask className="h-4 w-4" />} />
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {bqsrResult.table_upload_id && (
                    <a className="btn-ghost" href={api.downloadUrl(bqsrResult.table_upload_id)} download>
                      Descargar tabla JSON
                    </a>
                  )}
                  {bqsrResult.sam_upload_id && (
                    <a className="btn-ghost" href={api.downloadUrl(bqsrResult.sam_upload_id)} download>
                      Descargar SAM recalibrado
                    </a>
                  )}
                </div>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}

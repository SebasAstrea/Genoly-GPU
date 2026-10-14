const BASE = '/api'

async function request(path, options = {}) {
  const isFormData =
    typeof FormData !== 'undefined' && options.body instanceof FormData
  const res = await fetch(`${BASE}${path}`, {
    headers: isFormData ? {} : { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    let detail = await res.text()
    try {
      const parsed = JSON.parse(detail)
      if (parsed && typeof parsed.detail === 'string') detail = parsed.detail
    } catch {
      // cuerpo no JSON: se muestra tal cual
    }
    throw new Error(detail || `Error ${res.status}`)
  }
  return res.json()
}

// fetch para cuerpos binarios (chunks de subida): sin cabecera JSON.
async function rawFetch(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    let detail = await res.text()
    try {
      const parsed = JSON.parse(detail)
      if (parsed && typeof parsed.detail === 'string') detail = parsed.detail
    } catch {
      // cuerpo no JSON: se muestra tal cual
    }
    throw new Error(detail || `Error ${res.status}`)
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),
  setup: () => request('/setup'),
  device: () => request('/device'),

  upload: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/upload', { method: 'POST', body: form })
  },

  uploadStats: (uploadId) => request(`/upload/${uploadId}`),

  registerUpload: (path) =>
    request('/upload/register', { method: 'POST', body: JSON.stringify({ path }) }),

  uploadChunkedInit: (filename, size) =>
    request('/upload/chunked', { method: 'POST', body: JSON.stringify({ filename, size }) }),

  uploadChunkedGet: (uploadId) => request(`/upload/chunked/${uploadId}`),

  uploadChunkedPut: (uploadId, offset, blob) =>
    rawFetch(`/upload/chunked/${uploadId}?offset=${offset}`, {
      method: 'PUT',
      body: blob,
    }),

  uploadChunkedComplete: (uploadId, size) =>
    request(`/upload/chunked/${uploadId}/complete`, {
      method: 'POST',
      body: JSON.stringify({ size }),
    }),

  registerDataset: (path) =>
    request('/dataset/register', { method: 'POST', body: JSON.stringify({ path }) }),

  datasetStatus: (datasetId) => request(`/dataset/${datasetId}`),

  analyzeQc: (payload) =>
    request('/qc/analyze', { method: 'POST', body: JSON.stringify(payload) }),

  fastqQc: (payload) =>
    request('/qc/fastq', { method: 'POST', body: JSON.stringify(payload) }),

  fastqProcess: (payload) =>
    request('/qc/fastq/process', { method: 'POST', body: JSON.stringify(payload) }),

  mapReads: (payload) =>
    request('/map', { method: 'POST', body: JSON.stringify(payload) }),

  callVariantsFile: (payload) =>
    request('/variants/call-file', { method: 'POST', body: JSON.stringify(payload) }),

  gatkCallFile: (payload) =>
    request('/gatk/call-file', { method: 'POST', body: JSON.stringify(payload) }),

  gatkBqsr: (payload) =>
    request('/gatk/bqsr', { method: 'POST', body: JSON.stringify(payload) }),

  gatkMarkdups: (payload) =>
    request('/gatk/markdups', { method: 'POST', body: JSON.stringify(payload) }),

  runDownstream: (payload) =>
    request('/downstream', { method: 'POST', body: JSON.stringify(payload) }),

  annotateVariants: (payload) =>
    request('/annotation', { method: 'POST', body: JSON.stringify(payload) }),

  buildReport: (payload) =>
    request('/report', { method: 'POST', body: JSON.stringify(payload) }),

  qdataPreview: (uploadId) => request(`/qdata/preview`, {
    method: 'POST',
    body: JSON.stringify({ upload_id: uploadId }),
  }),

  qdataClean: (payload) =>
    request('/qdata/clean', { method: 'POST', body: JSON.stringify(payload) }),

  qdataGet: (cleanId) => request(`/qdata/${cleanId}`),

  qdataPreviewClean: (payload) =>
    request('/qdata/preview-clean', { method: 'POST', body: JSON.stringify(payload) }),

  qdataGridPage: (uploadId, { offset = 0, limit = 500, filters = {} } = {}) =>
    request(`/qdata/${encodeURIComponent(uploadId)}/grid-page`, {
      method: 'POST',
      body: JSON.stringify({ upload_id: uploadId, offset, limit, filters }),
    }),

  qdataColumnValues: (uploadId, col, filters = {}) =>
    request(`/qdata/${encodeURIComponent(uploadId)}/column-values`, {
      method: 'POST',
      body: JSON.stringify({ upload_id: uploadId, col, filters }),
    }),

  qdataApplyFilters: (payload) =>
    request('/qdata/apply-filters', { method: 'POST', body: JSON.stringify(payload) }),
  qdataCrossval: (payload) =>
    request('/qdata/crossval', { method: 'POST', body: JSON.stringify(payload) }),

  qdataGwas: (payload) =>
    request('/qdata/gwas', { method: 'POST', body: JSON.stringify(payload) }),

  qdataKinship: (payload) =>
    request('/qdata/kinship', { method: 'POST', body: JSON.stringify(payload) }),

  downloadUrl: (uploadId) => `${BASE}/upload/${uploadId}/download`,

  countKmers: (payload) =>
    request('/kmer/count', { method: 'POST', body: JSON.stringify(payload) }),

  countKmersAsync: (payload) =>
    request('/kmer/count-async', { method: 'POST', body: JSON.stringify(payload) }),


  // Suscripción SSE al progreso de un trabajo. Resuelve con el resultado
  // en el evento 'done' y rechaza con el detalle en 'error'.
  jobEvents: (jobId, { onProgress, onStart } = {}) =>
    new Promise((resolve, reject) => {
      const es = new EventSource(`${BASE}/jobs/${jobId}/events`)
      es.onmessage = (e) => {
        let data
        try {
          data = JSON.parse(e.data)
        } catch {
          return
        }
        if (data.type === 'progress') onProgress?.(data)
        else if (data.type === 'start') onStart?.(data)
        else if (data.type === 'done') {
          es.close()
          resolve(data.result)
        } else if (data.type === 'error') {
          es.close()
          reject(new Error(data.detail || 'El trabajo falló'))
        }
      }
      es.onerror = () => {
        es.close()
        reject(new Error('Conexión de progreso interrumpida'))
      }
    }),

  callVariants: (payload) =>
    request('/variants/call', { method: 'POST', body: JSON.stringify(payload) }),

  fitLmm: (payload) =>
    request('/quantitative/fit', { method: 'POST', body: JSON.stringify(payload) }),

  predictGblup: (payload) =>
    request('/gblup/predict', { method: 'POST', body: JSON.stringify(payload) }),

  predictPblup: (payload) =>
    request('/pblup/predict', { method: 'POST', body: JSON.stringify(payload) }),
}
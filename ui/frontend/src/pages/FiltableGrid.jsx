import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { api } from '../api.js'

// Tabla interactiva tipo Excel para datos de genética cuantitativa.
// Carga todas las filas del CSV/Excel, filtra por columna (valores únicos
// y vacíos/NaN/"-"), edita celdas con doble clic y envía lo filtrado.

const MISSING_TOKENS = new Set(['', 'na', 'nan', 'null', 'none', '-', 'nd'])
const MAX_RENDER = 3000

function isMissing(v) {
  if (v === null || v === undefined) return true
  return MISSING_TOKENS.has(String(v).trim().toLowerCase())
}

// Popover de filtro por columna (valores únicos + vacíos).
function FilterPopover({ col, uniqueVals, hasMissing, filter, onChange, onClose }) {
  const boxRef = useRef(null)
  const [vals, setVals] = useState(() => new Set(filter ? filter.vals : uniqueVals))
  const [vacios, setVacios] = useState(() => (filter ? filter.vacios : true))

  useEffect(() => {
    const onDown = (e) => {
      if (boxRef.current && !boxRef.current.contains(e.target)) onClose()
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [onClose])

  const toggle = (v) => {
    const next = new Set(vals)
    if (next.has(v)) next.delete(v)
    else next.add(v)
    setVals(next)
    onChange(col, next, vacios)
  }

  const setAll = (on) => {
    const next = on ? new Set(uniqueVals) : new Set()
    setVals(next)
    onChange(col, next, vacios)
  }

  return (
    <div ref={boxRef} className="absolute right-0 top-full z-40 mt-1 w-56 rounded-lg border border-line/40 bg-panel shadow-card">
      <div className="flex items-center justify-between border-b border-line/25 px-3 py-1.5 text-[11px] font-semibold text-ink-dim">
        <span className="truncate">Valores</span>
        <span className="flex gap-2">
          <button type="button" className="text-accent hover:underline" onClick={() => setAll(true)}>Todos</button>
          <button type="button" className="text-accent hover:underline" onClick={() => setAll(false)}>Ninguno</button>
        </span>
      </div>

      <ul className="max-h-52 overflow-y-auto">
        {uniqueVals.slice(0, 300).map((v) => (
          <li key={v}>
            <label className="flex cursor-pointer items-center gap-2 px-3 py-1 text-xs hover:bg-line/10">
              <input type="checkbox" checked={vals.has(v)} onChange={() => toggle(v)} className="accent-accent" />
              <span className="truncate font-mono">{v}</span>
            </label>
          </li>
        ))}
        {uniqueVals.length === 0 && (
          <li className="px-3 py-2 text-xs text-ink-faint">Sin valores distintos</li>
        )}
        {uniqueVals.length > 300 && (
          <li className="px-3 py-1 text-[10px] text-ink-faint">…{uniqueVals.length - 300} valores más</li>
        )}
      </ul>

      {hasMissing && (
        <label className="flex cursor-pointer items-center gap-2 border-t border-line/25 px-3 py-1.5 text-xs hover:bg-line/10">
          <input
            type="checkbox"
            checked={vacios}
            onChange={(e) => { setVacios(e.target.checked); onChange(col, vals, e.target.checked) }}
            className="accent-accent"
          />
          Mostrar vacíos / NaN / “-”
        </label>
      )}

      <div className="flex items-center justify-between border-t border-line/25 px-3 py-1.5 text-[10px] text-ink-faint">
        <span>{vals.size} de {uniqueVals.length}</span>
        <button type="button" className="text-accent hover:underline" onClick={onClose}>Cerrar</button>
      </div>
    </div>
  )
}

export default function FiltableGrid({ uploadId, onCleaned }) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [headerRow, setHeaderRow] = useState(null)
  const [columnNames, setColumnNames] = useState([])
  const [columnTypes, setColumnTypes] = useState([])
  const [rows, setRows] = useState([])

  const [filters, setFilters] = useState({}) // { [col]: { vals: Set, vacios: bool } }
  const [openCol, setOpenCol] = useState(null)
  const [edit, setEdit] = useState(null) // { r, c } índices globales
  const [editText, setEditText] = useState('')
  const [sending, setSending] = useState(false)

  const uvRef = useRef({}) // { [col]: string[] } valores únicos por columna
  const missRef = useRef({}) // { [col]: bool } ¿hay vacíos/NaN en la columna?

  // ── Carga completa de la grid ─────────────────────────────
  useEffect(() => {
    if (!uploadId) return
    let alive = true
    ;(async () => {
      try {
        setLoading(true)
        setError('')
        const data = await api.qdataGridFull(uploadId, null)
        if (!alive) return
        let body = data.grid || []
        let header = null
        if (data.header_detected && body.length) {
          header = body[0]
          body = body.slice(1)
        }
        setHeaderRow(header)
        setColumnNames(data.column_names || [])
        setColumnTypes((data.columns || []).map((c) => c.type))
        setRows(body)

        const n = Math.max(body[0]?.length || 0, (data.column_names || []).length)
        const uv = {}
        const miss = {}
        for (let j = 0; j < n; j++) {
          const s = new Set()
          let hasMiss = false
          for (const row of body) {
            const v = row[j]
            if (v === undefined || v === null || isMissing(v)) hasMiss = true
            else s.add(String(v))
          }
          uv[j] = [...s].sort()
          miss[j] = hasMiss
        }
        uvRef.current = uv
        missRef.current = miss
        setFilters({})
        setOpenCol(null)
        setEdit(null)
        setLoading(false)
      } catch (e) {
        if (!alive) return
        setError(e.message || 'No se pudo cargar el archivo')
        setLoading(false)
      }
    })()
    return () => { alive = false }
  }, [uploadId])

  const nCols = useMemo(() => {
    let n = columnNames.length
    for (const r of rows) if (r.length > n) n = r.length
    return n
  }, [rows, columnNames])

  // ── Filas visibles según filtros ──────────────────────────
  const visible = useMemo(() => {
    const keys = Object.keys(filters)
    if (!keys.length) return rows.map((_, i) => i)
    const out = []
    outer:
    for (let i = 0; i < rows.length; i++) {
      for (const k of keys) {
        const c = Number(k)
        const f = filters[k]
        const raw = c < rows[i].length ? rows[i][c] : null
        if (raw === null || raw === undefined || isMissing(raw)) {
          if (!f.vacios) continue outer
        } else if (!f.vals.has(String(raw))) {
          continue outer
        }
      }
      out.push(i)
    }
    return out
  }, [rows, filters])

  const activeFilters = Object.keys(filters).length

  // ── Filtros ───────────────────────────────────────────────
  const applyFilter = useCallback((col, valsSet, vacios) => {
    const uv = uvRef.current[col] || []
    const allChecked = valsSet.size === uv.length && uv.every((v) => valsSet.has(v))
    setFilters((prev) => {
      const next = { ...prev }
      if (allChecked && vacios) delete next[col]
      else next[col] = { vals: new Set(valsSet), vacios }
      return next
    })
  }, [])

  const clearFilters = () => { setFilters({}); setOpenCol(null) }

  // ── Edición de celdas ─────────────────────────────────────
  const saveEdit = () => {
    if (!edit) return
    const { r, c } = edit
    const value = editText
    const next = rows.map((row, i) => {
      if (i !== r) return row
      const copy = row.slice()
      while (copy.length <= c) copy.push('')
      copy[c] = value
      return copy
    })
    setRows(next)

    // recalcular valores únicos de la columna editada
    const s = new Set()
    let hasMiss = false
    for (const row of next) {
      const v = row[c]
      if (v === undefined || v === null || isMissing(v)) hasMiss = true
      else s.add(String(v))
    }
    uvRef.current[c] = [...s].sort()
    missRef.current[c] = hasMiss
    setFilters((prev) => {
      if (!prev[c]) return prev
      const f = prev[c]
      const nv = new Set([...f.vals].filter((v) => s.has(v)))
      const nextF = { ...prev }
      if (nv.size === s.size && f.vacios) delete nextF[c]
      else nextF[c] = { ...f, vals: nv }
      return nextF
    })
    setEdit(null)
  }

  // ── Enviar datos filtrados al pipeline ────────────────────
  const send = async () => {
    if (!onCleaned || !visible.length || sending) return
    setSending(true)
    try {
      const dataRows = visible.map((i) => rows[i])
      const payload = headerRow ? [headerRow, ...dataRows] : dataRows
      await onCleaned(payload)
    } finally {
      setSending(false)
    }
  }

  // ── Render ────────────────────────────────────────────────
  if (loading && !rows.length && !error) {
    return (
      <div className="flex h-64 items-center justify-center">
        <p className="text-sm text-ink-faint">Cargando tabla…</p>
      </div>
    )
  }

  if (error && !rows.length) {
    return (
      <p className="rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">
        {error}
      </p>
    )
  }

  const names = Array.from({ length: nCols }, (_, j) => columnNames[j] || `col_${j + 1}`)

  return (
    <div className="space-y-3">
      {/* Barra de estado y acciones */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="font-mono text-ink-dim">
          {visible.length.toLocaleString()} de {rows.length.toLocaleString()} filas visibles
          {activeFilters > 0 && (
            <span className="ml-2 rounded-full bg-warn/20 px-2 py-px text-[10px] font-bold uppercase tracking-wide text-warn">
              {activeFilters} filtro{activeFilters !== 1 ? 's' : ''}
            </span>
          )}
        </span>
        <div className="flex gap-2">
          {activeFilters > 0 && (
            <button type="button" className="btn-ghost h-auto px-3 py-1 text-xs font-normal" onClick={clearFilters}>
              Limpiar filtros
            </button>
          )}
          <button
            type="button"
            className="btn-primary h-auto px-3 py-1 text-xs"
            onClick={send}
            disabled={sending || visible.length === 0}
          >
            {sending ? 'Enviando…' : '✓ Aplicar y continuar'}
          </button>
        </div>
      </div>

      {/* Tabla */}
      <div className="overflow-auto rounded-lg border border-line/40" style={{ maxHeight: '65vh' }}>
        <table className="w-full border-collapse font-mono text-xs">
          <thead>
            <tr>
              <th className="sticky top-0 z-20 border-b border-r border-line/40 bg-panel-2 px-2 py-2 text-right text-[10px] font-bold text-ink-faint">
                #
              </th>
              {names.map((name, j) => (
                <th key={j} className="sticky top-0 z-20 border-b border-r border-line/40 bg-panel-2 px-2 py-2 text-left text-[11px] font-semibold text-ink">
                  <span className="flex items-center justify-between gap-1">
                    <span className="truncate max-w-[12rem]" title={name}>{name}</span>
                    <button
                      type="button"
                      title="Filtrar valores"
                      onClick={() => setOpenCol(openCol === j ? null : j)}
                      className={`shrink-0 rounded px-1 text-[9px] leading-none transition-colors ${
                        filters[j] ? 'bg-accent/20 text-accent-glow' : 'text-ink-faint hover:bg-line/20 hover:text-ink'
                      }`}
                    >
                      ▼
                    </button>
                  </span>
                  {openCol === j && (
                    <FilterPopover
                      col={j}
                      uniqueVals={uvRef.current[j] || []}
                      hasMissing={!!missRef.current[j]}
                      filter={filters[j]}
                      onChange={applyFilter}
                      onClose={() => setOpenCol(null)}
                    />
                  )}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {visible.slice(0, MAX_RENDER).map((gi) => (
              <tr key={gi} className="hover:bg-accent/5">
                <td className="border-b border-r border-line/20 bg-panel-2/40 px-2 py-1 text-right text-[10px] text-ink-faint select-none">
                  {gi + (headerRow ? 2 : 1)}
                </td>
                {names.map((_, c) => {
                  const raw = rows[gi]?.[c]
                  const isEd = edit && edit.r === gi && edit.c === c
                  const num = columnTypes[c] === 'numeric'
                  const missing = raw === undefined || raw === null || isMissing(raw)
                  const text = raw === undefined || raw === null ? '' : String(raw)
                  return (
                    <td
                      key={c}
                      onDoubleClick={() => { setEdit({ r: gi, c }); setEditText(text) }}
                      className={`cursor-text border-b border-r border-line/15 px-2 py-1 hover:bg-accent/10 ${
                        num ? 'text-right' : ''
                      } ${missing ? (text === '' ? 'bg-warn/15' : 'text-ink-faint') : 'text-ink'}`}
                    >
                      {isEd ? (
                        <input
                          autoFocus
                          value={editText}
                          onChange={(e) => setEditText(e.target.value)}
                          onBlur={saveEdit}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') saveEdit()
                            else if (e.key === 'Escape') setEdit(null)
                            e.stopPropagation()
                          }}
                          className={`w-full bg-transparent outline-none ${num ? 'text-right' : ''}`}
                          style={{ color: 'rgb(var(--ink))' }}
                        />
                      ) : (
                        text
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
            {visible.length === 0 && (
              <tr>
                <td colSpan={nCols + 1} className="px-4 py-10 text-center text-sm text-warn">
                  Ninguna fila pasa los filtros activos.
                </td>
              </tr>
            )}
          </tbody>
        </table>

        {visible.length > MAX_RENDER && (
          <p className="border-t border-line/25 bg-panel-2/40 px-3 py-2 text-[11px] text-ink-faint">
            Mostrando las primeras {MAX_RENDER.toLocaleString()} de {visible.length.toLocaleString()} filas visibles.
            Usa los filtros para reducir el conjunto.
          </p>
        )}
      </div>
    </div>
  )
}

export { FiltableGrid }

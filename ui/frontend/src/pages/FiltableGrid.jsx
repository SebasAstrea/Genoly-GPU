import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { api } from '../api.js'

// Tabla interactiva tipo Excel para datos de genética cuantitativa.
// Paginación server-side (~500 filas/página), filtros por definición
// {col:{vals,vacios}} y ediciones acumuladas que se envían al servidor.

const MISSING_TOKENS = new Set(['', 'na', 'nan', 'null', 'none', '-', 'nd'])
const PAGE_SIZE = 500

function isMissing(v) {
  if (v === null || v === undefined) return true
  return MISSING_TOKENS.has(String(v).trim().toLowerCase())
}

// Serializa los filtros (Set→array) para mandarlos al backend.
function serializeFilters(filters) {
  const out = {}
  for (const [k, f] of Object.entries(filters)) {
    out[k] = { vals: [...f.vals], vacios: f.vacios }
  }
  return out
}

// Popover de filtro por columna (valores únicos + vacíos).
function FilterPopover({ col, uniqueVals, hasMissing, filter, onChange, onClose, loading }) {
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

      {loading ? (
        <p className="px-3 py-3 text-xs text-ink-faint">Cargando valores…</p>
      ) : (
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
      )}

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
  const [columnNames, setColumnNames] = useState([])
  const [columnTypes, setColumnTypes] = useState([])
  const [rows, setRows] = useState([]) // [{index, cells}]
  const [totalMatching, setTotalMatching] = useState(0)
  const [rowsTotal, setRowsTotal] = useState(0)
  const [headerDetected, setHeaderDetected] = useState(false)
  const [offset, setOffset] = useState(0)
  const [loadingPage, setLoadingPage] = useState(false)

  const [filters, setFilters] = useState({}) // { [col]: { vals: Set, vacios: bool } }
  const [openCol, setOpenCol] = useState(null)
  const [uvVals, setUvVals] = useState([]) // valores únicos de la columna abierta
  const [uvLoading, setUvLoading] = useState(false)
  const [uvHasMissing, setUvHasMissing] = useState(false)
  const [edits, setEdits] = useState(new Map()) // "r,c" -> valor editado
  const [edit, setEdit] = useState(null) // { r, c } durante la edición activa
  const [editText, setEditText] = useState('')
  const [sending, setSending] = useState(false)

  const filtersRef = useRef(filters)
  filtersRef.current = filters

  // ── Carga de página ───────────────────────────────────────
  const loadPage = useCallback(async (off, filt) => {
    if (!uploadId) return
    let alive = true
    setLoadingPage(true)
    try {
      const data = await api.qdataGridPage(uploadId, {
        offset: off,
        limit: PAGE_SIZE,
        filters: serializeFilters(filt),
      })
      if (!alive) return
      setRows(data.rows || [])
      setTotalMatching(data.total_matching ?? 0)
      setRowsTotal(data.rows_total ?? 0)
      setHeaderDetected(!!data.header_detected)
      if (!columnNames.length) {
        setColumnNames(data.column_names || [])
        setColumnTypes(data.column_types || [])
      }
      setError('')
    } catch (e) {
      if (!alive) return
      setError(e.message || 'No se pudo cargar la página')
    } finally {
      if (alive) setLoadingPage(false)
    }
    return () => { alive = false }
  }, [uploadId, columnNames.length])

  // Carga inicial
  useEffect(() => {
    if (!uploadId) return
    setLoading(true)
    loadPage(0, {}).finally(() => setLoading(false))
  }, [uploadId, loadPage])

  // Recargar página cuando cambian filtros u offset
  const reloadKey = useMemo(
    () => [offset, serializeFilters(filters)],
    [offset, filters],
  )
  useEffect(() => {
    if (!uploadId) return
    loadPage(offset, filtersRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadKey, uploadId])

  const nCols = Math.max(columnNames.length, ...rows.map((r) => r.cells.length), 0)

  const activeFilters = Object.keys(filters).length

  // ── Filtros ───────────────────────────────────────────────
  const applyFilter = useCallback((col, valsSet, vacios) => {
    setFilters((prev) => {
      // Si todos los valores están marcados y vacíos=True, no hay filtro activo
      const uvLen = uvVals.length
      const allChecked = valsSet.size === uvLen && uvLen > 0
      const next = { ...prev }
      if (allChecked && vacios) delete next[col]
      else next[col] = { vals: new Set(valsSet), vacios }
      return next
    })
  }, [uvVals.length])

  const clearFilters = () => { setFilters({}); setOpenCol(null); setOffset(0) }

  // Al abrir el popover, cargar valores únicos vía API
  const openFilter = useCallback(async (col) => {
    if (openCol === col) { setOpenCol(null); return }
    setOpenCol(col)
    setUvLoading(true)
    try {
      const data = await api.qdataColumnValues(uploadId, col, serializeFilters(filtersRef.current))
      setUvVals(data.values || [])
      setUvHasMissing(!!data.has_missing)
    } catch {
      setUvVals([])
    } finally {
      setUvLoading(false)
    }
  }, [openCol, uploadId])

  // ── Edición de celdas ─────────────────────────────────────
  const saveEdit = () => {
    if (!edit) return
    const { r, c } = edit
    const value = editText
    const next = new Map(edits)
    next.set(`${r},${c}`, value)
    setEdits(next)
    setEdit(null)
  }

  // Valor visible de una celda (con ediciones aplicadas)
  const cellValue = (rowIndex, col) => {
    const key = `${rowIndex},${col}`
    if (edits.has(key)) return edits.get(key)
    const row = rows.find((x) => x.index === rowIndex)
    const raw = row && col < row.cells.length ? row.cells[col] : null
    return raw === null || raw === undefined ? '' : String(raw)
  }

  // ── Enviar filtros y ediciones al servidor ────────────────
  const send = async () => {
    if (!onCleaned || sending) return
    setSending(true)
    try {
      const editList = []
      for (const [key, value] of edits) {
        const [r, c] = key.split(',').map(Number)
        editList.push({ row: r, col: c, value })
      }
      await onCleaned({
        filters: serializeFilters(filters),
        edits: editList,
      })
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
  const pageStart = offset
  const pageEnd = offset + rows.length
  const hasNext = pageEnd < totalMatching
  const hasPrev = offset > 0

  return (
    <div className="space-y-3">
      {/* Barra de estado y acciones */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="font-mono text-ink-dim">
          {totalMatching.toLocaleString()} de {rowsTotal.toLocaleString()} filas visibles
          {edits.size > 0 && (
            <span className="ml-2 rounded-full bg-accent/20 px-2 py-px text-[10px] font-bold uppercase tracking-wide text-accent-glow">
              {edits.size} edit
            </span>
          )}
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
            disabled={sending || totalMatching === 0}
          >
            {sending ? 'Enviando…' : '✓ Aplicar y continuar'}
          </button>
        </div>
      </div>

      {/* Paginación */}
      <div className="flex items-center justify-between text-[11px] text-ink-faint">
        <span>
          Mostrando filas {pageStart + 1}–{pageEnd} de {totalMatching.toLocaleString()}
        </span>
        <div className="flex gap-1">
          <button
            type="button"
            className="btn-ghost h-auto px-2 py-0.5 text-[11px] font-normal"
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            disabled={!hasPrev || loadingPage}
          >
            ← Anterior
          </button>
          <button
            type="button"
            className="btn-ghost h-auto px-2 py-0.5 text-[11px] font-normal"
            onClick={() => setOffset(offset + PAGE_SIZE)}
            disabled={!hasNext || loadingPage}
          >
            Siguiente →
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
                      onClick={() => openFilter(j)}
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
                      uniqueVals={uvVals}
                      hasMissing={uvHasMissing}
                      filter={filters[j]}
                      onChange={applyFilter}
                      onClose={() => setOpenCol(null)}
                      loading={uvLoading}
                    />
                  )}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.map((row) => (
              <tr key={row.index} className="hover:bg-accent/5">
                <td className="border-b border-r border-line/20 bg-panel-2/40 px-2 py-1 text-right text-[10px] text-ink-faint select-none">
                  {row.index + (headerDetected ? 2 : 1)}
                </td>
                {names.map((_, c) => {
                  const raw = row.cells?.[c]
                  const isEd = edit && edit.r === row.index && edit.c === c
                  const num = columnTypes[c] === 'numeric'
                  const text = cellValue(row.index, c)
                  const missing = isMissing(text)
                  const wasEdited = edits.has(`${row.index},${c}`)
                  return (
                    <td
                      key={c}
                      onDoubleClick={() => { setEdit({ r: row.index, c }); setEditText(text) }}
                      className={`cursor-text border-b border-r border-line/15 px-2 py-1 hover:bg-accent/10 ${
                        num ? 'text-right' : ''
                      } ${missing ? (text === '' ? 'bg-warn/15' : 'text-ink-faint') : 'text-ink'} ${
                        wasEdited ? 'bg-accent/10 ring-1 ring-inset ring-accent/30' : ''
                      }`}
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
            {rows.length === 0 && (
              <tr>
                <td colSpan={nCols + 1} className="px-4 py-10 text-center text-sm text-warn">
                  Ninguna fila pasa los filtros activos.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export { FiltableGrid }

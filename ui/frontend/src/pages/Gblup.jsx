import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api.js'
import { Card, StatCard, Badge, Bar, PageHeader } from '../components/ui.jsx'
import { useQData } from '../qdata.jsx'
import {
  IconTarget,
  IconList,
  IconCog,
  IconStack,
  IconChart,
  IconPercent,
  IconRuler,
  IconPlay,
} from '../components/icons.jsx'

export default function Gblup() {
  const { qdata } = useQData()
  const [kinshipMethod, setKinshipMethod] = useState('vanraden')
  const [varGenetic, setVarGenetic] = useState('')
  const [varResidual, setVarResidual] = useState('')
  const [maxIter, setMaxIter] = useState(100)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const hasMarkers = (qdata?.genotypes?.[0]?.length ?? 0) > 0
  const individuals = qdata?.pedigree?.individuals ?? null

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const payload = {
        phenotypes: qdata.phenotypes,
        genotypes: qdata.genotypes,
        kinship_method: kinshipMethod,
        max_iter: maxIter,
      }
      if (individuals) payload.individuals = individuals
      if (varGenetic !== '' || varResidual !== '') {
        const g = Number(varGenetic)
        const e = Number(varResidual)
        if (!Number.isFinite(g) || g <= 0) {
          throw new Error('Varianza genética inválida: introduce un número positivo o deja ambas vacías')
        }
        if (!Number.isFinite(e) || e <= 0) {
          throw new Error('Varianza residual inválida: introduce un número positivo o deja ambas vacías')
        }
        payload.genetic_variance = g
        payload.residual_variance = e
      }
      const res = await api.predictGblup(payload)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const blupTop = result
    ? [...result.blup].sort((a, b) => b.value - a.value).slice(0, 15)
    : []
  const meanAcc = result
    ? result.blup.reduce((s, b) => s + b.accuracy, 0) / result.blup.length
    : 0
  const implicitH2 =
    result && result.genetic_variance + result.residual_variance > 0
      ? result.genetic_variance / (result.genetic_variance + result.residual_variance)
      : 0

  return (
    <div className="space-y-8">
      <PageHeader
        index="07 · Análisis"
        title="GBLUP"
        subtitle="Predicción genómica de valores de cría con fiabilidad y precisión"
        icon={<IconTarget className="h-6 w-6" />}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="space-y-4">
          {!qdata && (
            <Card title="Datos requeridos" icon={<IconList className="h-5 w-5" />}>
              <p className="text-sm text-ink-dim">
                GBLUP trabaja sobre la matriz limpia preparada en la página de
                Datos (obligatorio para todo el módulo de genética cuantitativa).
              </p>
              <Link className="btn-primary mt-3 w-full" to="/datos">
                Ir a Datos y limpieza
              </Link>
            </Card>
          )}

          {qdata && !hasMarkers && (
            <Card title="Sin marcadores" icon={<IconTarget className="h-5 w-5" />}>
              <p className="text-sm text-ink-dim">
                La matriz limpia no tiene marcadores genéticos (sólo pedigrí).
                GBLUP necesita dosis alélicas; para pedigrí sin genotipos usa
                PBLUP.
              </p>
              <Link className="btn-primary mt-3 w-full" to="/pblup">
                Ir a PBLUP
              </Link>
            </Card>
          )}

          {qdata && hasMarkers && (
            <>
              <Card title="Datos de entrada" icon={<IconList className="h-5 w-5" />}>
                <div className="flex items-center justify-between">
                  <span className="label">Matriz limpia de la sesión</span>
                  <Badge tone="ok">Datos: {qdata.source}</Badge>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <dt className="text-ink-faint">Individuos</dt>
                    <dd className="font-mono text-accent-glow">{qdata.phenotypes.length}</dd>
                  </div>
                  <div>
                    <dt className="text-ink-faint">Marcadores</dt>
                    <dd className="font-mono text-accent-glow">{qdata.genotypes[0]?.length ?? 0}</dd>
                  </div>
                </div>
                <p className="mt-2 text-xs text-ink-faint">
                  Los datos se cargan desde la limpieza de Datos. Para cambiar
                  opciones de limpieza o filtros, vuelve allí.
                </p>
              </Card>

              <Card title="Parámetros" icon={<IconCog className="h-5 w-5" />}>
                <div>
                  <label className="label">Parentesco</label>
                  <select
                    className="input"
                    value={kinshipMethod}
                    onChange={(e) => setKinshipMethod(e.target.value)}
                  >
                    <option value="vanraden">VanRaden</option>
                    <option value="gcta">GCTA</option>
                  </select>
                </div>

                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div>
                    <label className="label">σ² genética</label>
                    <input
                      type="number"
                      min={0}
                      step="0.01"
                      placeholder="REML"
                      className="input"
                      value={varGenetic}
                      onChange={(e) => setVarGenetic(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="label">σ² residual</label>
                    <input
                      type="number"
                      min={0}
                      step="0.01"
                      placeholder="REML"
                      className="input"
                      value={varResidual}
                      onChange={(e) => setVarResidual(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="label">Máx. iteraciones</label>
                    <input
                      type="number"
                      min={10}
                      max={1000}
                      className="input"
                      value={maxIter}
                      onChange={(e) => setMaxIter(Number(e.target.value))}
                    />
                  </div>
                </div>

                <button className="group btn-primary mt-3 w-full" onClick={run} disabled={loading}>
                  <IconPlay className="h-4 w-4 transition-transform duration-300 group-hover:scale-125" />
                  {loading ? 'Prediciendo…' : 'Predecir valores de cría'}
                </button>
              </Card>
            </>
          )}
        </div>

        <div className="space-y-4 lg:col-span-2">
          {error && (
            <div className="rounded-lg border border-bad/40 bg-bad/10 px-4 py-3 text-sm text-bad">
              {error}
            </div>
          )}

          {result && (
            <>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <StatCard label="Var. genética" value={result.genetic_variance.toFixed(4)} accent icon={<IconStack className="h-4 w-4" />} />
                <StatCard label="Var. residual" value={result.residual_variance.toFixed(4)} icon={<IconChart className="h-4 w-4" />} />
                <StatCard label="h² implícita" value={implicitH2.toFixed(3)} icon={<IconPercent className="h-4 w-4" />} />
                <StatCard label="Precisión media" value={meanAcc.toFixed(3)} icon={<IconRuler className="h-4 w-4" />} />
              </div>

              <Card
                title="Valores de cría genómicos (GEBV)"
                subtitle={`${result.n_individuals} individuos · ${result.n_markers} marcadores · parentesco ${kinshipMethod}`}
                icon={<IconTarget className="h-5 w-5" />}
                actions={
                  <>
                    <Badge tone="accent">
                      {result.variance_source === 'dadas' ? 'Varianzas dadas' : 'REML'}
                    </Badge>
                    {result.converged != null && (
                      <Badge tone={result.converged ? 'ok' : 'warn'}>
                        {result.converged ? 'Convergido' : 'Sin converger'}
                      </Badge>
                    )}
                  </>
                }
              >
                <div className="overflow-x-auto">
                  <table className="table-base min-w-[34rem]">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Individuo</th>
                        <th className="text-right">Valor de cría</th>
                        <th className="text-right">Fiabilidad</th>
                        <th className="text-right">Precisión</th>
                        <th className="w-28">Fiab.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {blupTop.map((b) => (
                        <tr key={b.individual}>
                          <td className="text-ink-faint">
                            {typeof b.individual === 'number' ? b.individual : ''}
                          </td>
                          <td className="font-mono">
                            {typeof b.individual === 'number'
                              ? `IND-${String(b.individual).padStart(4, '0')}`
                              : b.individual}
                          </td>
                          <td className="text-right font-mono text-accent-glow">
                            {b.value.toFixed(4)}
                          </td>
                          <td className="text-right font-mono">{b.reliability.toFixed(3)}</td>
                          <td className="text-right font-mono">{b.accuracy.toFixed(3)}</td>
                          <td>
                            <Bar value={b.reliability} glow />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-xs text-ink-faint">
                  Se muestran los 15 mayores valores de cría predichos, ordenados por mérito genético.
                </p>
              </Card>
            </>
          )}

          {!result && !error && qdata && hasMarkers && (
            <Card title="Ayuda" icon={<IconTarget className="h-5 w-5" />}>
              <p className="text-sm text-ink-faint">
                GBLUP resuelve el modelo animal en un solo paso y calcula la fiabilidad de cada
                valor de cría a partir del error de predicción (PEV): fiabilidad =
                1 − PEV/Var(u), con precisión = √fiabilidad.
              </p>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-ink-dim">
                <li>Si indicas σ² genética y residual (de estudios previos), la solución es directa sin iterar.</li>
                <li>Si los dejas vacíos, las varianzas se estiman por REML automáticamente.</li>
                <li>La fiabilidad mide cuánta información aporta cada individuo al predictor.</li>
              </ul>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

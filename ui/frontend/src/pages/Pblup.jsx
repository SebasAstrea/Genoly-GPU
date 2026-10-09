import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api.js'
import { Card, StatCard, Badge, Bar, PageHeader } from '../components/ui.jsx'
import { useQData } from '../qdata.jsx'
import {
  IconDna,
  IconCog,
  IconStack,
  IconChart,
  IconPercent,
  IconRuler,
  IconPlay,
  IconList,
} from '../components/icons.jsx'

export default function Pblup() {
  const { qdata } = useQData()
  const [varGenetic, setVarGenetic] = useState('')
  const [varResidual, setVarResidual] = useState('')
  const [maxIter, setMaxIter] = useState(100)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const pedigree = qdata?.pedigree
  const hasPedigree =
    pedigree?.individuals?.length &&
    (pedigree.sires?.some((s) => s != null) ||
      pedigree.dams?.some((d) => d != null))

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const payload = {
        phenotypes: qdata.phenotypes,
        individuals: pedigree.individuals,
        sires: pedigree.sires,
        dams: pedigree.dams,
        max_iter: maxIter,
      }
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
      const res = await api.predictPblup(payload)
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

  return (
    <div className="space-y-8">
      <PageHeader
        index="08 · Análisis"
        title="PBLUP"
        subtitle="Predicción de valores de cría por modelo animal (pedigrí)"
        icon={<IconDna className="h-6 w-6" />}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="space-y-4">
          {!qdata && (
            <Card title="Datos requeridos" icon={<IconList className="h-5 w-5" />}>
              <p className="text-sm text-ink-dim">
                PBLUP necesita un dataset con columnas de pedigrí (sire/dam)
                limpio en la página de Datos.
              </p>
              <Link className="btn-primary mt-3 w-full" to="/datos">
                Ir a Datos y limpieza
              </Link>
            </Card>
          )}

          {qdata && !hasPedigree && (
            <Card title="Pedigrí no disponible" icon={<IconDna className="h-5 w-5" />}>
              <p className="text-sm text-ink-dim">
                Los datos limpios no incluyen columnas de pedigrí (sire/dam).
                Vuelve a Datos y selecciona las columnas de padre y madre
                antes de limpiar.
              </p>
              <Link className="btn-primary mt-3 w-full" to="/datos">
                Volver a Datos
              </Link>
            </Card>
          )}

          {qdata && hasPedigree && (
            <>
              <Card title="Parámetros" icon={<IconCog className="h-5 w-5" />}>
                <div className="grid grid-cols-2 gap-3">
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

                <button
                  className="group btn-primary mt-3 w-full"
                  onClick={run}
                  disabled={loading}
                >
                  <IconPlay className="h-4 w-4 transition-transform duration-300 group-hover:scale-125" />
                  {loading ? 'Prediciendo…' : 'Predecir valores de cría'}
                </button>
              </Card>

              <Card title="Pedigrí" subtitle="Resumen del modelo animal"
                icon={<IconDna className="h-5 w-5" />}>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <dt className="text-ink-faint">Individuos</dt>
                    <dd className="font-mono text-accent-glow">
                      {pedigree.individuals.length}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-ink-faint">Datos</dt>
                    <dd className="truncate font-mono" title={qdata.source}>
                      {qdata.source}
                    </dd>
                  </div>
                </div>
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
                <StatCard label="h²" value={result.heritability.toFixed(3)} icon={<IconPercent className="h-4 w-4" />} />
                <StatCard label="Precisión media" value={meanAcc.toFixed(3)} icon={<IconRuler className="h-4 w-4" />} />
              </div>

              <Card
                title="Valores de cría (BLUP animal)"
                subtitle={`${result.n_individuals} individuos · ${result.founders.length} fundadores · ${result.unknown_parents} padres no listados`}
                icon={<IconDna className="h-5 w-5" />}
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
                  <table className="table-base min-w-[38rem]">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Individuo</th>
                        <th className="text-right">Valor de cría</th>
                        <th className="text-right">Fiabilidad</th>
                        <th className="text-right">Precisión</th>
                        <th className="text-right">Endogamia F</th>
                        <th className="w-28">Fiab.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {blupTop.map((b) => (
                        <tr key={b.individual}>
                          <td className="text-ink-faint">{b.index + 1}</td>
                          <td className="font-mono">{b.individual}</td>
                          <td className="text-right font-mono text-accent-glow">
                            {b.value.toFixed(4)}
                          </td>
                          <td className="text-right font-mono">{b.reliability.toFixed(3)}</td>
                          <td className="text-right font-mono">{b.accuracy.toFixed(3)}</td>
                          <td className="text-right font-mono">
                            {b.inbreeding > 0 ? (
                              <span className="text-warn">{b.inbreeding.toFixed(3)}</span>
                            ) : (
                              '0.000'
                            )}
                          </td>
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

          {!result && !error && qdata && hasPedigree && (
            <Card title="Ayuda" icon={<IconDna className="h-5 w-5" />}>
              <p className="text-sm text-ink-faint">
                PBLUP resuelve el modelo animal con la matriz de parentesco A
                construida a partir del pedigrí (sire/dam). Calcula la
                fiabilidad de cada valor de cría a partir del PEV y muestra el
                coeficiente de endogamia F de cada individuo.
              </p>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-ink-dim">
                <li>Si indicas σ² genética y residual, la solución es directa sin iterar.</li>
                <li>Si los dejas vacíos, las varianzas se estiman por REML automáticamente.</li>
                <li>Los padres no listados se consideran desconocidos (fundadores).</li>
              </ul>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

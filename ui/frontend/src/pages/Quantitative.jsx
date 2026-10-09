import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api.js'
import { Card, StatCard, Badge, Bar, PageHeader } from '../components/ui.jsx'
import { useQData } from '../qdata.jsx'
import {
  IconChart,
  IconList,
  IconCog,
  IconPercent,
  IconStack,
  IconPulse,
  IconTarget,
  IconPlay,
} from '../components/icons.jsx'

export default function Quantitative() {
  const { qdata } = useQData()
  const [method, setMethod] = useState('reml')
  const [kinshipMethod, setKinshipMethod] = useState('vanraden')
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
        method,
        kinship_method: kinshipMethod,
        max_iter: maxIter,
      }
      if (individuals) payload.individuals = individuals
      const res = await api.fitLmm(payload)
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
  const blupMin = blupTop.length ? blupTop[blupTop.length - 1].value : 0
  const blupMax = blupTop.length ? blupTop[0].value : 1

  return (
    <div className="space-y-8">
      <PageHeader
        index="06 · Análisis"
        title="Genética cuantitativa"
        subtitle="Modelos lineales mixtos (REML/ML) y valores de cría BLUP sobre GPU"
        icon={<IconChart className="h-6 w-6" />}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="space-y-4">
          {!qdata && (
            <Card title="Datos requeridos" icon={<IconList className="h-5 w-5" />}>
              <p className="text-sm text-ink-dim">
                Los modelos mixtos trabajan sobre la matriz limpia preparada en
                la página de Datos (obligatorio para todo el módulo de
                genética cuantitativa).
              </p>
              <Link className="btn-primary mt-3 w-full" to="/datos">
                Ir a Datos y limpieza
              </Link>
            </Card>
          )}

          {qdata && !hasMarkers && (
            <Card title="Sin marcadores" icon={<IconChart className="h-5 w-5" />}>
              <p className="text-sm text-ink-dim">
                La matriz limpia no tiene marcadores genéticos (sólo pedigrí).
                Los modelos mixtos y GBLUP necesitan dosis alélicas; para
                pedigrí sin genotipos usa PBLUP.
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
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="label">Método</label>
                    <select
                      className="input"
                      value={method}
                      onChange={(e) => setMethod(e.target.value)}
                    >
                      <option value="reml">REML</option>
                      <option value="ml">ML</option>
                    </select>
                  </div>
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
                  {loading ? 'Ajustando…' : 'Ajustar modelo'}
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
                <StatCard label="Heredabilidad h²" value={result.heritability.toFixed(3)} accent icon={<IconPercent className="h-4 w-4" />} />
                <StatCard label="Var. genética" value={result.genetic_variance.toFixed(4)} icon={<IconStack className="h-4 w-4" />} />
                <StatCard label="Var. residual" value={result.residual_variance.toFixed(4)} icon={<IconChart className="h-4 w-4" />} />
                <StatCard label="Log-verosimilitud" value={result.log_likelihood.toFixed(2)} icon={<IconPulse className="h-4 w-4" />} />
              </div>

              <Card
                title="Valores de cría (BLUP)"
                subtitle={`${result.n_individuals} individuos · ${result.n_markers} marcadores · ${result.iterations} iteraciones`}
                icon={<IconTarget className="h-5 w-5" />}
                actions={
                  <Badge tone={result.converged ? 'ok' : 'warn'}>
                    {result.converged ? 'Convergido' : 'Sin converger'}
                  </Badge>
                }
              >
                <div className="overflow-x-auto">
                  <table className="table-base min-w-[28rem]">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Individuo</th>
                        <th className="text-right">Valor de cría</th>
                        <th className="w-40">Efecto</th>
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
                          <td>
                            <Bar
                              value={
                                (b.value - blupMin) /
                                Math.max(blupMax - blupMin, 1e-9)
                              }
                              glow
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-xs text-ink-faint">
                  Se muestran los 15 mayores valores de cría predichos.
                </p>
              </Card>
            </>
          )}

          {!result && !error && qdata && hasMarkers && (
            <Card title="Ayuda" icon={<IconChart className="h-5 w-5" />}>
              <p className="text-sm text-ink-faint">
                El módulo ajusta el modelo animal y = Xβ + Zu + ε, donde u son los valores
                de cría con matriz de parentesco genómica K (VanRaden o GCTA). Las componentes de
                varianza se estiman por máxima verosimilitud restringida (REML) o máxima
                verosimilitud (ML) mediante puntuación de Fisher sobre GPU.
              </p>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-ink-dim">
                <li>Los datos llegan de la limpieza en la página Datos.</li>
                <li>La heredabilidad h² = σ²g / (σ²g + σ²e) se muestra al ajustar.</li>
                <li>El BLUP ordena los individuos por su mérito genético predicho.</li>
              </ul>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

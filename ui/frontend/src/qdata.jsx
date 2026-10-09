import { createContext, useContext, useState } from 'react'

// Matriz limpia compartida entre la página de carga y los análisis
// (Modelos mixtos / GBLUP). Una sola sesión: al limpiar en Datos, las
// pantallas de análisis la consumen.
const QDataContext = createContext(null)

export function QDataProvider({ children }) {
  const [qdata, setQdata] = useState(null)
  return (
    <QDataContext.Provider value={{ qdata, setQdata }}>
      {children}
    </QDataContext.Provider>
  )
}

export function useQData() {
  return useContext(QDataContext)
}
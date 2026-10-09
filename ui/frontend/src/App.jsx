import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import { QDataProvider } from './qdata.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Device from './pages/Device.jsx'
import Qc from './pages/Qc.jsx'
import Fastq from './pages/Fastq.jsx'
import Kmer from './pages/Kmer.jsx'
import Map from './pages/Map.jsx'
import Downstream from './pages/Downstream.jsx'
import Annotation from './pages/Annotation.jsx'
import Report from './pages/Report.jsx'
import Datos from './pages/Datos.jsx'
import Variants from './pages/Variants.jsx'
import Quantitative from './pages/Quantitative.jsx'
import Gblup from './pages/Gblup.jsx'
import Pblup from './pages/Pblup.jsx'

export default function App() {
  return (
    <QDataProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/device" element={<Device />} />
          <Route path="/qc" element={<Qc />} />
          <Route path="/fastq" element={<Fastq />} />
          <Route path="/kmer" element={<Kmer />} />
          <Route path="/map" element={<Map />} />
          <Route path="/downstream" element={<Downstream />} />
          <Route path="/annotation" element={<Annotation />} />
          <Route path="/report" element={<Report />} />
          <Route path="/datos" element={<Datos />} />
          <Route path="/variants" element={<Variants />} />
          <Route path="/quantitative" element={<Quantitative />} />
          <Route path="/gblup" element={<Gblup />} />
          <Route path="/pblup" element={<Pblup />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </QDataProvider>
  )
}
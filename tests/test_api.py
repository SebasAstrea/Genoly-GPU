"""Tests de la API REST (FastAPI): auth, límites de subida, LFI y SSE."""

import asyncio
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from Genoly import FastaRecord, write_fasta
from ui.backend import jobs, maintenance, security
from ui.backend.main import app
from ui.backend.routers import upload as upload_router
from ui.backend.uploads import UPLOAD_DIR, delete_upload

client = TestClient(app)


def _api_routes():
    """(método, path) de toda la superficie /api/* documentada por la app.

    Los routers se montan de forma perezosa en FastAPI >= 0.115, así que
    se lee el esquema OpenAPI en lugar de ``app.routes``.
    """
    spec = app.openapi()
    out = []
    for path, methods in spec["paths"].items():
        for method in methods:
            if method.upper() in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                out.append((method.upper(), path))
    return out


class AuthTest(unittest.TestCase):
    """El middleware exige la API key en /api/* menos en /api/health."""

    def setUp(self):
        self._key = os.environ.get("GENOLY_API_KEY")
        os.environ["GENOLY_API_KEY"] = "secreta-prueba"
        self.addCleanup(self._restore)

    def _restore(self):
        if self._key is None:
            os.environ.pop("GENOLY_API_KEY", None)
        else:
            os.environ["GENOLY_API_KEY"] = self._key

    def test_health_sigue_publica(self):
        r = client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_sin_clave_devuelve_401_en_toda_la_superficie(self):
        rutas = [r for r in _api_routes() if r[1] != "/api/health"]
        self.assertGreaterEqual(len(rutas), 30, "faltan rutas /api/*")
        for method, path in rutas:
            for param in ("job_id", "upload_id", "dataset_id", "name", "job"):
                if "{" + param + "}" in path:
                    path = path.replace("{" + param + "}", "a" * 32)
            with self.subTest(method=method, path=path):
                r = client.request(method, path)
                self.assertEqual(r.status_code, 401)
                self.assertEqual(r.headers.get("www-authenticate"), "Bearer")

    def test_claves_validas_pasan(self):
        for headers in ({"X-API-Key": "secreta-prueba"},
                        {"Authorization": "Bearer secreta-prueba"}):
            with self.subTest(headers=headers):
                r = client.get("/api/health", headers=headers)
                self.assertEqual(r.status_code, 200)
        r = client.get("/api/health?api_key=secreta-prueba")
        self.assertEqual(r.status_code, 200)

    def test_clave_incorrecta_devuelve_401(self):
        r = client.get("/api/jobs/" + "a" * 32, headers={"X-API-Key": "otra"})
        self.assertEqual(r.status_code, 401)
        # /api/health se salta la comprobacion (HEALTHCHECK de Docker)
        r = client.get("/api/health", headers={"X-API-Key": "otra"})
        self.assertEqual(r.status_code, 200)

    def test_is_authorized_a_runitario(self):
        # sin clave configurada la API queda abierta (modo local)
        os.environ.pop("GENOLY_API_KEY", None)
        self.assertTrue(security.is_authorized({}, {}, "/api/jobs/x"))
        # y con clave, la clave correcta pasa por cualquiera de los tres canales
        os.environ["GENOLY_API_KEY"] = "secreta-prueba"
        self.assertTrue(security.is_authorized({"x-api-key": "secreta-prueba"}, None, "/x"))
        self.assertTrue(security.is_authorized({"authorization": "Bearer secreta-prueba"}, None, "/x"))
        self.assertTrue(security.is_authorized({}, {"api_key": "secreta-prueba"}, "/x"))
        self.assertFalse(security.is_authorized({}, None, "/x"))
        self.assertTrue(security.is_authorized({}, None, "/api/health"))


class UploadTest(unittest.TestCase):
    """Subidas: límites de tamaño, chunking y registro de rutas del servidor."""

    def setUp(self):
        self._key = os.environ.get("GENOLY_API_KEY")
        os.environ.pop("GENOLY_API_KEY", None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._key is None:
            os.environ.pop("GENOLY_API_KEY", None)
        else:
            os.environ["GENOLY_API_KEY"] = self._key

    def test_subida_multipart_genera_stats(self):
        tmp = tempfile.mkdtemp(prefix="api_up_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = Path(tmp) / "mini.fasta"
        write_fasta(path, [FastaRecord(id="s1", sequence="ACGT" * 25)])
        with path.open("rb") as fh:
            r = client.post("/api/upload", files={"file": ("mini.fasta", fh, "text/plain")})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["filename"], "mini.fasta")
        self.assertIn(body["stats_status"], ("pending", "ready", "error"))
        self.addCleanup(delete_upload, body["upload_id"])
        st = client.get(f"/api/upload/{body['upload_id']}")
        self.assertEqual(st.status_code, 200)

    def test_subida_vacia_devuelve_400(self):
        r = client.post("/api/upload",
                        files={"file": ("vacio.fasta", b"", "text/plain")})
        self.assertEqual(r.status_code, 400)

    def test_subida_sin_nombre_devuelve_422(self):
        # FastAPI ya exige un UploadFile con nombre antes de llegar al handler
        r = client.post("/api/upload", files={"file": ("", b"ACGT", "text/plain")})
        self.assertEqual(r.status_code, 422)

    def test_413_si_supera_el_limite(self):
        original = upload_router.MAX_UPLOAD_BYTES
        upload_router.MAX_UPLOAD_BYTES = 16
        try:
            r = client.post("/api/upload",
                            files={"file": ("g.fasta", b"A" * 64, "text/plain")})
        finally:
            upload_router.MAX_UPLOAD_BYTES = original
        self.assertEqual(r.status_code, 413)

    def test_chunked_rechaza_offset_y_tamano(self):
        r = client.post("/api/upload/chunked", json={"filename": "g.fasta", "size": 10})
        self.assertEqual(r.status_code, 200, r.text)
        upload_id = r.json()["upload_id"]
        self.addCleanup(delete_upload, upload_id)
        # el primer chunk va a offset 0 y deja 4 bytes
        r = client.put(f"/api/upload/chunked/{upload_id}?offset=0", content=b"AAAA")
        self.assertEqual(r.status_code, 200, r.text)
        # offset por delante del final -> 409
        r = client.put(f"/api/upload/chunked/{upload_id}?offset=99", content=b"BBBB")
        self.assertEqual(r.status_code, 409)
        # declarar más bytes que el límite -> 413
        original = maintenance.MAX_UPLOAD_BYTES
        maintenance.MAX_UPLOAD_BYTES = 8
        try:
            r = client.put(f"/api/upload/chunked/{upload_id}?offset=4",
                           content=b"CCCC",
                           headers={"content-length": "64"})
        finally:
            maintenance.MAX_UPLOAD_BYTES = original
        self.assertEqual(r.status_code, 413)
        # completar con el tamaño declarado distinto del recibido -> 400
        r = client.post(f"/api/upload/chunked/{upload_id}/complete", json={"size": 99})
        self.assertEqual(r.status_code, 400)

    def test_chunked_checksum_invalido(self):
        r = client.post("/api/upload/chunked", json={"filename": "c.fasta"})
        upload_id = r.json()["upload_id"]
        self.addCleanup(delete_upload, upload_id)
        client.put(f"/api/upload/chunked/{upload_id}?offset=0", content=b">s\nACGT\n")
        r = client.post(f"/api/upload/chunked/{upload_id}/complete",
                        json={"expected_sha256": "0" * 64})
        self.assertEqual(r.status_code, 400)
        # el checksum invalido borra el archivo: la subida deja de existir
        self.assertEqual(
            client.post(f"/api/upload/chunked/{upload_id}/complete",
                        json={"size": 1}).status_code,
            404)

    def test_registro_de_ruta_del_servidor(self):
        # el directorio de subidas siempre esta en el allowlist
        tmp = tempfile.mkdtemp(prefix="api_reg_", dir=UPLOAD_DIR)
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = Path(tmp) / "servidor.fasta"
        write_fasta(path, [FastaRecord(id="g", sequence="ACGT" * 10)])
        r = client.post("/api/upload/register", json={"path": str(path)})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.addCleanup(delete_upload, body["upload_id"])
        self.assertEqual(body["filename"], "servidor.fasta")
        # la ruta registrada sigue en el servidor (no se copia)
        self.assertTrue(path.is_file())

    def test_registro_rechaza_lfi_y_rutas_sensibles(self):
        for bad, code in (("/etc/passwd", 403),        # prefijo del sistema
                          ("/proc/self/environ", 403),  # prefijo del sistema
                          ("/no/existe/archivo.fasta", 400),
                          ("", 400)):
            with self.subTest(path=bad):
                r = client.post("/api/upload/register", json={"path": bad})
                self.assertEqual(r.status_code, code, r.text)

    def test_registro_fuera_del_allowlist_devuelve_403(self):
        tmp = tempfile.mkdtemp(prefix="api_fora_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = Path(tmp) / "lejano.fasta"
        write_fasta(path, [FastaRecord(id="g", sequence="ACGT")])
        r = client.post("/api/upload/register", json={"path": str(path)})
        self.assertEqual(r.status_code, 403)

    def test_descarga_e_status_de_id_desconocido(self):
        self.assertEqual(client.get("/api/upload/" + "f" * 32).status_code, 404)
        self.assertEqual(
            client.get("/api/upload/" + "f" * 32 + "/download").status_code, 404)

    def test_estado_de_subida_inexistente(self):
        # id con formato invalido -> 400; id valido sin registrar -> 404
        self.assertEqual(client.get("/api/upload/zzzz").status_code, 400)
        self.assertEqual(client.get("/api/upload/" + "f" * 32).status_code, 404)


class JobsTest(unittest.TestCase):
    """Polling y stream SSE de trabajos de fondo."""

    def test_status_y_404(self):
        self.assertEqual(client.get("/api/jobs/" + "f" * 32).status_code, 404)
        self.assertEqual(client.get("/api/jobs/!!.basura").status_code, 400)

    def _finished_job(self) -> str:
        async def make():
            job = jobs.manager.create("kmer")
            job.status = "done"
            job.result = {"records": 1}
            job.terminal = {"type": "done", "result": {"records": 1}}
            return job.id
        return asyncio.run(make())

    def test_sse_entrega_evento_terminal(self):
        job_id = self._finished_job()
        self.addCleanup(jobs.manager._jobs.pop, job_id, None)
        r = client.get(f"/api/jobs/{job_id}/events")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("data: {", r.text)
        self.assertIn('"type": "status"', r.text)
        self.assertIn('"type": "done"', r.text)

    def test_sse_de_trabajo_en_curso_no_colgar(self):
        # un job desconocido responde 404 antes de abrir el stream
        r = client.get(f"/api/jobs/{'a' * 32}/events")
        self.assertEqual(r.status_code, 404)

    def test_eventos_de_un_trabajo_real(self):
        tmp = tempfile.mkdtemp(prefix="api_job_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = Path(tmp) / "j.fasta"
        write_fasta(path, [FastaRecord(id="s", sequence="ACGT" * 10)])

        async def main():
            job = jobs.manager.create("kmer")
            jobs.manager.submit_process(
                job, {"kind": "kmer", "path": str(path), "k": 5,
                      "canonical": True, "min_abundance": 1, "top": 3,
                      "sequences": []})
            for _ in range(2400):
                if job.status in ("done", "error"):
                    break
                await asyncio.sleep(0.05)
            return job

        job = asyncio.run(main())
        self.addCleanup(jobs.manager._jobs.pop, job.id, None)
        self.assertEqual(job.status, "done", job.error)
        r = client.get(f"/api/jobs/{job.id}/events")
        self.assertIn('"type": "done"', r.text)


class RegistryTest(unittest.TestCase):
    """Registro de subidas compartido entre procesos (flock)."""

    def test_registrar_y_leer(self):
        from ui.backend import uploads
        upload_id = uuid4().hex
        uploads.create_upload(upload_id, "fasta", UPLOAD_DIR / "x.fasta",
                              "x.fasta", 10)
        self.addCleanup(delete_upload, upload_id)
        self.assertIsNotNone(uploads.get_upload(upload_id))
        uploads.delete_upload(upload_id)
        self.assertIsNone(uploads.get_upload(upload_id))


class PblupApiTest(unittest.TestCase):
    """Smoke test del endpoint PBLUP."""

    def test_predict_con_pedigree(self):
        payload = {
            "phenotypes": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "individuals": ["A1", "A2", "A3", "A4", "A5", "A6"],
            "sires": [None, None, "A1", "A1", "A2", "A3"],
            "dams": [None, None, "A4", "A5", None, None],
            "genetic_variance": 1.0,
            "residual_variance": 1.0,
        }
        r = client.post("/api/pblup/predict", json=payload)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["n_individuals"], 6)
        self.assertEqual(body["variance_source"], "dadas")
        self.assertEqual(len(body["blup"]), 6)
        for b in body["blup"]:
            self.assertIn("individual", b)
            self.assertIn("value", b)
            self.assertIn("reliability", b)
            self.assertIn("accuracy", b)
            self.assertIn("inbreeding", b)
            self.assertGreaterEqual(b["reliability"], 0.0)
            self.assertLessEqual(b["reliability"], 1.0)

    def test_longitudes_incoherentes_400(self):
        r = client.post("/api/pblup/predict", json={
            "phenotypes": [1.0] * 5,
            "individuals": ["A", "B", "C", "D", "E"],
            "sires": [None],
            "dams": [None] * 5,
            "genetic_variance": 1.0, "residual_variance": 1.0,
        })
        self.assertEqual(r.status_code, 400)

    def test_ciclo_400(self):
        r = client.post("/api/pblup/predict", json={
            "phenotypes": [1.0] * 5,
            "individuals": ["A", "B", "C", "D", "E"],
            "sires": ["B", "C", "D", "E", "A"],
            "dams": [None] * 5,
            "genetic_variance": 1.0, "residual_variance": 1.0,
        })
        self.assertEqual(r.status_code, 400)

    def test_gblup_acepta_individuals(self):
        r = client.post("/api/gblup/predict", json={
            "phenotypes": [1.0, 2.0, 3.0, 4.0, 5.0],
            "genotypes": [[0, 1, 2], [1, 1, 0], [2, 0, 1],
                          [0, 2, 1], [1, 0, 2]],
            "individuals": ["X1", "X2", "X3", "X4", "X5"],
            "genetic_variance": 1.0, "residual_variance": 1.0,
        })
        self.assertEqual(r.status_code, 200, r.text)
        ids = [b["individual"] for b in r.json()["blup"]]
        self.assertEqual(ids, ["X1", "X2", "X3", "X4", "X5"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

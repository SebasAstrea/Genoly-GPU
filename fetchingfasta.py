import urllib.request


def descargar_fasta(accession, filename):
    """Descarga una secuencia FASTA desde NCBI dado su número de acceso."""
    # Construye la URL para la API efetch de NCBI
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = f"?db=nuccore&id={accession}&rettype=fasta&retmode=text"
    url = base_url + params
    
    try:
        # Descarga la secuencia
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Secuencia {accession} descargada como '{filename}'")
    except Exception as e:
        print(f"✗ Error al descargar {accession}: {e}")

# Ejemplo: Descargar el gen BRCA1 humano
descargar_fasta("NM_007294", "BRCA1_humano.fasta")
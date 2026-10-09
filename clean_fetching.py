import requests


def descargar_archivo(url, filename):
    try:
        # 1. Petición con stream para eficiencia
        respuesta = requests.get(url, stream=True)
        respuesta.raise_for_status()
        
        # Guardaremos una pequeña muestra para el print final
        muestra = b"" 

        # 2. Escritura por pedazos (chunks)
        with open(filename, "wb") as archivo:
            for chunk in respuesta.iter_content(chunk_size=8024):
                if chunk:
                    archivo.write(chunk)
                    # Guardamos los primeros bytes solo para la previsualización
                    if len(muestra) < 100:
                        muestra += chunk

        print(f"✓ Archivo '{filename}' guardado correctamente.")
        # Decodificamos la muestra para que sea legible en consola
        print(f"--- Muestra del contenido ---\n{muestra[:100].decode('utf-8')}...") 
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error en la descarga: {e}")
    except Exception as e:
        print(f"✗ Error inesperado: {e}")

# URL de prueba (Gen BRCA1)
url_fasta = str(input("Introduce la URL del archivo FASTA: "))
name = input("Introduce el nombre del archivo a guardar (con extensión .fasta): ")
descargar_archivo(url_fasta, name)
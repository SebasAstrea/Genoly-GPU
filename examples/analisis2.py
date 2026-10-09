from pathlib import Path

from Bio import SeqIO, motifs
from Bio.Restriction import RestrictionBatch, Analysis

# Analizar sitios de enzimas de restricción
def analizar_sitios_restriccion(archivo_fasta):
    print("🔪 ANÁLISIS DE SITIOS DE RESTRICCIÓN")
    
    enzimas_comunes = ['EcoRI', 'BamHI', 'HindIII', 'NotI', 'XbaI']
    rb = RestrictionBatch(enzimas_comunes)
    
    for record in SeqIO.parse(archivo_fasta, "fasta"):
        print(f"\nSecuencia: {record.id}")
        analisis = Analysis(rb, record.seq)
        
        # Mostrar enzimas que cortan
        for enzima in enzimas_comunes:
            sitios = analisis[enzima]
            if sitios:
                print(f"  {enzima}: {len(sitios)} sitio(s) en posiciones {sitios}")

# Buscar motivos consenso (ej: promotores, sitios de unión)
def buscar_motivos_consenso(secuencia, motivo="TATAA"):
    """Busca el motivo TATA box común en promotores"""
    posiciones = []
    secuencia_str = str(secuencia).upper()
    motivo = motivo.upper()
    
    for i in range(len(secuencia_str) - len(motivo) + 1):
        if secuencia_str[i:i+len(motivo)] == motivo:
            posiciones.append(i)
    
    return posiciones

# Ejemplo de uso
# El fichero se busca junto a este script, no en el directorio de trabajo
_EJEMPLO = Path(__file__).with_name("ejemplo.fasta")

for record in SeqIO.parse(_EJEMPLO, "fasta"):
    tata_positions = buscar_motivos_consenso(record.seq, "TATAA")
    if tata_positions:
        print(f"TATA-box encontrado en {record.id} en posiciones: {tata_positions}")
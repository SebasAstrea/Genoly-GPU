from pathlib import Path

from Bio import SeqIO
from Bio.SeqUtils import gc_fraction  # CORRECTO para Biopython ≥1.79
import matplotlib.pyplot as plt

# Usaremos este archivo de ejemplo si no tienes el tuyo
# Puedes descargar uno real de: https://www.ncbi.nlm.nih.gov/nuccore/NC_045512
# O usar este ejemplo mínimo

# El fichero se busca junto a este script, no en el directorio de trabajo
_SEQDUMP = Path(__file__).with_name("seqdump.txt")

def crear_ejemplo_fasta():
    
    with open(_SEQDUMP, "r") as f:
        contenido = f.read()
# Crear archivo de ejemplo (si no tienes uno)
crear_ejemplo_fasta()

# ANÁLISIS FASTA
print("=" * 60)
print("ANÁLISIS DE ARCHIVO FASTA")
print("=" * 60)

# 1. Leer y analizar secuencias
secuencias_info = []
for record in SeqIO.parse(_SEQDUMP, "fasta"):
    # Información básica
    info = {
        'id': record.id,
        'descripcion': record.description,
        'secuencia': str(record.seq),
        'longitud': len(record.seq),
        'contenido_gc': gc_fraction(record.seq) * 100  # Multiplicar por 100 para porcentaje
    }
    secuencias_info.append(info)
    
    # Mostrar información
    print(f"\n🔬 SECUENCIA: {record.id}")
    print(f"   Descripción: {record.description[:80]}...")
    print(f"   Longitud: {info['longitud']} nucleótidos")
    print(f"   Contenido GC: {info['contenido_gc']:.2f}%")
    
    # Calcular composición de bases
    secuencia_str = str(record.seq).upper()
    a_count = secuencia_str.count('A')
    t_count = secuencia_str.count('T')
    g_count = secuencia_str.count('G')
    c_count = secuencia_str.count('C')
    otros = info['longitud'] - (a_count + t_count + g_count + c_count)
    
    print(f"   Composición:")
    print(f"     A: {a_count} ({a_count/info['longitud']*100:.1f}%)")
    print(f"     T: {t_count} ({t_count/info['longitud']*100:.1f}%)")
    print(f"     G: {g_count} ({g_count/info['longitud']*100:.1f}%)")
    print(f"     C: {c_count} ({c_count/info['longitud']*100:.1f}%)")
    if otros > 0:
        print(f"     Otros: {otros} ({otros/info['longitud']*100:.1f}%)")
    
    # Mostrar primeros 100 caracteres
    print(f"   Primeros 100 nucleótidos:")
    print(f"     {secuencia_str[:100]}...")

# 2. Análisis estadístico
if secuencias_info:
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS GENERALES")
    print("=" * 60)
    
    longitudes = [s['longitud'] for s in secuencias_info]
    gc_contents = [s['contenido_gc'] for s in secuencias_info]
    
    print(f"Número total de secuencias: {len(secuencias_info)}")
    print(f"Longitud total: {sum(longitudes)} nucleótidos")
    print(f"Longitud promedio: {sum(longitudes)/len(longitudes):.0f} nucleótidos")
    print(f"Longitud mínima: {min(longitudes)}")
    print(f"Longitud máxima: {max(longitudes)}")
    print(f"Contenido GC promedio: {sum(gc_contents)/len(gc_contents):.2f}%")
    
    # 3. Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de longitudes
    axes[0].bar([s['id'][:15] for s in secuencias_info], longitudes, color='skyblue')
    axes[0].set_xlabel('Secuencia')
    axes[0].set_ylabel('Longitud (nucleótidos)')
    axes[0].set_title('Longitud de secuencias')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Gráfico de contenido GC
    axes[1].bar([s['id'][:15] for s in secuencias_info], gc_contents, color='lightgreen')
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% GC')
    axes[1].set_xlabel('Secuencia')
    axes[1].set_ylabel('Contenido GC (%)')
    axes[1].set_title('Contenido GC por secuencia')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("\n No se encontraron secuencias en el archivo.")
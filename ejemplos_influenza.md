# Ejemplos de Comandos para Influenza A virus

## 📋 Proteínas objetivo tradicionales
Las proteínas con las que trabajabas anteriormente:
- **Neuraminidase** (Neuraminidasa)
- **Hemagglutinin** (Hemaglutinina) 
- **Matrix protein 2** (Proteína Matrix 2)
- **Matrix protein 1** (Proteína Matrix 1)
- **Nucleoprotein** (Nucleoproteína)

## 🚀 Opción 1: Solo datos de UniProt (Nuevo método dinámico)

### Comando básico con in_db_maker.py
```powershell
python in_db_maker.py
```

Este comando ejecutará automáticamente con:
- Virus: "Influenza A virus"
- Proteínas: ["Neuraminidase", "Hemagglutinin", "Matrix protein 2", "Matrix protein 1", "Nucleoprotein"]

### Comando personalizado con in_db_maker.py
Para modificar el virus o las proteínas, edita el archivo `in_db_maker.py` en las líneas finales:

```python
# Parámetros configurables
VIRUS_NAME = "Influenza A virus"
TARGET_PROTEINS = [
    "Neuraminidase", 
    "Hemagglutinin", 
    "Matrix protein 2", 
    "Matrix protein 1", 
    "Nucleoprotein"
]
```

## 🚀 Opción 2: Solo datos de NCBI (Método tradicional)

### Usando archivo de IDs
```powershell
python ncbi_downloader.py --input ncbi_ids_dict.tsv --ncbi-only
```

## 🚀 Opción 3: Combinado NCBI + UniProt

### Comando básico combinado
```powershell
python ncbi_downloader.py --virus-name "Influenza A virus" --target-proteins Neuraminidase Hemagglutinin "Matrix protein 2" "Matrix protein 1" Nucleoprotein
```

### Con configuración específica
```powershell
python ncbi_downloader.py --input ncbi_ids_dict.tsv --virus-name "Influenza A virus" --target-proteins Neuraminidase Hemagglutinin "Matrix protein 2" "Matrix protein 1" Nucleoprotein --batch-size 50 --workers 4
```

### Solo UniProt desde ncbi_downloader.py
```powershell
python ncbi_downloader.py --uniprot-only --virus-name "Influenza A virus" --target-proteins Neuraminidase Hemagglutinin "Matrix protein 2" "Matrix protein 1" Nucleoprotein
```

## 📊 Archivos de salida esperados

### Con in_db_maker.py:
- `output/proteome_Influenza_A_virus.csv` - Datos del proteoma
- `output/Neuraminidase_data.csv` - Secuencias de neuraminidasa
- `output/Hemagglutinin_data.csv` - Secuencias de hemaglutinina
- `output/Matrix_protein_2_data.csv` - Secuencias de Matrix protein 2
- `output/Matrix_protein_1_data.csv` - Secuencias de Matrix protein 1
- `output/Nucleoprotein_data.csv` - Secuencias de nucleoproteína

### Con ncbi_downloader.py:
- Archivos FASTA individuales por proteína (si usa NCBI)
- Archivos CSV con datos de UniProt (si se especifica)
- Logs de progreso y estadísticas

## 🔧 Configuración avanzada

### Archivo config.json personalizado
Puedes crear un archivo `config.json` con:
```json
{
    "download": {
        "batch_size": 50,
        "max_workers": 4,
        "retry_attempts": 3,
        "timeout": 60
    },
    "uniprot": {
        "virus_name": "Influenza A virus",
        "target_proteins": [
            "Neuraminidase",
            "Hemagglutinin", 
            "Matrix protein 2",
            "Matrix protein 1",
            "Nucleoprotein"
        ]
    }
}
```

Y usar:
```powershell
python ncbi_downloader.py --config config.json
```

## 💡 Recomendaciones

### Para empezar rápido:
```powershell
python in_db_maker.py
```

### Para máximo control:
```powershell
python ncbi_downloader.py --virus-name "Influenza A virus" --target-proteins Neuraminidase Hemagglutinin "Matrix protein 2" "Matrix protein 1" Nucleoprotein --batch-size 100 --workers 6
```

### Para pruebas (sin descargar):
```powershell
python ncbi_downloader.py --dry-run --virus-name "Influenza A virus" --target-proteins Neuraminidase
```

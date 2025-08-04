# NCBI-Uniprot FASTA Downloader

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**[English](#english) | [Español](#español)**

---

## English

### Overview

A high-performance Python tool for downloading FASTA sequences from NCBI in bulk. Designed to efficiently handle large datasets with intelligent batch processing, parallel execution, and robust error handling.

## Features

- **Batch Processing**: Download multiple sequences per API call for efficiency
- **Parallel Execution**: Configurable worker threads for optimal performance
- **Smart Rate Limiting**: Respects NCBI's API limits with intelligent timing
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Progress Persistence**: Resume downloads from interruption points
- **Duplicate Detection**: Optional hash-based duplicate filtering
- **Comprehensive Logging**: Detailed logs with performance metrics
- **Graceful Shutdown**: Safe interruption handling with progress preservation

## Installation

### Clone and Setup
```bash
git clone https://github.com/SimonAirInsti/ncbi-fasta-downloader.git
cd ncbi-fasta-downloader
pip install -r requirements.txt
python setup.py
```

## Configuration

The setup script will ask for your email address (required by NCBI) and create a `config.json` file:

```json
{
  "email": "your.email@domain.com",
  "batch_size": 200,
  "max_workers": 4,
  "rate_limit_delay": 0.34
}
```

### Prepare Input File
Create a TSV file with your NCBI IDs:
```
Neuraminidase	3024910520,3024910499,3024910476
Hemagglutinin	3024910409,3024910387,3024910364
Matrix protein 1	164584001,164584002,164584003
```

## Quick Start

### Basic Usage
```bash
# Run with default settings
python ncbi_downloader.py

# Use custom input file
python ncbi_downloader.py --input my_ids.tsv

# Override batch size and workers
python ncbi_downloader.py --batch-size 100 --workers 2
```

### Test with Sample Data
```bash
# Test with small dataset
python ncbi_downloader.py --input sample_ncbi_ids.tsv --dry-run
python ncbi_downloader.py --input sample_ncbi_ids.tsv
```

### Monitor Progress
```bash
# Check current status
python check_progress.py

# Follow logs in real-time
tail -f output/ncbi_download.log
```

### 📁 Project Structure

```
ncbi-fasta-downloader/
├── ncbi_downloader.py         # Main downloader script
├── setup.py                   # Interactive setup script
├── config.json                # Configuration (created by setup)
├── requirements.txt           # Python dependencies
├── check_progress.py          # Progress monitoring utility
├── sample_ncbi_ids.tsv        # Sample data for testing
├── test_suite.py              # Comprehensive test suite
├── output/                    # Output directory
│   ├── all_sequences.fasta    # Downloaded sequences
│   ├── progress.json          # Detailed progress tracking
│   ├── failed_ids.txt         # Failed downloads log
│   └── ncbi_download.log      # Execution logs
└── .github/                   # GitHub Actions workflows
```

### 🔧 Advanced Configuration

#### Performance Tuning
```bash
# High-performance setup (stable connection required)
python ncbi_downloader.py --batch-size 500 --workers 6

# Conservative setup (unstable connection)
python ncbi_downloader.py --batch-size 50 --workers 2
```

#### Command Line Options
```bash
python ncbi_downloader.py --help
```

Options include:
- `--input`: Input TSV file path
- `--config`: Configuration file path  
- `--batch-size`: Sequences per API call
- `--workers`: Number of parallel workers
- `--dry-run`: Test mode without downloading

### 🔍 Monitoring and Troubleshooting

#### Check Progress
```bash
python check_progress.py
```

#### View Logs
```bash
# Recent activity
tail -50 output/ncbi_download.log

# Follow in real-time
tail -f output/ncbi_download.log

# Filter errors
grep ERROR output/ncbi_download.log
```

#### Resume Downloads
The downloader automatically resumes from the last checkpoint if interrupted.

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with tests
4. Submit a pull request

### 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- NCBI for providing the E-utilities API
- BioPython community for the excellent Bio library
- Contributors and users providing feedback

---

## Español

### Descripción General

Herramienta Python de alto rendimiento para descargar secuencias FASTA de NCBI en masa. Diseñada para manejar eficientemente grandes conjuntos de datos con procesamiento inteligente por lotes, ejecución paralela y manejo robusto de errores.

### 🚀 Características Principales

- **Procesamiento por Lotes**: Descarga múltiples secuencias por llamada API para mayor eficiencia
- **Ejecución Paralela**: Hilos de trabajo configurables para rendimiento óptimo
- **Limitación Inteligente de Velocidad**: Respeta los límites de la API de NCBI con temporización inteligente
- **Manejo Robusto de Errores**: Reintentos automáticos con retroceso exponencial
- **Persistencia de Progreso**: Reanuda descargas desde puntos de interrupción
- **Detección de Duplicados**: Filtrado opcional de duplicados basado en hash
- **Registro Completo**: Logs detallados con métricas de rendimiento
- **Apagado Elegante**: Manejo seguro de interrupciones con preservación del progreso

### 🛠️ Instalación y Configuración

#### 1. Clonar y Configurar
```bash
git clone https://github.com/SimonAirInsti/ncbi-fasta-downloader.git
cd ncbi-fasta-downloader
pip install -r requirements.txt
python setup.py
```

#### 2. Configuración
El script de configuración solicitará tu dirección de email (requerida por NCBI) y creará un archivo `config.json`:

```json
{
  "email": "tu.email@dominio.com",
  "batch_size": 200,
  "max_workers": 4,
  "rate_limit_delay": 0.34
}
```

#### 3. Preparar Archivo de Entrada
Crea un archivo TSV con tus IDs de NCBI:
```
Neuraminidase	3024910520,3024910499,3024910476
Hemagglutinin	3024910409,3024910387,3024910364
Matrix protein 1	164584001,164584002,164584003
```

### 🎯 Inicio Rápido

#### Uso Básico
```bash
# Ejecutar con configuración predeterminada
python ncbi_downloader.py

# Usar archivo de entrada personalizado
python ncbi_downloader.py --input mis_ids.tsv

# Sobrescribir tamaño de lote y trabajadores
python ncbi_downloader.py --batch-size 100 --workers 2
```

#### Probar con Datos de Muestra
```bash
# Probar con dataset pequeño
python ncbi_downloader.py --input sample_ncbi_ids.tsv --dry-run
python ncbi_downloader.py --input sample_ncbi_ids.tsv
```

#### Monitorear Progreso
```bash
# Verificar estado actual
python check_progress.py

# Seguir logs en tiempo real
tail -f output/ncbi_download.log
```

### 🔧 Configuración Avanzada

#### Ajuste de Rendimiento
```bash
# Configuración de alto rendimiento (conexión estable requerida)
python ncbi_downloader.py --batch-size 500 --workers 6

# Configuración conservadora (conexión inestable)
python ncbi_downloader.py --batch-size 50 --workers 2
```

### 🔍 Monitoreo y Solución de Problemas

#### Verificar Progreso
```bash
python check_progress.py
```

#### Ver Logs
```bash
# Actividad reciente
tail -50 output/ncbi_download.log

# Seguir en tiempo real
tail -f output/ncbi_download.log

# Filtrar errores
grep ERROR output/ncbi_download.log
```

### 🤝 Contribuir

1. Hacer fork del repositorio
2. Crear una rama de características: `git checkout -b nombre-caracteristica`
3. Realizar cambios con pruebas
4. Enviar un pull request

### 📝 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

**⭐ Si encuentras útil este proyecto, ¡considera darle una estrella en GitHub!**

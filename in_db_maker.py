import requests
import gzip
import io
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import re
import numpy as np
from Bio import SeqIO
import urllib.parse

class protDB():
    def __init__(self, virus_name="Influenza A virus", target_proteins=None):
        """
        Inicializa la clase con parámetros dinámicos.
        
        Args:
            virus_name (str): Nombre del virus (ej. "Influenza A virus")
            target_proteins (list): Lista de proteínas objetivo
        """
        if target_proteins is None:
            target_proteins = [
                "Neuraminidase", 
                "Hemagglutinin", 
                "Matrix protein 2", 
                "Matrix protein 1", 
                "Nucleoprotein"
            ]
        
        self.virus_name = virus_name
        self.target_proteins = target_proteins
        
        print(f"🦠 Virus objetivo: {virus_name}")
        print(f"🧬 Proteínas objetivo: {', '.join(target_proteins)}")
        
        # Descargar y procesar proteoma
        print("\n📥 Descargando proteoma...")
        self.proteome_db_incomplete = self.download_proteome(virus_name)
        
        # Añadir datos antigénicos
        print("\n🔍 Añadiendo datos antigénicos...")
        self.proteome_db = self.add_antigenic(self.proteome_db_incomplete, target_proteins)
        
        # Descargar y procesar archivos FASTA para cada proteína
        print("\n📥 Descargando archivos FASTA...")
        for protein in target_proteins:
            print(f"Procesando {protein}...")
            db = self.download_and_extract_protein_data(protein, virus_name)
            setattr(self, f"{protein.replace(' ', '_')}", db)
    
    def download_proteome(self, virus_name):
        """
        Descarga el proteoma desde UniProt API de forma optimizada.
        Solo descarga proteomas específicos y relevantes.
        
        Args:
            virus_name (str): Nombre del virus
            
        Returns:
            pd.DataFrame: Proteoma como DataFrame
        """
        try:
            # Codificar el nombre del virus para URL
            encoded_virus = urllib.parse.quote_plus(virus_name)
            
            # URL optimizada - buscar proteomas de referencia y completos
            url = f"https://rest.uniprot.org/proteomes/stream?compressed=true&fields=upid%2Corganism%2Corganism_id%2Cprotein_count%2Cbusco%2Ccpd&format=tsv&query=organism_name%3A%22{encoded_virus}%22+AND+%28proteome_type%3Areference+OR+proteome_type%3Acomplete%29"
            
            print(f"🌐 Descargando proteomas de referencia desde: {url}")
            
            # Enviar solicitud
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Descomprimir contenido
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                content = gz.read().decode('utf-8')
            
            # Convertir a DataFrame
            df = pd.read_csv(io.StringIO(content), sep='\t')
            
            if df.empty:
                print(f"⚠️ No se encontraron proteomas de referencia para '{virus_name}'")
                print(f"🔄 Intentando búsqueda más amplia...")
                
                # Fallback - búsqueda más específica pero limitada
                url_fallback = f"https://rest.uniprot.org/proteomes/stream?compressed=true&fields=upid%2Corganism%2Corganism_id%2Cprotein_count%2Cbusco%2Ccpd&format=tsv&query=organism_name%3A%22{encoded_virus}%22&size=50"
                
                response = requests.get(url_fallback, timeout=60)
                response.raise_for_status()
                
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
                
                df = pd.read_csv(io.StringIO(content), sep='\t')
            
            if df.empty:
                print(f"⚠️ No se encontraron proteomas para '{virus_name}'")
                return pd.DataFrame()
            
            # Filtrar solo entradas relevantes del virus específico
            if 'Organism' in df.columns:
                virus_filter = df['Organism'].str.contains(virus_name, case=False, na=False)
                df = df[virus_filter]
            
            # Usar 'Proteome Id' como índice si existe
            if 'Proteome Id' in df.columns:
                df = df.set_index('Proteome Id')
            
            print(f"✅ Proteomas filtrados descargados: {len(df)} entradas (optimizado)")
            return df
            
        except Exception as e:
            print(f"❌ Error descargando proteoma: {e}")
            return pd.DataFrame()
    
    def download_protein_fasta(self, protein_name, virus_name):
        """
        Descarga archivo FASTA de una proteína específica desde UniProt de forma optimizada.
        
        Args:
            protein_name (str): Nombre de la proteína
            virus_name (str): Nombre del virus para filtrado específico
            
        Returns:
            str: Contenido FASTA como string
        """
        try:
            # Codificar nombres para URL
            encoded_protein = urllib.parse.quote_plus(protein_name)
            encoded_virus = urllib.parse.quote_plus(virus_name)
            
            # URL optimizada - búsqueda específica por proteína y organismo
            url = f"https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=protein_name%3A%22{encoded_protein}%22+AND+organism_name%3A%22{encoded_virus}%22&size=1000"
            
            print(f"🌐 Descargando FASTA optimizado: {protein_name} en {virus_name}")
            
            # Enviar solicitud
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Descomprimir contenido
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                content = gz.read().decode('utf-8')
            
            if not content.strip():
                print(f"⚠️ FASTA vacío para '{protein_name}' en '{virus_name}'")
                # Fallback con búsqueda más amplia
                url_fallback = f"https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%22{encoded_protein}%22+AND+taxonomy_name%3A%22{encoded_virus}%22&size=1000"
                
                response = requests.get(url_fallback, timeout=120)
                response.raise_for_status()
                
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
            
            if content.strip():
                # Contar secuencias FASTA
                seq_count = len([line for line in content.split('\n') if line.startswith('>')])
                print(f"✅ FASTA descargado para {protein_name}: {seq_count} secuencias")
            else:
                print(f"⚠️ No se encontraron secuencias para {protein_name}")
            
            return content
            
        except Exception as e:
            print(f"❌ Error descargando FASTA para {protein_name}: {e}")
            return ""
    
    def fasta_string_to_dataframe(self, fasta_content):
        """
        Convierte contenido FASTA string a DataFrame.
        
        Args:
            fasta_content (str): Contenido FASTA
            
        Returns:
            pd.DataFrame: DataFrame con ID, Sequence, Description
        """
        data = []
        try:
            # Usar StringIO para simular un archivo
            fasta_io = io.StringIO(fasta_content)
            
            for record in SeqIO.parse(fasta_io, "fasta"):
                data.append((record.id, str(record.seq), str(record.description)))
                
        except Exception as e:
            print(f"❌ Error parseando FASTA: {e}")
            
        return pd.DataFrame(data, columns=["ID", "Sequence", "Description"])
    
    def download_and_extract_protein_data(self, protein_name, virus_name):
        """
        Descarga y procesa datos de una proteína específica.
        
        Args:
            protein_name (str): Nombre de la proteína
            virus_name (str): Nombre del virus para filtrado
            
        Returns:
            pd.DataFrame: Datos procesados de la proteína
        """
        try:
            # Descargar FASTA
            fasta_content = self.download_protein_fasta(protein_name, virus_name)
            
            if not fasta_content:
                print(f"⚠️ No se pudo descargar FASTA para {protein_name}")
                return pd.DataFrame()
            
            # Convertir a DataFrame
            data = self.fasta_string_to_dataframe(fasta_content)
            
            if data.empty:
                print(f"⚠️ No hay datos para {protein_name}")
                return pd.DataFrame()

            # Añadir columna "Prot ID"
            data["Prot ID"] = data['ID'].str.split('|').str[1]

            # Añadir columna "Protein type"
            data["Protein type"] = protein_name
            
            # Extraer organismo usando regex
            organism_pattern = re.compile(r'OS=(.*?) OX=')
            matched_text = []
            
            for description in data['Description']:
                match = organism_pattern.search(description)
                if match:
                    matched_text.append(match.group(1))
                else:
                    matched_text.append(None)
            
            data['Organism'] = matched_text

            # Filtrar por virus específico
            virus_filter = data['Organism'].str.contains(virus_name, na=False)
            
            # Filtrar por proteína en la descripción
            protein_filter = data['Description'].str.contains(protein_name, case=False, na=False)
            
            # Aplicar ambos filtros
            filtered_data = data[virus_filter & protein_filter].drop_duplicates(subset=['Sequence'], keep='first')

            # Extraer variantes
            if not filtered_data.empty:
                variant_pattern = re.compile(rf'{re.escape(virus_name)} \((.*?)\)')
                variants = []
                
                for organism in filtered_data['Organism']:
                    match = variant_pattern.search(organism) if organism else None
                    if match:
                        variants.append(match.group(1))
                    else:
                        variants.append(np.nan)
                
                # Crear copia para evitar warnings
                filtered_data = filtered_data.copy()
                filtered_data["Variants"] = variants
            
            print(f"✅ {protein_name}: {len(filtered_data)} secuencias procesadas")
            return filtered_data
            
        except Exception as e:
            print(f"❌ Error procesando {protein_name}: {e}")
            return pd.DataFrame()
    
    def add_antigenic(self, df, target_proteins):
        """
        Añade datos antigénicos al DataFrame de proteomas de forma optimizada.
        
        Args:
            df (pd.DataFrame): DataFrame de proteomas
            target_proteins (list): Lista de proteínas objetivo
            
        Returns:
            pd.DataFrame: DataFrame con datos antigénicos añadidos
        """
        if df.empty:
            print("⚠️ DataFrame de proteomas vacío, saltando datos antigénicos")
            return df
            
        try:
            # Limitar a los primeros 10 proteomas más relevantes para acelerar el proceso
            if len(df) > 10:
                print(f"🔧 Limitando procesamiento a los primeros 10 proteomas más relevantes (de {len(df)})")
                df_to_process = df.head(10)
            else:
                df_to_process = df

            def find_antigenic_entries(prot_id):
                """Buscar proteínas antigénicas para un proteoma específico."""
                try:
                    # URL optimizada para búsqueda específica
                    proteins_query = "+OR+".join([f'protein_name%3A"{p}"' for p in target_proteins])
                    url = f"https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=tsv&query=proteome%3A{prot_id}+AND+%28{proteins_query}%29&size=100"
                    
                    # Enviar solicitud
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Descomprimir contenido
                    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                        content = gz.read().decode('utf-8')
                    
                    if not content.strip():
                        return {protein: "NaN" for protein in target_proteins}
                    
                    # Leer DataFrame
                    df_api = pd.read_csv(io.StringIO(content), sep='\t')
                    
                    # Buscar coincidencias para cada proteína objetivo
                    matching_entries = {}
                    for protein in target_proteins:
                        if 'Protein names' in df_api.columns:
                            protein_matches = df_api["Protein names"].str.contains(protein, case=False, na=False)
                            entries = df_api[protein_matches]["Entry"].tolist()
                            matching_entries[protein] = ", ".join(entries) if entries else "NaN"
                        else:
                            matching_entries[protein] = "NaN"
                    
                    return matching_entries
                    
                except Exception as e:
                    print(f"⚠️ Error obteniendo datos para proteoma {prot_id}: {e}")
                    return {protein: "NaN" for protein in target_proteins}

            def process_row(index):
                """Procesar cada fila del DataFrame."""
                matching_entries = find_antigenic_entries(index)
                return index, matching_entries

            # Procesamiento paralelo optimizado
            n_jobs = min(10, len(df_to_process))  # Máximo 10 workers para evitar sobrecarga
            print(f"🔄 Procesando {len(df_to_process)} proteomas con {n_jobs} workers (optimizado)...")
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_row)(index) for index in tqdm(df_to_process.index, desc="Obteniendo proteínas antigénicas")
            )

            # Actualizar DataFrame
            df_copy = df.copy()
            for index, matching_entries in results:
                for protein in target_proteins:
                    df_copy.loc[index, f"{protein} IDs"] = matching_entries.get(protein, "NaN")
            
            print("✅ Datos antigénicos añadidos (optimizado)")
            return df_copy
            
        except Exception as e:
            print(f"❌ Error añadiendo datos antigénicos: {e}")
            return df
    
    def save_results(self, output_dir="output"):
        """
        Guarda todos los resultados en archivos CSV.
        
        Args:
            output_dir (str): Directorio de salida
        """
        import os
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Guardar proteoma
            proteome_file = os.path.join(output_dir, f"proteome_{self.virus_name.replace(' ', '_')}.csv")
            self.proteome_db.to_csv(proteome_file)
            print(f"💾 Proteoma guardado: {proteome_file}")
            
            # Guardar datos de cada proteína
            for protein in self.target_proteins:
                attr_name = protein.replace(' ', '_')
                if hasattr(self, attr_name):
                    protein_data = getattr(self, attr_name)
                    if not protein_data.empty:
                        protein_file = os.path.join(output_dir, f"{attr_name}_data.csv")
                        protein_data.to_csv(protein_file, index=False)
                        print(f"💾 {protein} guardado: {protein_file}")
                    else:
                        print(f"⚠️ No hay datos para {protein}")
                        
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")

# Ejemplo de uso dinámico
if __name__ == "__main__":
    # Parámetros configurables
    VIRUS_NAME = "Influenza A virus"
    TARGET_PROTEINS = [
        "Neuraminidase", 
        "Hemagglutinin", 
        "Matrix protein 2", 
        "Matrix protein 1", 
        "Nucleoprotein"
    ]
    
    # Crear base de datos
    print("🚀 Iniciando descarga automática de datos UniProt...")
    db = protDB(virus_name=VIRUS_NAME, target_proteins=TARGET_PROTEINS)
    
    # Guardar resultados
    db.save_results()
    
    print("\n🎉 ¡Proceso completado!")

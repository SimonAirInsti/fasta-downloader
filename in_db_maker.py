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
        Initializes the class with dynamic parameters.
        
        Args:
            virus_name (str): Name of the virus (e.g. "Influenza A virus")
            target_proteins (list): List of target proteins
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
        
        print(f"🦠 Target virus: {virus_name}")
        print(f"🧬 Target proteins: {', '.join(target_proteins)}")
        
        # Download and process proteome
        print("\n📥 Downloading proteome...")
        self.proteome_db_incomplete = self.download_proteome(virus_name)
        
        # Add antigenic data
        print("\n🔍 Adding antigenic data...")
        self.proteome_db = self.add_antigenic(self.proteome_db_incomplete, target_proteins)
        
        # Download and process FASTA files for each protein
        print("\n📥 Downloading FASTA files...")
        for protein in target_proteins:
            print(f"Processing {protein}...")
            db = self.download_and_extract_protein_data(protein, virus_name)
            setattr(self, f"{protein.replace(' ', '_')}", db)
    
    def download_proteome(self, virus_name):
        """
        Downloads the proteome from UniProt API in an optimized way.
        Only downloads specific and relevant proteomes.
        
        Args:
            virus_name (str): Name of the virus
            
        Returns:
            pd.DataFrame: Proteome as DataFrame
        """
        try:
            # Encode virus name for URL
            encoded_virus = urllib.parse.quote_plus(virus_name)
            
            # Optimized URL - search for reference and complete proteomes
            url = f"https://rest.uniprot.org/proteomes/stream?compressed=true&fields=upid%2Corganism%2Corganism_id%2Cprotein_count%2Cbusco%2Ccpd&format=tsv&query=organism_name%3A%22{encoded_virus}%22+AND+%28proteome_type%3Areference+OR+proteome_type%3Acomplete%29"
            
            print(f"🌐 Downloading reference proteomes from: {url}")
            
            # Send request
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Decompress content
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                content = gz.read().decode('utf-8')
            
            # Convert to DataFrame
            df = pd.read_csv(io.StringIO(content), sep='\t')
            
            if df.empty:
                print(f"⚠️ No reference proteomes found for '{virus_name}'")
                print(f"🔄 Trying broader search...")
                
                # Fallback - more specific but limited search
                url_fallback = f"https://rest.uniprot.org/proteomes/stream?compressed=true&fields=upid%2Corganism%2Corganism_id%2Cprotein_count%2Cbusco%2Ccpd&format=tsv&query=organism_name%3A%22{encoded_virus}%22&size=50"
                
                response = requests.get(url_fallback, timeout=60)
                response.raise_for_status()
                
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
                
                df = pd.read_csv(io.StringIO(content), sep='\t')
            
            if df.empty:
                print(f"⚠️ No proteomes found for '{virus_name}'")
                return pd.DataFrame()
            
            # Filter only relevant entries of the specific virus
            if 'Organism' in df.columns:
                virus_filter = df['Organism'].str.contains(virus_name, case=False, na=False)
                df = df[virus_filter]
            
            # Use 'Proteome Id' as index if it exists
            if 'Proteome Id' in df.columns:
                df = df.set_index('Proteome Id')
            
            print(f"✅ Filtered proteomes downloaded: {len(df)} entries (optimized)")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading proteome: {e}")
            return pd.DataFrame()
    
    def download_protein_fasta(self, protein_name, virus_name):
        """
        Downloads FASTA file of a specific protein from UniProt in an optimized way.
        
        Args:
            protein_name (str): Name of the protein
            virus_name (str): Name of the virus for specific filtering
            
        Returns:
            str: FASTA content as string
        """
        try:
            # Encode names for URL
            encoded_protein = urllib.parse.quote_plus(protein_name)
            encoded_virus = urllib.parse.quote_plus(virus_name)
            
            # Optimized URL - specific search by protein and organism
            url = f"https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=protein_name%3A%22{encoded_protein}%22+AND+organism_name%3A%22{encoded_virus}%22&size=1000"
            
            print(f"🌐 Downloading optimized FASTA: {protein_name} in {virus_name}")
            
            # Send request
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Decompress content
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                content = gz.read().decode('utf-8')
            
            if not content.strip():
                print(f"⚠️ Empty FASTA for '{protein_name}' in '{virus_name}'")
                # Fallback with broader search
                url_fallback = f"https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%22{encoded_protein}%22+AND+taxonomy_name%3A%22{encoded_virus}%22&size=1000"
                
                response = requests.get(url_fallback, timeout=120)
                response.raise_for_status()
                
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
            
            if content.strip():
                # Count FASTA sequences
                seq_count = len([line for line in content.split('\n') if line.startswith('>')])
                print(f"✅ FASTA downloaded for {protein_name}: {seq_count} sequences")
            else:
                print(f"⚠️ No sequences found for {protein_name}")
            
            return content
            
        except Exception as e:
            print(f"❌ Error downloading FASTA for {protein_name}: {e}")
            return ""
    
    def fasta_string_to_dataframe(self, fasta_content):
        """
        Converts FASTA string content to DataFrame.
        
        Args:
            fasta_content (str): FASTA content
            
        Returns:
            pd.DataFrame: DataFrame with ID, Sequence, Description
        """
        data = []
        try:
            # Use StringIO to simulate a file
            fasta_io = io.StringIO(fasta_content)
            
            for record in SeqIO.parse(fasta_io, "fasta"):
                data.append((record.id, str(record.seq), str(record.description)))
                
        except Exception as e:
            print(f"❌ Error parsing FASTA: {e}")
            
        return pd.DataFrame(data, columns=["ID", "Sequence", "Description"])
    
    def download_and_extract_protein_data(self, protein_name, virus_name):
        """
        Downloads and processes data for a specific protein.
        
        Args:
            protein_name (str): Name of the protein
            virus_name (str): Name of the virus for filtering
            
        Returns:
            pd.DataFrame: Processed protein data
        """
        try:
            # Download FASTA
            fasta_content = self.download_protein_fasta(protein_name, virus_name)
            
            if not fasta_content:
                print(f"⚠️ Could not download FASTA for {protein_name}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = self.fasta_string_to_dataframe(fasta_content)
            
            if data.empty:
                print(f"⚠️ No data for {protein_name}")
                return pd.DataFrame()

            # Add "Prot ID" column
            data["Prot ID"] = data['ID'].str.split('|').str[1]

            # Add "Protein type" column
            data["Protein type"] = protein_name
            
            # Extract organism using regex
            organism_pattern = re.compile(r'OS=(.*?) OX=')
            matched_text = []
            
            for description in data['Description']:
                match = organism_pattern.search(description)
                if match:
                    matched_text.append(match.group(1))
                else:
                    matched_text.append(None)
            
            data['Organism'] = matched_text

            # Filter by specific virus
            virus_filter = data['Organism'].str.contains(virus_name, na=False)
            
            # Filter by protein in description
            protein_filter = data['Description'].str.contains(protein_name, case=False, na=False)
            
            # Apply both filters
            filtered_data = data[virus_filter & protein_filter].drop_duplicates(subset=['Sequence'], keep='first')

            # Extract variants
            if not filtered_data.empty:
                variant_pattern = re.compile(rf'{re.escape(virus_name)} \((.*?)\)')
                variants = []
                
                for organism in filtered_data['Organism']:
                    match = variant_pattern.search(organism) if organism else None
                    if match:
                        variants.append(match.group(1))
                    else:
                        variants.append(np.nan)
                
                # Create copy to avoid warnings
                filtered_data = filtered_data.copy()
                filtered_data["Variants"] = variants
            
            print(f"✅ {protein_name}: {len(filtered_data)} sequences processed")
            return filtered_data
            
        except Exception as e:
            print(f"❌ Error processing {protein_name}: {e}")
            return pd.DataFrame()
    
    def add_antigenic(self, df, target_proteins):
        """
        Adds antigenic data to the proteome DataFrame in an optimized way.
        
        Args:
            df (pd.DataFrame): Proteome DataFrame
            target_proteins (list): List of target proteins
            
        Returns:
            pd.DataFrame: DataFrame with added antigenic data
        """
        if df.empty:
            print("⚠️ Empty proteome DataFrame, skipping antigenic data")
            return df
            
        try:
            # Limit to the first 10 most relevant proteomes to speed up the process
            if len(df) > 10:
                print(f"🔧 Limiting processing to the first 10 most relevant proteomes (of {len(df)})")
                df_to_process = df.head(10)
            else:
                df_to_process = df

            def find_antigenic_entries(prot_id):
                """Search for antigenic proteins for a specific proteome."""
                try:
                    # Optimized URL for specific search
                    proteins_query = "+OR+".join([f'protein_name%3A"{p}"' for p in target_proteins])
                    url = f"https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=tsv&query=proteome%3A{prot_id}+AND+%28{proteins_query}%29&size=100"
                    
                    # Send request
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Decompress content
                    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                        content = gz.read().decode('utf-8')
                    
                    if not content.strip():
                        return {protein: "NaN" for protein in target_proteins}
                    
                    # Read DataFrame
                    df_api = pd.read_csv(io.StringIO(content), sep='\t')
                    
                    # Search for matches for each target protein
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
                    print(f"⚠️ Error getting data for proteome {prot_id}: {e}")
                    return {protein: "NaN" for protein in target_proteins}

            def process_row(index):
                """Process each DataFrame row."""
                matching_entries = find_antigenic_entries(index)
                return index, matching_entries

            # Optimized parallel processing
            n_jobs = min(10, len(df_to_process))  # Maximum 10 workers to avoid overload
            print(f"🔄 Processing {len(df_to_process)} proteomes with {n_jobs} workers (optimized)...")
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_row)(index) for index in tqdm(df_to_process.index, desc="Getting antigenic proteins")
            )

            # Update DataFrame
            df_copy = df.copy()
            for index, matching_entries in results:
                for protein in target_proteins:
                    df_copy.loc[index, f"{protein} IDs"] = matching_entries.get(protein, "NaN")
            
            print("✅ Antigenic data added (optimized)")
            return df_copy
            
        except Exception as e:
            print(f"❌ Error adding antigenic data: {e}")
            return df
    
    def save_results(self, output_dir="output"):
        """
        Saves all results to CSV files.
        
        Args:
            output_dir (str): Output directory
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save proteome
            proteome_file = os.path.join(output_dir, f"proteome_{self.virus_name.replace(' ', '_')}.csv")
            self.proteome_db.to_csv(proteome_file)
            print(f"💾 Proteome saved: {proteome_file}")
            
            # Save data for each protein
            for protein in self.target_proteins:
                attr_name = protein.replace(' ', '_')
                if hasattr(self, attr_name):
                    protein_data = getattr(self, attr_name)
                    if not protein_data.empty:
                        protein_file = os.path.join(output_dir, f"{attr_name}_data.csv")
                        protein_data.to_csv(protein_file, index=False)
                        print(f"💾 {protein} saved: {protein_file}")
                    else:
                        print(f"⚠️ No data for {protein}")
                        
        except Exception as e:
            print(f"❌ Error saving results: {e}")

# Dynamic usage example
if __name__ == "__main__":
    # Configurable parameters
    VIRUS_NAME = "Influenza A virus"
    TARGET_PROTEINS = [
        "Neuraminidase", 
        "Hemagglutinin", 
        "Matrix protein 2", 
        "Matrix protein 1", 
        "Nucleoprotein"
    ]
    
    # Create database
    print("🚀 Starting automatic UniProt data download...")
    db = protDB(virus_name=VIRUS_NAME, target_proteins=TARGET_PROTEINS)
    
    # Save results
    db.save_results()
    
    print("\n🎉 Process completed!")

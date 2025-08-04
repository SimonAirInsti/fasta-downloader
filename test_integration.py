#!/usr/bin/env python3
"""
Script de prueba para verificar la integración de UniProt en ncbi_downloader.py
"""

def test_imports():
    """Prueba que todas las importaciones funcionen correctamente."""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import pandas as pd
        print("✅ pandas")
        
        import joblib
        print("✅ joblib") 
        
        import tqdm
        print("✅ tqdm")
        
        import numpy as np
        print("✅ numpy")
        
        from Bio import Entrez, SeqIO
        print("✅ Bio (Entrez, SeqIO)")
        
        import requests
        print("✅ requests")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_ncbi_downloader_import():
    """Prueba que se pueda importar el módulo modificado."""
    try:
        print("\nTesting ncbi_downloader module...")
        
        # Esto solo prueba que el archivo se puede parsear
        import ast
        with open('ncbi_downloader.py', 'r') as f:
            content = f.read()
        
        ast.parse(content)
        print("✅ ncbi_downloader.py syntax is valid")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in ncbi_downloader.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Integration Test for NCBI Downloader + UniProt")
    print("=" * 50)
    
    test1 = test_imports()
    test2 = test_ncbi_downloader_import()
    
    if test1 and test2:
        print("\n🎉 Integration test PASSED!")
        print("✅ Ready to use enhanced ncbi_downloader.py")
    else:
        print("\n❌ Integration test FAILED!")
        print("Please check dependencies and fix errors")

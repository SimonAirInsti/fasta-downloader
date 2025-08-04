#!/usr/bin/env python3
"""
Ejemplo de uso del código dinámico para descarga automática desde UniProt.
"""

# Importar la clase actualizada
from in_db_maker import protDB

def ejemplo_basico():
    """Ejemplo básico con parámetros por defecto."""
    print("=" * 60)
    print("🧪 EJEMPLO 1: Configuración por defecto")
    print("=" * 60)
    
    # Usar configuración por defecto (Influenza A virus)
    db = protDB()
    
    # Guardar resultados
    db.save_results("output_ejemplo1")
    
    print(f"✅ Proteínas procesadas: {len(db.target_proteins)}")

def ejemplo_personalizado():
    """Ejemplo con virus y proteínas personalizadas."""
    print("=" * 60)
    print("🧪 EJEMPLO 2: Configuración personalizada")
    print("=" * 60)
    
    # Parámetros personalizados
    virus_personalizado = "Influenza A virus"
    proteinas_personalizadas = [
        "Hemagglutinin",
        "Neuraminidase"
    ]
    
    # Crear base de datos con parámetros específicos
    db = protDB(
        virus_name=virus_personalizado,
        target_proteins=proteinas_personalizadas
    )
    
    # Guardar resultados
    db.save_results("output_ejemplo2")
    
    print(f"✅ Virus: {virus_personalizado}")
    print(f"✅ Proteínas: {', '.join(proteinas_personalizadas)}")

def ejemplo_otros_virus():
    """Ejemplo con otros virus."""
    print("=" * 60)
    print("🧪 EJEMPLO 3: Otro virus")
    print("=" * 60)
    
    # Intentar con otro virus (puede que no tenga tantos datos)
    db = protDB(
        virus_name="Influenza B virus",
        target_proteins=["Hemagglutinin", "Neuraminidase"]
    )
    
    # Guardar resultados
    db.save_results("output_influenza_b")

if __name__ == "__main__":
    try:
        print("🚀 Iniciando ejemplos de descarga dinámica...")
        
        # Ejecutar ejemplos
        ejemplo_basico()
        print("\n")
        ejemplo_personalizado()
        print("\n")
        ejemplo_otros_virus()
        
        print("\n🎉 ¡Todos los ejemplos completados!")
        print("\n📁 Archivos generados:")
        print("  - output_ejemplo1/")
        print("  - output_ejemplo2/") 
        print("  - output_influenza_b/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de tener conexión a internet y las dependencias instaladas")

#!/usr/bin/env python3
"""
Usage example of dynamic code for automatic download from UniProt.
"""

# Import the updated class
from in_db_maker import protDB

def basic_example():
    """Basic example with default parameters."""
    print("=" * 60)
    print("🧪 EXAMPLE 1: Default configuration")
    print("=" * 60)
    
    # Use default configuration (Influenza A virus)
    db = protDB()
    
    # Save results
    db.save_results("output_example1")
    
    print(f"✅ Proteins processed: {len(db.target_proteins)}")

def custom_example():
    """Example with custom virus and proteins."""
    print("=" * 60)
    print("🧪 EXAMPLE 2: Custom configuration")
    print("=" * 60)
    
    # Custom parameters
    custom_virus = "Influenza A virus"
    custom_proteins = [
        "Hemagglutinin",
        "Neuraminidase"
    ]
    
    # Create database with specific parameters
    db = protDB(
        virus_name=custom_virus,
        target_proteins=custom_proteins
    )
    
    # Save results
    db.save_results("output_example2")
    
    print(f"✅ Virus: {custom_virus}")
    print(f"✅ Proteins: {', '.join(custom_proteins)}")

def other_virus_example():
    """Example with other viruses."""
    print("=" * 60)
    print("🧪 EXAMPLE 3: Other virus")
    print("=" * 60)
    
    # Try with another virus (may not have as much data)
    db = protDB(
        virus_name="Influenza B virus",
        target_proteins=["Hemagglutinin", "Neuraminidase"]
    )
    
    # Save results
    db.save_results("output_influenza_b")

if __name__ == "__main__":
    try:
        print("🚀 Starting dynamic download examples...")
        
        # Run examples
        basic_example()
        print("\n")
        custom_example()
        print("\n")
        other_virus_example()
        
        print("\n🎉 All examples completed!")
        print("\n📁 Generated files:")
        print("  - output_example1/")
        print("  - output_example2/") 
        print("  - output_influenza_b/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have internet connection and dependencies installed")

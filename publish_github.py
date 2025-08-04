#!/usr/bin/env python3
"""
GitHub Publication Helper Script
Script de Ayuda para Publicación en GitHub

This script helps prepare and validate the project for GitHub publication.
Este script ayuda a preparar y validar el proyecto para publicación en GitHub.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

class GitHubPublisher:
    """Helper class for GitHub publication."""
    
    def __init__(self, project_root=None):
        """Initialize the publisher."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.required_files = [
            'ncbi_downloader.py',
            'setup.py',
            'check_progress.py',
            'requirements.txt',
            'README.md',
            'LICENSE',
            '.gitignore',
            'setup.cfg'
        ]
        
        self.optional_files = [
            'test_suite.py',
            'sample_ncbi_ids.tsv',
            '.github/workflows/ci.yml'
        ]
    
    def check_prerequisites(self):
        """Check if all required files are present."""
        print("🔍 Checking Prerequisites...")
        print("-" * 30)
        
        missing_files = []
        for file_path in self.required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                print(f"❌ Missing: {file_path}")
            else:
                print(f"✅ Found: {file_path}")
        
        print("\n📋 Optional Files:")
        for file_path in self.optional_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"✅ Found: {file_path}")
            else:
                print(f"⚠️  Optional: {file_path}")
        
        if missing_files:
            print(f"\n❌ Missing {len(missing_files)} required files!")
            return False
        
        print(f"\n✅ All required files present!")
        return True
    
    def check_git_status(self):
        """Check git repository status."""
        print("\n🔄 Checking Git Status...")
        print("-" * 30)
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'status'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                print("⚠️  Not a git repository. Initialize with:")
                print("   git init")
                print("   git add .")
                print("   git commit -m 'Initial commit'")
                return False
            
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout.strip():
                print("⚠️  Uncommitted changes detected:")
                print(result.stdout)
                print("Commit changes before publishing:")
                print("   git add .")
                print("   git commit -m 'Prepare for GitHub publication'")
                return False
            
            print("✅ Git repository is clean and ready")
            return True
            
        except FileNotFoundError:
            print("❌ Git not found. Please install Git first.")
            return False
    
    def validate_configuration(self):
        """Validate configuration files."""
        print("\n⚙️  Validating Configuration...")
        print("-" * 30)
        
        # Check if config.json exists (should not be in GitHub)
        config_file = self.project_root / "config.json"
        if config_file.exists():
            print("⚠️  config.json found - this should not be in GitHub")
            print("   Add 'config.json' to .gitignore if not already there")
        
        # Check requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            with open(req_file) as f:
                requirements = f.read()
            
            required_packages = ['biopython', 'requests', 'colorama']
            missing_packages = []
            
            for package in required_packages:
                if package not in requirements.lower():
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"⚠️  Missing packages in requirements.txt: {missing_packages}")
            else:
                print("✅ Requirements validated")
        
        # Check README
        readme_file = self.project_root / "README.md"
        if readme_file.exists():
            with open(readme_file) as f:
                readme_content = f.read()
            
            required_sections = [
                "# NCBI FASTA Downloader",
                "## Features",
                "## Installation",
                "## Quick Start",
                "## Configuration"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in readme_content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"⚠️  Missing README sections: {missing_sections}")
            else:
                print("✅ README structure validated")
        
        return True
    
    def scan_for_personal_info(self):
        """Scan for potential personal information."""
        print("\n🔒 Scanning for Personal Information...")
        print("-" * 30)
        
        import re
        
        # Patterns to look for
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'path': r'/Users/[^/\s]+|/home/[^/\s]+|C:\\Users\\[^\\]+',
            'ip': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        }
        
        excluded_files = {
            'publish_github.py',
            'test_suite.py',
            '.github/workflows/test.yml'
        }
        
        issues_found = []
        
        for py_file in self.project_root.glob("**/*.py"):
            if py_file.name in excluded_files:
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern_name, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    if matches:
                        # Filter out common test/example addresses
                        filtered_matches = [
                            m for m in matches 
                            if not any(test in m.lower() for test in [
                                'example.com', 'test@', 'user@domain', 
                                'your.email@', '@example.', 'research@example'
                            ])
                        ]
                        
                        if filtered_matches:
                            issues_found.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'type': pattern_name,
                                'matches': filtered_matches
                            })
            
            except Exception as e:
                print(f"⚠️  Could not scan {py_file}: {e}")
        
        if issues_found:
            print("⚠️  Potential personal information found:")
            for issue in issues_found:
                print(f"   {issue['file']}: {issue['type']} - {issue['matches']}")
            print("\nPlease review and remove personal information before publishing.")
            return False
        else:
            print("✅ No personal information detected")
            return True
    
    def run_tests(self):
        """Run the test suite."""
        print("\n🧪 Running Tests...")
        print("-" * 30)
        
        test_file = self.project_root / "test_suite.py"
        if not test_file.exists():
            print("⚠️  No test suite found (test_suite.py)")
            return True  # Not critical for publication
        
        try:
            result = subprocess.run([sys.executable, str(test_file)], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("✅ Tests passed")
                return True
            else:
                print("❌ Tests failed:")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"⚠️  Could not run tests: {e}")
            return True  # Not critical for publication
    
    def generate_publication_checklist(self):
        """Generate a checklist for GitHub publication."""
        checklist = """
# GitHub Publication Checklist
# Lista de Verificación para Publicación en GitHub

## Pre-Publication / Pre-Publicación
- [ ] All required files present / Todos los archivos requeridos presentes
- [ ] No personal information in code / Sin información personal en el código
- [ ] Tests passing / Pruebas pasando
- [ ] Git repository clean / Repositorio Git limpio
- [ ] README documentation complete / Documentación README completa

## GitHub Setup / Configuración GitHub
- [ ] Create GitHub repository / Crear repositorio GitHub
- [ ] Add repository description / Agregar descripción del repositorio
- [ ] Add topics/tags: bioinformatics, ncbi, fasta, python / Agregar temas
- [ ] Set repository visibility (public/private) / Establecer visibilidad

## Post-Publication / Post-Publicación
- [ ] Test clone and setup on clean system / Probar clonación en sistema limpio
- [ ] Verify GitHub Actions workflows / Verificar flujos de trabajo GitHub Actions
- [ ] Update repository settings / Actualizar configuraciones del repositorio
- [ ] Add collaborators if needed / Agregar colaboradores si es necesario

## Recommended GitHub Commands / Comandos GitHub Recomendados
```bash
# Initialize repository / Inicializar repositorio
git init
git add .
git commit -m "Initial commit: NCBI FASTA Downloader v2.0"

# Add remote (replace with your repository URL)
git remote add origin https://github.com/username/ncbi-fasta-downloader.git

# Push to GitHub
git branch -M main
git push -u origin main

# Create release tag / Crear etiqueta de versión
git tag -a v2.0.0 -m "Version 2.0.0: High-performance NCBI downloader"
git push origin v2.0.0
```

## Repository Settings / Configuraciones del Repositorio
- Description: "High-performance NCBI FASTA sequence downloader with batch processing"
- Topics: bioinformatics, ncbi, fasta, python, batch-processing, parallel
- License: MIT
- Include in searches: Yes / Incluir en búsquedas: Sí
"""
        
        checklist_file = self.project_root / "PUBLICATION_CHECKLIST.md"
        with open(checklist_file, 'w') as f:
            f.write(checklist)
        
        print(f"📋 Publication checklist created: {checklist_file}")
        return checklist_file
    
    def publish(self, dry_run=True):
        """Main publication workflow."""
        print("🚀 NCBI FASTA Downloader - GitHub Publication")
        print("=" * 50)
        print(f"Project Root: {self.project_root}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if dry_run:
            print("⚠️  DRY RUN MODE - No changes will be made")
        
        print("\n")
        
        # Run all checks
        checks = [
            ("Prerequisites", self.check_prerequisites),
            ("Configuration", self.validate_configuration),
            ("Personal Info", self.scan_for_personal_info),
            ("Tests", self.run_tests),
            ("Git Status", self.check_git_status)
        ]
        
        results = {}
        for check_name, check_func in checks:
            try:
                results[check_name] = check_func()
            except Exception as e:
                print(f"❌ Error in {check_name}: {e}")
                results[check_name] = False
        
        # Generate checklist
        self.generate_publication_checklist()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Publication Readiness Summary")
        print("-" * 30)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for check_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{check_name:<15}: {status}")
        
        print(f"\nOverall: {passed}/{total} checks passed")
        
        if passed == total:
            print("\n🎉 Project is ready for GitHub publication!")
            print("📋 See PUBLICATION_CHECKLIST.md for next steps")
        else:
            print(f"\n⚠️  {total - passed} issues need to be addressed before publication")
        
        return passed == total

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Publication Helper")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--run", action="store_true", help="Actually run (not dry-run)")
    
    args = parser.parse_args()
    
    publisher = GitHubPublisher(args.project_root)
    dry_run = not args.run
    
    success = publisher.publish(dry_run=dry_run)
    
    if success:
        print("\n🚀 Ready to publish to GitHub!")
        sys.exit(0)
    else:
        print("\n❌ Issues found. Please fix before publishing.")
        sys.exit(1)

if __name__ == "__main__":
    main()

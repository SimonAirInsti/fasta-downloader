#!/usr/bin/env python3
"""
Advanced progress checker with enhanced statistics and visualization.
Verificador de progreso avanzado con estadísticas mejoradas y visualización.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import argparse

def format_size(bytes_size):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def format_duration(seconds):
    """Format seconds to human readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"

def check_progress_enhanced(output_dir="output", show_details=False):
    """Enhanced progress checker with detailed statistics."""
    
    output_path = Path(output_dir)
    progress_file = output_path / "progress.json"
    output_file = output_path / "all_sequences.fasta"
    log_file = output_path / "ncbi_download.log"
    failed_file = output_path / "failed_ids.txt"
    
    print("🔍 NCBI Download Progress Check - Enhanced")
    print("=" * 50)
    print()
    
    # Check if download has started
    if not progress_file.exists() and not output_file.exists():
        print("❌ No download progress found.")
        print(f"Expected files in: {output_path.absolute()}")
        print()
        print("To start:")
        print("  1. Run: python setup.py (if not configured)")
        print("  2. Run: python ncbi_downloader_v2.py")
        return False
    
    # Load progress data
    progress_data = {}
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not read progress file: {e}")
    
    # Count sequences in output file
    sequence_count = 0
    file_size = 0
    if output_file.exists():
        try:
            file_size = output_file.stat().st_size
            with open(output_file, 'r') as f:
                sequence_count = sum(1 for line in f if line.startswith('>'))
        except Exception as e:
            print(f"⚠️  Warning: Could not read output file: {e}")
    
    # Count failed IDs
    failed_count = 0
    failed_by_type = {}
    if failed_file.exists():
        try:
            with open(failed_file, 'r') as f:
                for line in f:
                    if line.strip():
                        failed_count += 1
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            error_type = parts[2] if len(parts) > 2 else "Unknown"
                            failed_by_type[error_type] = failed_by_type.get(error_type, 0) + 1
        except Exception as e:
            print(f"⚠️  Warning: Could not read failed IDs file: {e}")
    
    # Display main statistics
    print("📊 MAIN STATISTICS:")
    print("-" * 30)
    
    if progress_data:
        processed = progress_data.get('processed', 0)
        successful = progress_data.get('successful', sequence_count)
        failed = progress_data.get('failed', failed_count)
        bytes_downloaded = progress_data.get('bytes_downloaded', file_size)
        timestamp = progress_data.get('timestamp', 'Unknown')
        session_start = progress_data.get('session_start', None)
        
        print(f"📈 Total processed: {processed:,}")
        print(f"✅ Successfully downloaded: {successful:,}")
        print(f"❌ Failed downloads: {failed:,}")
        print(f"📁 Output file size: {format_size(file_size)}")
        print(f"💾 Data downloaded: {format_size(bytes_downloaded)}")
        
        if processed > 0:
            success_rate = (successful / processed) * 100
            print(f"🎯 Success rate: {success_rate:.1f}%")
        
        # Time information
        if timestamp != 'Unknown':
            try:
                last_update = datetime.fromisoformat(timestamp)
                time_since_update = datetime.now() - last_update
                print(f"🕐 Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"⏱️  Time since update: {str(time_since_update).split('.')[0]}")
                
                # Session duration
                if session_start:
                    session_duration = last_update.timestamp() - session_start
                    print(f"⌛ Session duration: {format_duration(session_duration)}")
                    
                    if successful > 0 and session_duration > 0:
                        rate = successful / (session_duration / 60)  # per minute
                        print(f"🚀 Average rate: {rate:.1f} sequences/minute")
                        
            except Exception as e:
                print(f"⚠️  Could not parse timestamp: {e}")
    else:
        print(f"✅ Sequences in file: {sequence_count:,}")
        print(f"📁 File size: {format_size(file_size)}")
    
    print()
    
    # Per-protein statistics
    if progress_data and 'sequences_per_protein' in progress_data:
        print("🧬 PER-PROTEIN STATISTICS:")
        print("-" * 35)
        sequences_per_protein = progress_data['sequences_per_protein']
        total_by_protein = sum(sequences_per_protein.values())
        
        for protein_type, count in sorted(sequences_per_protein.items()):
            percentage = (count / total_by_protein) * 100 if total_by_protein > 0 else 0
            print(f"  {protein_type}: {count:,} ({percentage:.1f}%)")
        print()
    
    # Failed downloads breakdown
    if failed_by_type and show_details:
        print("💥 FAILURE ANALYSIS:")
        print("-" * 25)
        for error_type, count in sorted(failed_by_type.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / failed_count) * 100 if failed_count > 0 else 0
            print(f"  {error_type}: {count} ({percentage:.1f}%)")
        print()
    
    # File status
    print("📁 FILE STATUS:")
    print("-" * 20)
    files_to_check = [
        ("Configuration", "config.json"),
        ("Progress file", progress_file),
        ("Output file", output_file),
        ("Failed IDs", failed_file),
        ("Log file", log_file)
    ]
    
    for file_desc, file_path in files_to_check:
        file_path = Path(file_path)
        if file_path.exists():
            size = file_path.stat().st_size
            size_str = format_size(size)
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"✅ {file_desc}: {size_str} (modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"❌ {file_desc}: Not found")
    
    print()
    
    # Recent log entries
    if log_file.exists() and show_details:
        print("📝 RECENT LOG ENTRIES:")
        print("-" * 25)
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Show last 10 lines
                for line in lines[-10:]:
                    # Clean up the line
                    clean_line = line.strip()
                    if clean_line:
                        # Truncate very long lines
                        if len(clean_line) > 100:
                            clean_line = clean_line[:97] + "..."
                        print(f"   {clean_line}")
        except Exception as e:
            print(f"Could not read log file: {e}")
        print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-" * 25)
    
    if not progress_file.exists() and not output_file.exists():
        print("🚀 Start the download with: python ncbi_downloader_v2.py")
    elif progress_data and progress_data.get('processed', 0) > 0:
        if progress_data.get('processed', 0) < 100000:  # Arbitrary "completion" threshold
            print("▶️  Continue the download with: python ncbi_downloader_v2.py")
            print("🔍 Monitor progress with: python check_progress_v2.py")
            print("📊 Detailed view: python check_progress_v2.py --details")
        else:
            print("🎉 Download appears to be substantial!")
            print("🔍 Check failed_ids.txt for any sequences that need manual retry")
            print("📊 View details with: python check_progress_v2.py --details")
    else:
        print("🔄 Download in progress. Monitor with: python check_progress_v2.py")
        print("📊 Check logs for any issues: tail -f output/ncbi_download.log")
    
    return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced progress checker for NCBI FASTA Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output',
        help='Output directory to check (default: output)'
    )
    
    parser.add_argument(
        '--details', '-d',
        action='store_true',
        help='Show detailed information including recent logs and failure analysis'
    )
    
    parser.add_argument(
        '--watch', '-w',
        type=int,
        metavar='SECONDS',
        help='Watch mode: refresh every N seconds'
    )
    
    args = parser.parse_args()
    
    if args.watch:
        import time
        import os
        
        try:
            while True:
                # Clear screen (works on most terminals)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                check_progress_enhanced(args.output_dir, args.details)
                
                print(f"\n🔄 Refreshing every {args.watch} seconds... (Ctrl+C to stop)")
                time.sleep(args.watch)
                
        except KeyboardInterrupt:
            print("\n👋 Watch mode stopped.")
    else:
        check_progress_enhanced(args.output_dir, args.details)

if __name__ == "__main__":
    main()

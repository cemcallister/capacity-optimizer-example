"""
Standard Output Manager
======================

This module provides a standardized interface for all optimization approaches
to output their results to a consistent location, making validation and 
comparison much more maintainable.

All optimization models should use this manager to:
1. Write qualification matrices to standard locations
2. Include metadata about which optimization was used
3. Archive previous results for comparison
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add logging and error handling
from ..utils.logger import get_logger, log_function_entry, log_function_exit, log_data_summary
from ..utils.exceptions import FileOperationError


class StandardOutputManager:
    """Manages standardized output for all optimization approaches"""
    
    def __init__(self, base_dir="outputs"):
        self.logger = get_logger(__name__)
        log_function_entry(self.logger, "__init__", base_dir=base_dir)
        
        self.base_dir = Path(base_dir)
        self.current_dir = self.base_dir / "current"
        self.archive_dir = self.base_dir / "archive"
        
        try:
            # Create required directories
            self.current_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directories initialized: {self.current_dir}")
            # Note: archive_dir is only created when actually needed for archiving
        except OSError as e:
            raise FileOperationError(
                f"Failed to create output directories: {e}",
                file_path=str(self.current_dir),
                operation="create_directory"
            )
    
    def save_optimization_results(self, 
                                qualification_matrices: Dict[int, Dict],
                                optimization_name: str,
                                optimization_config: Optional[Dict] = None,
                                validation_results: Optional[Dict] = None,
                                archive_previous: bool = False):
        """
        Save optimization results to standard location
        
        Args:
            qualification_matrices: Dict of {team: {engineer_id: assignment_data}}
            optimization_name: Name of the optimization approach (e.g., "coverage_optimized")
            optimization_config: Optional configuration used for this optimization
            validation_results: Optional validation results if already computed
            archive_previous: Whether to archive previous results (default: False)
        """
        log_function_entry(self.logger, "save_optimization_results", 
                          optimization_name=optimization_name,
                          archive_previous=archive_previous)
        self.logger.info(f"Saving {optimization_name} optimization results")
        
        print(f"\n💾 SAVING RESULTS TO STANDARD LOCATION")
        print(f"   Optimization: {optimization_name}")
        
        # Archive current results if they exist and archiving is requested
        if archive_previous:
            self._archive_current_results(optimization_name)
        elif (self.current_dir / "metadata.json").exists():
            print(f"   🔄 Overwriting previous results (no archiving)")
        
        # Skip JSON files - only generating CSV files for dashboard
        print(f"   ⏭️  Skipping JSON files (only CSVs needed for dashboard)")
        
        print(f"   ✅ Standard output saved to: {self.current_dir}")
    
    def load_current_matrices(self) -> Optional[Dict[int, Dict]]:
        """Load the current qualification matrices"""
        matrices = {}
        
        for team in [1, 2]:
            matrix_file = self.current_dir / f"team_{team}_qualification_matrix.json"
            if matrix_file.exists():
                with open(matrix_file, 'r') as f:
                    matrices[team] = json.load(f)
        
        return matrices if matrices else None
    
    def load_current_metadata(self) -> Optional[Dict]:
        """Load metadata about current results"""
        metadata_file = self.current_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def load_current_validation(self) -> Optional[Dict]:
        """Load current validation results if available"""
        validation_file = self.current_dir / "validation_results.json"
        if validation_file.exists():
            with open(validation_file, 'r') as f:
                return json.load(f)
        return None
    
    def _archive_current_results(self, new_optimization_name: str):
        """Archive current results before overwriting"""
        # Check if current results exist
        if not (self.current_dir / "metadata.json").exists():
            return  # Nothing to archive
        
        # Load current metadata to get the optimization name
        try:
            current_metadata = self.load_current_metadata()
            if current_metadata:
                old_optimization_name = current_metadata.get('optimization_name', 'unknown')
                timestamp = current_metadata.get('created_timestamp', 'unknown_time')
            else:
                old_optimization_name = 'unknown'
                timestamp = 'unknown_time'
        except:
            old_optimization_name = 'unknown'
            timestamp = 'unknown_time'
        
        # Create archive directory for this optimization
        self.archive_dir.mkdir(parents=True, exist_ok=True)  # Create archive dir when needed
        archive_subdir = self.archive_dir / f"{old_optimization_name}_{timestamp.replace(':', '-')}"
        archive_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy current files to archive
        files_to_archive = [
            "team_1_qualification_matrix.json",
            "team_2_qualification_matrix.json", 
            "metadata.json",
            "validation_results.json"
        ]
        
        archived_count = 0
        for filename in files_to_archive:
            source_file = self.current_dir / filename
            if source_file.exists():
                target_file = archive_subdir / filename
                shutil.copy2(source_file, target_file)
                archived_count += 1
        
        if archived_count > 0:
            print(f"   📦 Archived {archived_count} files from '{old_optimization_name}' to: {archive_subdir}")
    
    def list_archive(self) -> list:
        """List all archived optimization results"""
        if not self.archive_dir.exists():
            return []
        
        archives = []
        for subdir in self.archive_dir.iterdir():
            if subdir.is_dir():
                metadata_file = subdir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        archives.append({
                            'directory': subdir.name,
                            'optimization_name': metadata.get('optimization_name', 'unknown'),
                            'timestamp': metadata.get('created_timestamp', 'unknown'),
                            'teams': metadata.get('teams_included', [])
                        })
                    except:
                        archives.append({
                            'directory': subdir.name,
                            'optimization_name': 'unknown',
                            'timestamp': 'unknown',
                            'teams': []
                        })
        
        return sorted(archives, key=lambda x: x['timestamp'], reverse=True)
    
    def restore_from_archive(self, archive_directory: str):
        """Restore results from archive to current"""
        archive_path = self.archive_dir / archive_directory
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive directory not found: {archive_path}")
        
        # Archive current results first
        current_metadata = self.load_current_metadata()
        if current_metadata:
            self._archive_current_results("before_restore")
        
        # Copy archived files to current
        files_to_restore = [
            "team_1_qualification_matrix.json",
            "team_2_qualification_matrix.json",
            "metadata.json",
            "validation_results.json"
        ]
        
        restored_count = 0
        for filename in files_to_restore:
            source_file = archive_path / filename
            if source_file.exists():
                target_file = self.current_dir / filename
                shutil.copy2(source_file, target_file)
                restored_count += 1
        
        print(f"   🔄 Restored {restored_count} files from archive: {archive_directory}")
    
    def get_standard_paths(self) -> Dict[str, Path]:
        """Get the standard file paths for current optimization"""
        return {
            'team_1_matrix': self.current_dir / "team_1_qualification_matrix.json",
            'team_2_matrix': self.current_dir / "team_2_qualification_matrix.json",
            'metadata': self.current_dir / "metadata.json",
            'validation': self.current_dir / "validation_results.json",
            'current_dir': self.current_dir,
            'archive_dir': self.archive_dir
        }


def main():
    """Example usage of StandardOutputManager"""
    manager = StandardOutputManager()
    
    print("📁 STANDARD OUTPUT MANAGER")
    print("=" * 50)
    
    # Show current status
    metadata = manager.load_current_metadata()
    if metadata:
        print(f"\n📋 CURRENT RESULTS:")
        print(f"   Optimization: {metadata['optimization_name']}")
        print(f"   Created: {metadata['created_timestamp']}")
        print(f"   Teams: {metadata['teams_included']}")
    else:
        print(f"\n📋 No current results found")
    
    # Show archives
    archives = manager.list_archive()
    if archives:
        print(f"\n📦 ARCHIVED RESULTS ({len(archives)} found):")
        for archive in archives[:5]:  # Show last 5
            print(f"   {archive['optimization_name']} - {archive['timestamp']}")
    else:
        print(f"\n📦 No archived results found")
    
    # Show standard paths
    paths = manager.get_standard_paths()
    print(f"\n📂 STANDARD PATHS:")
    print(f"   Current: {paths['current_dir']}")
    print(f"   Archive: {paths['archive_dir']}")


if __name__ == "__main__":
    main() 
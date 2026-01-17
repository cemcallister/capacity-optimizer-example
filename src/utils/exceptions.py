"""
Custom exception classes for the capacity optimization system.

This module provides structured exception handling with:
- Base exception class for all capacity optimizer errors
- Specific exception types for different failure modes
- Consistent error messaging and context preservation
"""

from typing import Optional, Dict, Any


class CapacityOptimizerError(Exception):
    """
    Base exception for all capacity optimizer errors.
    
    Provides consistent error handling and context preservation
    across the entire optimization system.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the base exception.
        
        Args:
            message: Human-readable error description
            error_code: Machine-readable error code for programmatic handling
            context: Additional context information for debugging
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        if self.context:
            parts.append(f"Context: {self.context}")
        return " | ".join(parts)


class DataValidationError(CapacityOptimizerError):
    """
    Raised when data validation fails.
    
    Used for:
    - Missing required data files
    - Invalid data formats or structures
    - Data consistency checks failing
    - Constraint violations in input data
    """
    
    def __init__(self, message: str, data_source: Optional[str] = None, validation_type: Optional[str] = None, **kwargs):
        context = {
            "data_source": data_source,
            "validation_type": validation_type,
            **kwargs
        }
        super().__init__(message, error_code="DATA_VALIDATION", context=context)


class OptimizationError(CapacityOptimizerError):
    """
    Raised when optimization algorithms fail.
    
    Used for:
    - MILP solver failures
    - Infeasible optimization problems  
    - Training algorithm convergence issues
    - Mathematical constraint violations
    """
    
    def __init__(self, message: str, algorithm: Optional[str] = None, solver_status: Optional[str] = None, **kwargs):
        context = {
            "algorithm": algorithm,
            "solver_status": solver_status,
            **kwargs
        }
        super().__init__(message, error_code="OPTIMIZATION", context=context)


class PPMCoverageError(CapacityOptimizerError):
    """
    Raised when PPM coverage validation fails.
    
    Used for:
    - Insufficient qualification coverage
    - PPM schedule conflicts
    - Coverage gap detection
    - Shift pattern violations
    """
    
    def __init__(self, message: str, team: Optional[int] = None, ride_code: Optional[str] = None, coverage_gap: Optional[str] = None, **kwargs):
        context = {
            "team": team,
            "ride_code": ride_code,
            "coverage_gap": coverage_gap,
            **kwargs
        }
        super().__init__(message, error_code="PPM_COVERAGE", context=context)


class ConfigurationError(CapacityOptimizerError):
    """
    Raised when configuration issues are detected.
    
    Used for:
    - Missing configuration files
    - Invalid configuration parameters
    - Environment setup problems
    - Dependency issues
    """
    
    def __init__(self, message: str, config_file: Optional[str] = None, parameter: Optional[str] = None, **kwargs):
        context = {
            "config_file": config_file,
            "parameter": parameter,
            **kwargs
        }
        super().__init__(message, error_code="CONFIGURATION", context=context)


class FileOperationError(CapacityOptimizerError):
    """
    Raised when file operations fail.
    
    Used for:
    - File read/write failures
    - Missing input files
    - Output directory issues
    - Backup operation failures
    """
    
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None, **kwargs):
        context = {
            "file_path": file_path,
            "operation": operation,
            **kwargs
        }
        super().__init__(message, error_code="FILE_OPERATION", context=context)


class DatabaseConnectionError(CapacityOptimizerError):
    """
    Raised when Databricks/database operations fail.
    
    Used for:
    - Connection failures
    - Authentication issues
    - Query execution problems
    - Data upload/download failures
    """
    
    def __init__(self, message: str, connection_type: Optional[str] = None, endpoint: Optional[str] = None, **kwargs):
        context = {
            "connection_type": connection_type,
            "endpoint": endpoint,
            **kwargs
        }
        super().__init__(message, error_code="DATABASE_CONNECTION", context=context)


# Utility functions for exception handling

def handle_file_operation(operation_name: str, file_path: str):
    """
    Decorator factory for handling file operations with consistent error handling.
    
    Args:
        operation_name: Description of the file operation
        file_path: Path to the file being operated on
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                raise FileOperationError(
                    f"File not found during {operation_name}",
                    file_path=file_path,
                    operation=operation_name,
                    original_error=str(e)
                )
            except PermissionError as e:
                raise FileOperationError(
                    f"Permission denied during {operation_name}",
                    file_path=file_path,
                    operation=operation_name,
                    original_error=str(e)
                )
            except OSError as e:
                raise FileOperationError(
                    f"OS error during {operation_name}",
                    file_path=file_path,
                    operation=operation_name,
                    original_error=str(e)
                )
        return wrapper
    return decorator


def create_error_context(**kwargs) -> Dict[str, Any]:
    """
    Create standardized error context dictionary.
    
    Args:
        **kwargs: Key-value pairs for context
        
    Returns:
        Dictionary with non-None values for error context
    """
    return {k: v for k, v in kwargs.items() if v is not None}
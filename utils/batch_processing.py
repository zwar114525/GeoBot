"""
Batch processing for document ingestion with progress tracking.
"""
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    status: str = "pending"  # pending, running, completed, failed
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class FileResult:
    """Result of processing a single file."""
    file_path: str
    success: bool
    chunks_created: int = 0
    error: Optional[str] = None
    processing_time_ms: float = 0


class BatchProcessor:
    """Process multiple documents with progress tracking."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum concurrent workers
        """
        self.max_workers = max_workers
        self.jobs: Dict[str, BatchJob] = {}
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def _notify_progress(self, job_id: str, progress: Dict):
        """Notify progress callback."""
        if self._progress_callback:
            self._progress_callback(job_id, progress)
    
    def create_job(self, file_paths: List[str]) -> str:
        """
        Create a new batch job.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Job ID
        """
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.jobs[job_id] = BatchJob(
            job_id=job_id,
            total_files=len(file_paths),
        )
        
        logger.info(f"Created batch job {job_id} with {len(file_paths)} files")
        return job_id
    
    def process_files(
        self,
        job_id: str,
        process_fn: Callable,
        directory: str = None,
        pattern: str = "*.pdf",
    ) -> BatchJob:
        """
        Process files in batch.
        
        Args:
            job_id: Job ID
            process_fn: Function to process each file
            directory: Directory to scan (alternative to file_paths)
            pattern: File pattern if using directory
            
        Returns:
            BatchJob with results
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        job.status = "running"
        job.started_at = datetime.now().isoformat()
        
        # Get file list
        if directory:
            dir_path = Path(directory)
            file_paths = [str(f) for f in dir_path.glob(pattern)]
            job.total_files = len(file_paths)
        else:
            file_paths = [r.file_path for r in job.results] if job.results else []
        
        logger.info(f"Starting batch processing for job {job_id}")
        
        # Process files
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_file, file_path, process_fn): file_path
                for file_path in file_paths
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    job.results.append({
                        "file_path": result.file_path,
                        "success": result.success,
                        "chunks_created": result.chunks_created,
                        "error": result.error,
                        "processing_time_ms": result.processing_time_ms,
                    })
                    
                    if result.success:
                        job.processed_files += 1
                    else:
                        job.failed_files += 1
                        job.errors.append(f"{file_path}: {result.error}")
                    
                    # Notify progress
                    self._notify_progress(job_id, {
                        "processed": job.processed_files,
                        "failed": job.failed_files,
                        "total": job.total_files,
                        "percent": (job.processed_files / job.total_files * 100) if job.total_files > 0 else 0,
                    })
                    
                except Exception as e:
                    job.failed_files += 1
                    job.errors.append(f"{file_path}: {str(e)}")
                    logger.error(f"Failed to process {file_path}: {e}")
        
        job.completed_at = datetime.now().isoformat()
        job.status = "completed" if job.failed_files == 0 else "completed_with_errors"
        
        logger.info(f"Batch job {job_id} completed: {job.processed_files}/{job.total_files} successful")
        
        return job
    
    def _process_single_file(self, file_path: str, process_fn: Callable) -> FileResult:
        """Process a single file."""
        start_time = time.time()
        
        try:
            chunks_created = process_fn(file_path)
            
            return FileResult(
                file_path=file_path,
                success=True,
                chunks_created=chunks_created,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            return FileResult(
                file_path=file_path,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status and progress."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        return {
            "job_id": job.job_id,
            "status": job.status,
            "total_files": job.total_files,
            "processed_files": job.processed_files,
            "failed_files": job.failed_files,
            "progress_percent": (job.processed_files / job.total_files * 100) if job.total_files > 0 else 0,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "errors": job.errors[:10],  # Limit errors in status
        }
    
    def get_job_results(self, job_id: str) -> Optional[BatchJob]:
        """Get complete job results."""
        return self.jobs.get(job_id)


# Streamlit progress helper
class StreamlitProgress:
    """Helper for Streamlit progress bars."""
    
    def __init__(self, st, container=None):
        self.st = st
        self.container = container or st
        self.progress_bar = None
        self.status_text = None
    
    def start(self, job_id: str):
        """Initialize progress display."""
        self.progress_bar = self.container.progress(0)
        self.status_text = self.container.empty()
    
    def update(self, job_id: str, progress: Dict):
        """Update progress display."""
        if self.progress_bar:
            self.progress_bar.progress(int(progress["percent"]))
        if self.status_text:
            self.status_text.text(
                f"Processing: {progress['processed']}/{progress['total']} "
                f"({progress['percent']:.1f}%) - {progress['failed']} failed"
            )
    
    def complete(self, job_id: str, job: BatchJob):
        """Complete progress display."""
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        if self.status_text:
            status = "Completed successfully" if job.failed_files == 0 else f"Completed with {job.failed_files} errors"
            self.status_text.text(status)


def create_batch_processor(max_workers: int = 4) -> BatchProcessor:
    """Factory function to create batch processor."""
    return BatchProcessor(max_workers=max_workers)

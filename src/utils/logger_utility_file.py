"""
Logging utilities for AHP-PSO Object Detection
Provides comprehensive logging functionality for experiments
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    log_file: Optional[Path] = None,
    name: str = 'ahp_pso',
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        log_file: Path to log file (optional)
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file is not None:
        # Create parent directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file created at: {log_file}")
    
    return logger


def get_logger(name: str = 'ahp_pso') -> logging.Logger:
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Advanced logger for experiment tracking
    Tracks metrics, timings, and experiment metadata
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: Path,
        log_level: int = logging.INFO
    ):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory for logs
            log_level: Logging level
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{experiment_name}_{timestamp}.log"
        self.log_file = self.output_dir / log_filename
        
        # Setup logger
        self.logger = setup_logger(
            log_file=self.log_file,
            name=experiment_name,
            level=log_level
        )
        
        # Experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': timestamp,
            'log_file': str(self.log_file)
        }
        
        self.logger.info(f"Experiment '{experiment_name}' started")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def log_config(self, config: dict):
        """
        Log experiment configuration
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("="*60)
        self.logger.info("EXPERIMENT CONFIGURATION")
        self.logger.info("="*60)
        
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        
        self.metadata['config'] = config
    
    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log performance metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step/iteration number
        """
        step_str = f"[Step {step}] " if step is not None else ""
        
        self.logger.info(f"{step_str}Metrics:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric_name}: {value:.4f}")
            else:
                self.logger.info(f"  {metric_name}: {value}")
    
    def log_progress(self, current: int, total: int, message: str = ""):
        """
        Log progress information
        
        Args:
            current: Current iteration/step
            total: Total iterations/steps
            message: Additional message
        """
        percentage = (current / total) * 100 if total > 0 else 0
        progress_msg = f"Progress: {current}/{total} ({percentage:.1f}%)"
        
        if message:
            progress_msg += f" - {message}"
        
        self.logger.info(progress_msg)
    
    def log_error(self, error: Exception, context: str = ""):
        """
        Log error with context
        
        Args:
            error: Exception object
            context: Additional context information
        """
        error_msg = f"ERROR in {context}: {type(error).__name__}: {str(error)}"
        self.logger.error(error_msg)
        
        # Log traceback
        import traceback
        self.logger.error(traceback.format_exc())
    
    def log_results(self, results: dict):
        """
        Log final results
        
        Args:
            results: Dictionary of final results
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL RESULTS")
        self.logger.info("="*60)
        
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")
        
        self.metadata['results'] = results
    
    def finalize(self):
        """Finalize experiment logging"""
        end_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metadata['end_time'] = end_time
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"Experiment '{self.experiment_name}' completed")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("="*60)
        
        # Save metadata
        import json
        metadata_file = self.output_dir / 'experiment_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)


class MetricLogger:
    """
    Logger specifically for tracking metrics over time
    Useful for convergence tracking and performance monitoring
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize metric logger
        
        Args:
            output_dir: Directory for metric logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = {}
        self.logger = setup_logger(name='metric_logger')
    
    def log_metric(self, metric_name: str, value: float, step: int = None):
        """
        Log a single metric value
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Optional step/iteration number
        """
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        entry = {'value': value}
        if step is not None:
            entry['step'] = step
        
        self.metrics_history[metric_name].append(entry)
    
    def log_metrics_dict(self, metrics: dict, step: int = None):
        """
        Log multiple metrics at once
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step/iteration number
        """
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(metric_name, value, step)
    
    def save_metrics(self, filename: str = 'metrics_history.json'):
        """
        Save metrics history to JSON file
        
        Args:
            filename: Output filename
        """
        import json
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        
        self.logger.info(f"Metrics saved to {output_path}")
    
    def get_metric_history(self, metric_name: str) -> list:
        """
        Get history of a specific metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values
        """
        return self.metrics_history.get(metric_name, [])


# Convenience function for quick logging setup
def quick_logger(name: str = 'ahp_pso', log_dir: str = 'logs') -> logging.Logger:
    """
    Quick logger setup with default configuration
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"{name}_{timestamp}.log"
    
    return setup_logger(log_file=log_file, name=name)


# Example usage
if __name__ == '__main__':
    # Basic logger
    logger = setup_logger(log_file=Path('logs/test.log'))
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Experiment logger
    exp_logger = ExperimentLogger(
        experiment_name='test_experiment',
        output_dir=Path('logs/experiments')
    )
    
    exp_logger.log_config({
        'algorithm': 'ahp_pso',
        'domain': 'aerial',
        'num_particles': 50
    })
    
    exp_logger.log_metrics({
        'precision': 0.92,
        'recall': 0.89,
        'f1_score': 0.90
    })
    
    exp_logger.finalize()
    
    # Metric logger
    metric_logger = MetricLogger(output_dir=Path('logs/metrics'))
    for i in range(100):
        metric_logger.log_metric('loss', 1.0 / (i + 1), step=i)
    
    metric_logger.save_metrics()

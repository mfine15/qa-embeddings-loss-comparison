#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to automate running experiments on the dev machine.
Handles git operations, pod execution, and running the experiment.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from kubernetes import config
from kubernetes.client import CoreV1Api
from kubernetes.stream import stream


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd: list, check: bool = True) -> Optional[str]:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"Error: {e.stderr}")
        if check:
            raise
        return None

def git_commit_and_push():
    """Commit and push changes to git."""
    logger.info("Checking for changes...")
    
    # Check if there are any changes
    status = run_command(["git", "status", "--porcelain"], check=False)
    if not status:
        logger.info("No changes to commit")
        return
    
    logger.info("Committing and pushing changes...")
    
    # Add all changes
    run_command(["git", "add", "."])
    
    # Commit with timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    run_command(["git", "commit", "-m", f"Auto commit: {timestamp}"])
    
    # Push
    run_command(["git", "push"])
    logger.info("Git operations completed successfully")

def run_on_dev_machine(config_file: str):
    """Run the experiment on the dev machine using Kubernetes API."""
    logger.info(f"Running experiment with config: {config_file}")
    
    # Load kubernetes configuration
    try:
        config.load_kube_config()
    except Exception as e:
        logger.error(f"Failed to load kubernetes config: {e}")
        raise
    
    # Create API client
    v1 = CoreV1Api()
    
    # Construct the command to run in the pod
    command = [
        "bash",
        "-c",
        f"cd /home/qa-embeddings-loss-comparison && "
        f"git pull && "
        f"uv run rank-test --config-file {config_file}"
    ]
    
    try:
        # Execute command in pod with streaming
        resp = stream(
            v1.connect_get_namespaced_pod_exec,
            "dev-machine-0",
            "dev",
            command=command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False  # Don't preload content to enable streaming
        )
        
        # Stream the output in real-time
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                print(resp.read_stdout(), end='')
            if resp.peek_stderr():
                print(resp.read_stderr(), end='', file=sys.stderr)
        
        # Close the stream
        resp.close()
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to execute command in pod: {e}")
        raise

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <config_file>")
        print("Example: python run_experiment.py configs/multiple_positives_rank.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Validate config file exists
    if not Path(config_file).exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)
    
    try:
        # Step 1: Commit and push changes
        git_commit_and_push()
        
        # Step 2: Run on dev machine
        run_on_dev_machine(config_file)
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
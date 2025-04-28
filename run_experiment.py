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
    """Run the experiment on the dev machine."""
    logger.info(f"Running experiment with config: {config_file}")
    
    
    # Construct the command to run in the pod
    pod_cmd = [
        "kubectl", "exec", "--namespace=dev", "-it", "pod/dev-machine-0", "--", "bash",
        "-c",
        f"cd /home/qa-embeddings-loss-comparison && "
        f"git pull && "
        f"uv run rank-test --config-file {config_file}"
    ]
    
    # Run the command and stream output
    process = subprocess.Popen(
        pod_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output in real-time
    for line in process.stdout or []:
        processed_line = line.rstrip() # Remove trailing newline/carriage return/spaces
        if processed_line: # Avoid printing empty lines
            print(f"[REMOTE] {processed_line}")
    
    # Wait for process to complete
    return_code = process.wait()
    
    if return_code != 0:
        logger.error(f"Experiment failed with return code: {return_code}")
        sys.exit(return_code)
    
    logger.info("Experiment completed successfully")

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
#!/usr/bin/env python
import cProfile
import pstats
import io
from rank_test.train import cli

# Run the CLI with profiling
profiler = cProfile.Profile()
profiler.enable()

# Run the training process
cli()

# Disable profiler and print stats
profiler.disable()
s = io.StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
stats.print_stats(30)  # Print top 30 time-consuming functions
print(s.getvalue())

# Also save detailed stats to a file
with open('profile_results.txt', 'w') as f:
    stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
    stats.print_stats(100)  # Print top 100 time-consuming functions
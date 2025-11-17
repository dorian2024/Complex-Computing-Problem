#!/bin/bash

# Clean and rebuild the GPU example
make clean
make example3-gpu

# Remove old profiling files
rm -f example3_report.*

# Run the profiler and generate a new report
nsys profile -o example3_report ./example3

# Generate readable text stats from the report
nsys stats example3_report.nsys-rep > example3_report.txt

# Optional: print confirmation and show brief output
echo "Profiling complete. Summary saved to example3_report.txt"
echo "Preview of report:"
head -n 20 example3_report.txt


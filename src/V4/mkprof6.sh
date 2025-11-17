#!/bin/bash

# Clean and rebuild the GPU example
make clean
make example6-gpu

# Remove old profiling files
rm -f example6_report.*

# Run the profiler and generate a new report
nsys profile -o example6_report ./example6

# Generate readable text stats from the report
nsys stats example6_report.nsys-rep > example6_report.txt

# Optional: print confirmation and show brief output
echo "Profiling complete. Summary saved to example6_report.txt"
echo "Preview of report:"
head -n 20 example6_report.txt


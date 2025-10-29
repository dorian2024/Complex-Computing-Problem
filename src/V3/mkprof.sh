#!/bin/bash

make clean
make example3-gpu
nsys profile -o example3_report ./example3
nsys stats example3_report.nsys-rep

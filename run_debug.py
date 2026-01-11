#!/usr/bin/env python3
import os
import sys

# Change to the project directory
os.chdir('/Users/prathamarunshetty/Desktop/PPG_Estimation_Project')

# Import and run the debug function
from debug_wesad_structure import examine_wesad_structure

print("Debugging WESAD data structure...")
examine_wesad_structure()

print("\n" + "="*50)
print("Checking another subject (S3) for comparison...")
examine_wesad_structure(subject='S3')


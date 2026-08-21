#!/bin/bash

# Stop the script immediately if the build fails
set -e

# stop script if there is a typo
set -u

echo "========================================"
echo "Building project... "
echo "========================================"

cmake --build build_debug

echo ""
echo "========================================"
echo " Running all tests... "
echo "========================================"

# Look inside the build folder for any file starting with "test_"
# (Change "build/test_*" to match where your test files live)
for test_exe in build_debug/tests/*; do

    # Make sure it is actually a file we can run (executable)
    if [[ -f "$test_exe" && -x "$test_exe" ]]; then
        
        # Get just the name of the file (strips away "build/")
        test_name=$(basename "$test_exe")
        
        echo ""
        echo ">>> STARTING TEST: $test_name <<<"
        echo "----------------------------------------"
        
        # Run the test executable
         ./"$test_exe"
        
        echo "----------------------------------------"
    fi
done

echo "All tests finished!"

#!/bin/bash
# Full pipeline: .obj -> .msh -> FEM simulation
# Usage: ./run_pipeline.sh path/to/model.obj

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.obj>"
    exit 1
fi

INPUT_OBJ="$1"

# Extract filename without path and extension
BASENAME=$(basename "$INPUT_OBJ" .obj)

# Paths
MSH_OUTPUT="./tetrahedral/${BASENAME}.msh"
GEO_TEMP="./temp_convert.geo"

echo "=== Pipeline for: $BASENAME ==="

# Create tetrahedral directory if it doesn't exist
mkdir -p ./tetrahedral

# Step 1: Create temporary .geo file for conversion
echo "Step 1: Creating Gmsh geo file..."
cat > "$GEO_TEMP" << EOF
Merge "${INPUT_OBJ}";

// Wrap the mesh into a volume
Surface Loop(1) = {1};
Volume(1) = {1};

// Mesh size control (adjust for your model)
Mesh.CharacteristicLengthMin = 0.5;
Mesh.CharacteristicLengthMax = 2.0;
EOF

# Step 2: Run Gmsh to convert OBJ to MSH
echo "Step 2: Converting OBJ to MSH with Gmsh..."
gmsh "$GEO_TEMP" -3 -o "$MSH_OUTPUT" -format msh2

# Clean up temp file
rm -f "$GEO_TEMP"

echo "Generated: $MSH_OUTPUT"

# Step 3: Run FEM simulation
echo "Step 3: Running FEM simulation..."
python fed_test.py "$BASENAME"

echo "=== Pipeline complete ==="
echo "Results saved to: vtk/${BASENAME}.vtu"

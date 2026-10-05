# Go to the reBot Isaac Sim project directory
cd ~/rebot-arm-dli-isaacsim

# Find all relevant 3D scene, model, and Blender files in the project
find . -type f \( \
  -iname "*.usd" -o \
  -iname "*.usda" -o \
  -iname "*.usdc" -o \
  -iname "*.blend" -o \
  -iname "*.fbx" -o \
  -iname "*.obj" -o \
  -iname "*.glb" -o \
  -iname "*.gltf" \
\) | sort

# Search Python and shell scripts for code related to scene loading,
# USD references, object creation, stationery assets, the cup, and the table
grep -RInE \
'open_stage|add_reference|add_payload|create_prim|pencil|cup|stationery|table' \
. \
--include="*.py" \
--include="*.sh" \
2>/dev/null | head -200

# Define the original project directory
SRC="$HOME/rebot-arm-dli-isaacsim"

# Define a new "scene" folder on the Desktop
DST="$(xdg-user-dir DESKTOP)/scene"

# Create the destination folder if it does not already exist
mkdir -p "$DST"

# Copy the project to the Desktop while excluding unnecessary files,
# caches, datasets, logs, and virtual environments
rsync -a \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='datasets/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  "$SRC/" "$DST/"

# Print the final destination path
echo "$DST"

# Show the total size of the copied scene folder
du -sh "$DST"

# List the copied files up to three directory levels deep
find "$DST" -maxdepth 3 -type f | sort

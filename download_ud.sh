#! /bin/bash

set -e

# Create temporary directory.
TEMP_DIR=$(mktemp -d)

# Download UD.
pushd $TEMP_DIR
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1548{/ud-treebanks-v1.2.tgz,/ud-documentation-v1.2.tgz,/ud-tools-v1.2.tgz}
tar -xvf ud-treebanks-v1.2.tgz
popd

# Convert UD.
python preprocess/convert_ud.py --input-dir $TEMP_DIR/universal-dependencies-1.2 --output-dir data/ud_v1.2

# Clean up.
rm -r $TEMP_DIR

echo 'Data is now in data/ud_v1.2'

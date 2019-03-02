#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

cd ops/

echo "install PyTorch Detection..."
$python setup.py build develop

echo "Building deformable convolution v2 op..."
cd deform_conv_v2/src/
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py install

echo "Building deformable psroi v2 op..."
cd deform_psroi/src/
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py install
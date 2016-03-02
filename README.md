## Hierarchical Max Flow GUI 

### License:  
BSD (see license.md)

### Details: 
Early research prototype of the HMF for ventricular scar tissue segmentation. If used, please cite our paper: 

M. Rajchl, J. Yuan, J. White, E. Ukwatta, J. Stirrat, C. Nambakhsh, F. Li, T. Peters. (2014) "Interactive Hierarchical Max-Flow Segmentation of Scar Tissue from Late-Enhancement Cardiac MR Images". IEEE Trans Med Imag, 33(1), 159-172.

### Compile/Installation instructions:  
Requires some older dependencies. Versions stated here were used to compile and run this interface correctly (All instructions for Ubuntu 14.04 LTS):

1. Install CUDA 6.0 and corresponding NVIDIA drivers
2. Build ITK 3.20 (www.itk.org/)
3. Build/install Qt 4.5 
4. Build VTK 5.8 (www.vtk.org/) w/ GUI support option 
5. Use CMake to compile the interface and set all paths
6. make

The executable can be found in ./$BUILD_DIR/UI/HierarchicalMFGUI



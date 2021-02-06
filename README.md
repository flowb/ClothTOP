# ClothTOP

This project demonstrates how to use NVIDIA FleX for GPU cloth simulation in a TouchDesigner Custom Operator. It also shows how to render dynamic meshes from the texture data using custom PBR GLSL material shaders inside TouchDesigner.

Features:
- Cloth/inflatable simulation with controllable anchor points.
- Triangle mesh collision.
- Spheres, boxes and planes collisions.
- Many controlable parameters like wind, gravity, adherence, stiffness, etc.

## Demo Previews

![](img/dali_gif.gif)
![](img/balloon_gif.gif)
![](img/cloak_gif.gif)

Demos:
- **ClothTOP_dali.toe** shows complex triangle mesh collision with multiple cloth bodies.
- **ClothTOP_inflatable.toe** shows an inflatable body with dynamic pressure.
- **ClothTOP_cloak.toe** shows a cloth body with animated anchor points.

## Install NVIDIA FleX

- Download [FleX 1.2](https://github.com/NVIDIAGameWorks/FleX) (get access [here](https://developer.nvidia.com/gameworks-source-github)).
- Create a new environment variable called `NvFlexDir` that holds the path to the `/flex` folder you unpacked.

## Compilation

- Install the [CUDA SDK](https://developer.nvidia.com/Cuda-downloads) you want to use. <br>
- Generate the Visual Studio project using [CMake](https://cmake.org/download/). <br>
- Building will automatically copy the .dll to the Plugins folder. If instead you are using the release, just create the Plugins folder yourself and place the .dll there. <br>
- [TouchDesigner](https://derivative.ca/download) 2020.28110+ supported (tested on Windows 10).

## 3D Models

- [Dali](https://sketchfab.com/3d-models/dalithe-persistence-of-memory-ab3e99facbdb4d9d8661d3f07815638e) 3D model (download and place the .fbx next to ClothTOP_dali.toe) <br>
- [Inflatable](https://www.turbosquid.com/3d-models/unity-decor-model-1360123) 3D model (already in ClothTOP_inflatable.toe) <br>
- [Cloak](https://www.turbosquid.com/3d-models/free-cloak-cape-robe-3d-model/299477) 3D model (already in ClothTOP_cloak.toe)<br>

## References

- The FlexCHOP by Vincent Houzé provided a starting point for this project.

## Known Issues
- Moving anchors too fast can cause crashes with the solver not converging for extreme deltas.
- The **UV** output mode can easily crash for certain meshes. Recommended use of **Linear** mode.
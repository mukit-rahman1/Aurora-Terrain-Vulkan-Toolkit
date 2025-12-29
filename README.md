# Aurora Terrain Tool
A small terrain pipeline that takes any 16-bit 256x256 grayscale heightmap (you may get it from anywhere) and turns it into a mesh in Blender with northern lights.  

The Project takes a 256x256 px heightmap and assigns it to 16x16 workgroups, each covering 16x16px. Each work group also has 16x16 threads. This means when computing in parallel, that is 65,536 threads in parallel.

**NOTE:** I DID NOT make the model for the aurora lights and the night sky. A free template from [Kammerbild](https://www.youtube.com/@Kammerbild) was [used](https://www.patreon.com/Kammerbild).

## Goal Behind Project

After finishing my operating systems course (ELEC377), I became curious and began researching more about Computer Systems, multithreading, and low-level operations.  
I wanted to apply concepts of multithreading, synchronization, CLI subcommands, and concurrent data structures. And I wanted to explore and experiment with parallel programming models using Vulkan compute.

- **NOTE:** The goal of the project was strictly to learn. The heightmap used was so small that the overhead from setting up layouts for parallel computing could make it slower than just computing sequentially. 
## Usage/Examples
You can take your heightmap from this:  
![Height map pic](https://github.com/mukit-rahman1/Aurora-Terrain-Vulkan-Toolkit/blob/main/example.png?raw=true)


To this:  
This is a pic of Mount Kilimanjaro
![Kilimanjaro](https://github.com/mukit-rahman1/Aurora-Terrain-Vulkan-Toolkit/blob/main/Screenshot%202025-12-28%20144119.png?raw=true)

  
This is a picture of a section of Wall Maria from AoT.  
![Maria](https://github.com/mukit-rahman1/Aurora-Terrain-Vulkan-Toolkit/blob/main/Screenshot%202025-12-28%20185450.png?raw=true)

## Installation

### Step 0
Prerequisites:
- **NOTE:** I ran this project on a AMD RX7700XT(sapphire). At the very least, this project will require a GPU for parallel computing.
- Mingw32, GCC, and CMake setup
- Blender installed
- Delete any build folder in the root directory

### Step 1
Get a 256x256 greyscale heightmap, name it "hm.png" and place it in src/assets/  
You can get one of these from many sites; I used [this](https://manticorp.github.io/unrealheightmap).

### Step 2
Make a clean build using CMake (can be done with VSCode extension)

### Step 3
Run the following commands one by one in bash.
```bash
cd build
./auroraterrian.exe build --heightmap ../src/assets/hm.png --out ../out/world --lods 1
./auroraterrian.exe export_mesh --in ../out/world --out out/meshes --lods 1 --scale 100 --spacing 1
"/c/Program Files/Blender Foundation/Blender 4.3/blender.exe"/ --python ../src/setup_scene.py -- "out/meshes/tile_0_0_lod0.obj"  "../src/assets/KB_procedural-Aurora.blend" 
```
**NOTE:** You may need to change the last command to match your Blender install location AND version.
### Step 3
Blender should come up on its own after the last command. Once in Blender, hold Z and click "Render" to go to render mode. Press spacebar to animate the aurora.
## Authors

- [@mukit-rahman1](https://github.com/mukit-rahman1)
- **NOTE:** I DID NOT make the model for the aurora lights and the night sky. A free template from [Kammerbild](https://www.youtube.com/@Kammerbild) was [used](https://www.patreon.com/Kammerbild).

## Code Flow
This diagram shows the process of build_command.cpp and export_mesh.cpp.  
![flow](https://github.com/mukit-rahman1/Aurora-Terrain-Vulkan-Toolkit/blob/main/Aurora.png?raw=true)


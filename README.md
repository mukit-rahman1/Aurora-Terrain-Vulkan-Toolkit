# Aurora Terrain Tool
A small terrain pipeline that takes any 16-bit 256x256 grayscale heightmap (you may get it from anywhere) and turns it into a mesh in blender with northern lights.  

The Project takes a 256x256 px heightmap and have assigns it to 16x16 workgroups, each cover 16x16px. Each work group also has 16x16 threads. This means when computing in parallel, that is 65,536 threads in parallel.

**NOTE:** I DID NOT make the model for the aurora lights and the night sky. A free template from [Kammerbild](https://www.youtube.com/@Kammerbild) was [used](https://www.patreon.com/Kammerbild).

## Goal Behind Project

After finishing my operating systems course (ELEC377), I became curious and began researching more about Computer Systems, multithreading, and low-level operations.  
I wanted to apply concepts of multithreading, synchronization, CLI subcommands, and concurrent data structures. And I wanted to explore and experiment with parallel programming models using Vulkan compute.

- **NOTE:** The goal of the project was strictly to learn. The heightmap used was so small that the overhead from setting up layout for parallel computing could make it slower than just computing sequetially. 
## Usage/Examples
You can take your heightmap from this:  
![Height map pic](<img width="256" height="256" alt="hm" src="https://github.com/user-attachments/assets/4ed2a468-8d5a-4279-92b3-de3732b48e8a" />)


To this:  
![Kilimanjaro](<img width="1703" height="1076" alt="Screenshot 2025-12-28 144119" src="https://github.com/user-attachments/assets/9bb6eea3-33cb-455c-8a0e-d29be76a7027" />)

This is a pic of mount Kilimanjaro
  
![Maria](<img width="1904" height="1184" alt="Screenshot 2025-12-28 185450" src="https://github.com/user-attachments/assets/2c118bb2-2a1d-40d2-96c9-167bf320b118" />)
This is a picture a section of wall Maria from AoT.
## Installation

### Step 0
Prerequisites:
- **NOTE:** I ran this project on a AMD RX7700XT(sapphire). At the very least, this project will require a GPU for parallel computing.
- Mingw32, GCC, and CMake setup
- Blender installed
- Delete any build folder in root directory

### Step 1
Get a 256x256 greyscale heightmap, name it "hm.png" and place it in src/assest/  
You can get one of these from many site, I used [this](https://manticorp.github.io/unrealheightmap.).

### Step 2
Make a clean build using CMake (can be done with VSCode extension)

### Step 3
Run the following commands one by one into bash.
```bash
cd build
./auroraterrian.exe build --heightmap ../src/assets/hm.png --out ../out/world --lods 1
./auroraterrian.exe export_mesh --in ../out/world --out out/meshes --lods 1 --scale 100 --spacing 1
"/c/Program Files/Blender Foundation/Blender 4.3/blender.exe"/ --python ../src/setup_scene.py -- "out/meshes/tile_0_0_lod0.obj"  "../src/assets/KB_procedural-Aurora.blend" 
```
**NOTE:** You may need to change the last command to match your Blender install location AND version.
### Step 3
Blender should up on its own after the last command. Once in Blender, hold Z and click "Render" to go to render mode. Press spacebar to animate the aurora.
## Authors

- [@mukit-rahman1](https://github.com/mukit-rahman1)
- **NOTE:** I DID NOT make the model for the aurora lights and the night sky. A free template from [Kammerbild](https://www.youtube.com/@Kammerbild) was [used](https://www.patreon.com/Kammerbild).

## Code Flow
This diagram shows the process of build_command.cpp and export_mesh.cpp.  
![flow](<img width="933" height="831" alt="Aurora" src="https://github.com/user-attachments/assets/d5265063-62a9-4e66-b2ed-1d718cf2af70" />)


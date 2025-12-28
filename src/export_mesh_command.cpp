#include "export_mesh_command.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <exception>
#include <algorithm>
/*
1) find the file and make new dir if needed , find raw file 
2) jobs thread(push a job for each .raw file)
3) worker thread(read heights and build mesh)
4) writer thread(Write OBJ) 

 the bounded buffers used and their producer and consumer are as follows:
 jobs -> [jobQ 64] -> workers -> [writeQ 16] -> writer

 each have while loop that only stops when a buffer is closed
*/

static constexpr uint32_t TILE0_SIZE = 256;
//helpers
static void ensureDir(const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path));
}
static bool fileExists(const std::string& path) {
    return std::filesystem::exists(std::filesystem::path(path));
}
static std::vector<uint16_t> readRawU16(const std::string& path, size_t count) {
    std::vector<uint16_t> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open: " + path);

    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(count * sizeof(uint16_t)));
    if (!f) throw std::runtime_error("Failed to read enough bytes: " + path);
    return data;
}
//helper to write OBJ files
static void writeOBJ(const std::string& path, const std::vector<float>& vertsXYZ, const std::vector<uint32_t>& indices)
{
    std::ofstream o(path);
    if (!o) throw std::runtime_error("Failed to write: " + path);

    for (size_t i = 0; i < vertsXYZ.size(); i += 3) {
        o << "v " << vertsXYZ[i] << " " << vertsXYZ[i + 1] << " " << vertsXYZ[i + 2] << "\n";
    }
    for (size_t i = 0; i < indices.size(); i += 3) {
        o << "f " << (indices[i] + 1) << " " << (indices[i + 1] + 1) << " " << (indices[i + 2] + 1) << "\n";
    }
}

// Parse "tile_X_Y" -> (X, Y). Returns false if format unexpected.
static bool parseTileXY(const std::string& folderName, uint32_t& tx, uint32_t& ty) {
    const std::string prefix = "tile_"; //check tiles_0_0 folder in world
    if (folderName.rfind(prefix, 0) != 0) return false;

    size_t p1 = folderName.find('_', 5);
    if (p1 == std::string::npos) return false;

    std::string sx = folderName.substr(5, p1 - 5); //get dimensions
    std::string sy = folderName.substr(p1 + 1);
    try {
        tx = static_cast<uint32_t>(std::stoul(sx));
        ty = static_cast<uint32_t>(std::stoul(sy));
        return true;
    } catch (...) {
        return false;
    }
}

static void buildGridMeshFromHeightU16(const std::vector<uint16_t>& h, uint32_t N, float spacing,
                                       float heightScale, uint32_t tileX, uint32_t tileY,
                                       std::vector<float>& outVertsXYZ, std::vector<uint32_t>& outIdx)
{
    outVertsXYZ.clear();
    outVertsXYZ.resize(static_cast<size_t>(N) * N * 3);

    const float tileWorldStride = float(TILE0_SIZE - 1) * spacing; //calc world position
    const float baseX = tileWorldStride * float(tileX);
    const float baseZ = tileWorldStride * float(tileY);

    //for all Vertices in x in z
    for (uint32_t z = 0; z < N; z++) {
        for (uint32_t x = 0; x < N; x++) {
            const uint32_t i = z * N + x;

            float px = float(x) * spacing + baseX; //horizontal plane
            float pz = float(z) * spacing + baseZ;

            float yn = float(h[i]) / 65535.0f;   //Vertical plane. (Height is Normalized since in a heightmap, height = intensity)
            float py = yn * heightScale;

            const size_t vi = (size_t)i * 3; //store the data in the array
            outVertsXYZ[vi + 0] = px;
            outVertsXYZ[vi + 1] = py;
            outVertsXYZ[vi + 2] = pz;
        }
    }

    outIdx.clear();
    outIdx.reserve(static_cast<size_t>(N - 1) * (N - 1) * 6);//tell how much mem to reserve for output. 2 triangles each quad. 6 points
    
    //for all Vertices in x in z
    for (uint32_t z = 0; z < N - 1; z++) {
        for (uint32_t x = 0; x < N - 1; x++) {
            uint32_t i0 = z * N + x;        //top left triangle
            uint32_t i1 = z * N + (x + 1);  //top right triangle
            uint32_t i2 = (z + 1) * N + x;  //bottom left triangle
            uint32_t i3 = (z + 1) * N + (x + 1);//bottom right triangle

            outIdx.push_back(i0); outIdx.push_back(i2); outIdx.push_back(i1);//add these numbers at the very end of the list
            outIdx.push_back(i1); outIdx.push_back(i2); outIdx.push_back(i3);
        }
    }
}

// --- Bounded Buffer ---
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    // blocks if full returns false if closed
    bool push(T item) {
        std::unique_lock<std::mutex> lk(m_);//wait(mutex)
        cvNotFull_.wait(lk, [&] { return closed_ || q_.size() < cap_; }); //if predicate is true, go immediately. Else sleep.//wait(empty)
        if (closed_) return false;
        q_.push_back(std::move(item));  //Critical section. Writes to buffer
        cvNotEmpty_.notify_one();       //signal(full)
        return true;                    //mutex automatically signals. signal(mutex)
    }

    // blocks if empty; returns false if empty+closed
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);//wait(mutex)
        cvNotEmpty_.wait(lk, [&] { return closed_ || !q_.empty(); });//if predicate is true, go immediately. Else sleep.//wait(full)
        if (q_.empty()) return false; // closed + empty
        out = std::move(q_.front());
        q_.pop_front();
        cvNotFull_.notify_one();       //signal(empty)
        return true;                //mutex automatically signals. signal(mutex)
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cvNotEmpty_.notify_all();
        cvNotFull_.notify_all();
    }

private:
    size_t cap_;
    std::deque<T> q_;
    bool closed_ = false;
    std::mutex m_;
    std::condition_variable cvNotEmpty_;
    std::condition_variable cvNotFull_;
};

// --- jobs ---
struct ExportJob {
    std::string tileFolderName; // "tile_X_Y"
    std::string tileDirPath;    // full path to that tile directory
    uint32_t tileX = 0;
    uint32_t tileY = 0;
};
struct WriteJob {
    std::string outObjPath;
    std::vector<float> verts;
    std::vector<uint32_t> idx;
};

static void setExceptionOnce(std::exception_ptr& dst, std::mutex& m, std::exception_ptr e) {
    std::lock_guard<std::mutex> lk(m);
    if (!dst) dst = e;
}

//function to run export mesh command
int runExportMeshCommand(const ExportMeshArgs& args) {
    // ---1) find the file and make new dir if needed ---
    const std::string tilesDir = args.inDir + "/tiles"; 
    if (!std::filesystem::exists(tilesDir)) {
        throw std::runtime_error("Tiles folder not found: " + tilesDir);
    }
    ensureDir(args.outDir);

    // Export only LOD0 
    const uint32_t lod = 0;
    const uint32_t N = TILE0_SIZE; // 256
    //jobQ has 64 spaces. writeQ has 16 spaces
    BoundedQueue<ExportJob> jobQ(64);
    BoundedQueue<WriteJob>  writeQ(16);

    std::atomic<size_t> exported{0};
    std::exception_ptr exPtr = nullptr;
    std::mutex exM;

    // --------------- writer thread (I/O) ---------------
    std::thread writer([&] {
        try {           //pop from writeQ and write
            WriteJob wj;
            while (writeQ.pop(wj)) { //LOOP 
                writeOBJ(wj.outObjPath, wj.verts, wj.idx);
                exported.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (...) {
            setExceptionOnce(exPtr, exM, std::current_exception());
            jobQ.close();
            writeQ.close();
        }
    });

    // --------------- worker threads (CPU compute) ---------------
    uint32_t hw = std::thread::hardware_concurrency();//how many worker threads can the CPU take?
    if (hw == 0) hw = 4;
    const uint32_t workerCount = std::max(1u, hw - 1u); //one less thread just can case
    std::vector<std::thread> workers;   
    workers.reserve(workerCount);       //reserve, alloc space

    for (uint32_t t = 0; t < workerCount; t++) {
        workers.emplace_back([&] {
            try {
                ExportJob j;
                while (jobQ.pop(j)) { 
                    const std::string hPath = j.tileDirPath + "/lod0.height.raw";
                    if (!fileExists(hPath)) continue;

                    auto h = readRawU16(hPath, static_cast<size_t>(N) * N); //calc height data
                    WriteJob wj;
                    wj.outObjPath = args.outDir + "/" + j.tileFolderName + "_lod0.obj";

                    buildGridMeshFromHeightU16(h, N, args.spacing, args.heightScale, j.tileX, j.tileY, wj.verts, wj.idx);

                    if (!writeQ.push(std::move(wj))) break; // writer stopped/closed. push to writeQ
                }
            } catch (...) {
                setExceptionOnce(exPtr, exM, std::current_exception());
                jobQ.close();
                writeQ.close();
            }
        });
    }

    // --------------- jobs (main thread, producer) ---------------
    try {
        for (const auto& entry : std::filesystem::directory_iterator(tilesDir)) {// for all files
            if (!entry.is_directory()) continue;

            const std::string folderName = entry.path().filename().string();
            uint32_t tileX = 0, tileY = 0;
            if (!parseTileXY(folderName, tileX, tileY)) continue;

            ExportJob j;
            j.tileFolderName = folderName;
            j.tileDirPath = entry.path().string();
            j.tileX = tileX;
            j.tileY = tileY;

            if (!jobQ.push(std::move(j))) break; // closed due to error. else, push to jobQ
        }
    } catch (...) {
        setExceptionOnce(exPtr, exM, std::current_exception());
    }

    // signal no more jobs
    jobQ.close();

    // join workers then close writer queue
    for (auto& th : workers) th.join();
    writeQ.close();
    writer.join();

    // rethrow if any thread hit an error
    if (exPtr) std::rethrow_exception(exPtr);
    std::cout << "Exported " << exported.load() << " OBJ files to: " << args.outDir
              << " using " << workerCount << " worker threads\n";
    return 0;
}


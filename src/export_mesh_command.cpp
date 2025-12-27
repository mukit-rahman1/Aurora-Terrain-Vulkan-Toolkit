#ifdef _WIN32
  #include <windows.h>
#endif

#include "export_mesh_command.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t TILE0_SIZE = 256;

static void launchProcess(const std::string& cmdLine) {
    std::vector<char> buf(cmdLine.begin(), cmdLine.end());
    buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    BOOL ok = CreateProcessA(
        nullptr,
        buf.data(),
        nullptr, nullptr,
        FALSE,
        0,
        nullptr,
        nullptr,
        &si,
        &pi
    );

    if (!ok) {
        DWORD err = GetLastError();
        throw std::runtime_error("CreateProcessA failed. GetLastError=" + std::to_string(err));
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
}

#ifdef _WIN32
static void launchBlenderCreateProcess(const std::string& blenderExe,
                                       const std::string& pyPath,
                                       const std::string& objPath)
{
    // CreateProcess needs a mutable command line buffer.
    std::string cmdLine =
        "\"" + blenderExe + "\" --python \"" + pyPath + "\" -- \"" + objPath + "\"";

    std::vector<char> buf(cmdLine.begin(), cmdLine.end());
    buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    BOOL ok = CreateProcessA(
        nullptr,            // app name (nullptr = use cmdLine)
        buf.data(),         // command line (mutable)
        nullptr, nullptr,
        FALSE,
        0,
        nullptr,            // env
        nullptr,            // current dir
        &si,
        &pi
    );

    if (!ok) {
        DWORD err = GetLastError();
        throw std::runtime_error("CreateProcessA failed. GetLastError=" + std::to_string(err));
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
}
#endif

static void ensureDir(const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path));
}

static bool fileExists(const std::string& path) {
    return std::filesystem::exists(std::filesystem::path(path));
}
static void writeTextFile(const std::string& path, const std::string& text) {
    std::ofstream o(path);
    if (!o) throw std::runtime_error("Failed to write: " + path);
    o << text;
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


static void writeOBJ(const std::string& path,
                     const std::vector<float>& vertsXYZ,
                     const std::vector<uint32_t>& indices)
{
    std::ofstream o(path);
    if (!o) throw std::runtime_error("Failed to write: " + path);

    // vertices
    for (size_t i = 0; i < vertsXYZ.size(); i += 3) {
        o << "v " << vertsXYZ[i] << " " << vertsXYZ[i + 1] << " " << vertsXYZ[i + 2] << "\n";
    }

    // faces (OBJ is 1-based)
    for (size_t i = 0; i < indices.size(); i += 3) {
        o << "f "
          << (indices[i] + 1) << " "
          << (indices[i + 1] + 1) << " "
          << (indices[i + 2] + 1) << "\n";
    }
}

// Parse "tile_X_Y" -> (X, Y). Returns false if format unexpected.
static bool parseTileXY(const std::string& folderName, uint32_t& tx, uint32_t& ty) {
    // expected: "tile_3_5"
    const std::string prefix = "tile_";
    if (folderName.rfind(prefix, 0) != 0) return false;

    // split by '_'
    // tile_ X _ Y
    size_t p1 = folderName.find('_', 5);
    if (p1 == std::string::npos) return false;

    std::string sx = folderName.substr(5, p1 - 5);
    std::string sy = folderName.substr(p1 + 1);

    try {
        tx = static_cast<uint32_t>(std::stoul(sx));
        ty = static_cast<uint32_t>(std::stoul(sy));
        return true;
    } catch (...) {
        return false;
    }
}

static void buildGridMeshFromHeightU16(const std::vector<uint16_t>& h,
                                       uint32_t N,
                                       float spacing,
                                       float heightScale,
                                       uint32_t tileX,
                                       uint32_t tileY,
                                       std::vector<float>& outVertsXYZ,
                                       std::vector<uint32_t>& outIdx)
{
    // Vertex positions
    outVertsXYZ.clear();
    outVertsXYZ.resize(static_cast<size_t>(N) * N * 3);

    // Offset tiles so they line up in world space.
    // Using (TILE0_SIZE - 1) reduces duplicated seams when adjacent tiles meet.
    const float tileWorldStride = float(TILE0_SIZE - 1) * spacing;
    const float baseX = tileWorldStride * float(tileX);
    const float baseZ = tileWorldStride * float(tileY);

    for (uint32_t z = 0; z < N; z++) {
        for (uint32_t x = 0; x < N; x++) {
            const uint32_t i = z * N + x;

            float hx = float(x) * spacing + baseX;
            float hz = float(z) * spacing + baseZ;

            float yn = float(h[i]) / 65535.0f;   // normalized 0..1
            float hy = yn * heightScale;

            const size_t vi = (size_t)i * 3;
            outVertsXYZ[vi + 0] = hx; // x
            outVertsXYZ[vi + 1] = hy; // y
            outVertsXYZ[vi + 2] = hz; // z
        }
    }

    // Indices (two triangles per quad)
    outIdx.clear();
    outIdx.reserve(static_cast<size_t>(N - 1) * (N - 1) * 6);

    for (uint32_t z = 0; z < N - 1; z++) {
        for (uint32_t x = 0; x < N - 1; x++) {
            uint32_t i0 = z * N + x;
            uint32_t i1 = z * N + (x + 1);
            uint32_t i2 = (z + 1) * N + x;
            uint32_t i3 = (z + 1) * N + (x + 1);

            // CCW winding
            outIdx.push_back(i0); outIdx.push_back(i2); outIdx.push_back(i1);
            outIdx.push_back(i1); outIdx.push_back(i2); outIdx.push_back(i3);
        }
    }
}

//Bounded Queue and job structs
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    // returns false if closed (item not pushed)
    bool push(T item) {
        std::unique_lock<std::mutex> lk(m_);
        cv_not_full_.wait(lk, [&]{ return closed_ || q_.size() < cap_; });
        if (closed_) return false;
        q_.push_back(std::move(item));
        cv_not_empty_.notify_one();
        return true;
    }

    // returns false when closed AND empty
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_not_empty_.wait(lk, [&]{ return closed_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        cv_not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    size_t cap_;
    std::mutex m_;
    std::condition_variable cv_not_empty_, cv_not_full_;
    std::deque<T> q_;
    bool closed_ = false;
};

struct ExportJob {
    std::string hPath;
    std::string outObj;
    uint32_t N = 0;
    float spacing = 1.0f;
    float heightScale = 1.0f;
    uint32_t tileX = 0, tileY = 0;
};

struct WriteJob {
    std::string outObj;
    std::vector<float> vertsXYZ;
    std::vector<uint32_t> indices;
};


int runExportMeshCommand(const ExportMeshArgs& args) {
    const std::string tilesDir = args.inDir + "/tiles";
    if (!std::filesystem::exists(tilesDir)) {
        throw std::runtime_error("Tiles folder not found: " + tilesDir);
    }

    ensureDir(args.outDir);

    // --- bounded buffers ---
    BoundedQueue<ExportJob> jobQ(64);
    BoundedQueue<WriteJob>  writeQ(64);

    std::atomic<size_t> exported{0};

    // --- exception capture across threads ---
    std::mutex exM;
    std::exception_ptr exPtr = nullptr;
    auto setExceptionOnce = [&](std::exception_ptr ep) {
        std::lock_guard<std::mutex> lk(exM);
        if (!exPtr) exPtr = ep;
    };

    // --- writer thread (controlled I/O) ---
    std::thread writer([&] {
        try {
            WriteJob w;
            while (writeQ.pop(w)) {
                writeOBJ(w.outObj, w.vertsXYZ, w.indices);
                exported.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (...) {
            setExceptionOnce(std::current_exception());
            // Stop the world so everyone can exit
            jobQ.close();
            writeQ.close();
        }
    });

    // --- worker threads (parallel compute) ---
    unsigned hc = std::thread::hardware_concurrency();
    unsigned workerCount = (hc > 1) ? (hc - 1) : 1; // leave 1 core for OS/writer
    std::vector<std::thread> workers;
    workers.reserve(workerCount);

    for (unsigned t = 0; t < workerCount; t++) {
        workers.emplace_back([&] {
            try {
                ExportJob j;
                std::vector<float> verts;
                std::vector<uint32_t> idx;

                while (jobQ.pop(j)) {
                    // read heights
                    auto h = readRawU16(j.hPath, static_cast<size_t>(j.N) * j.N);

                    // build mesh (CPU compute)
                    buildGridMeshFromHeightU16(h, j.N, j.spacing, j.heightScale,
                                               j.tileX, j.tileY, verts, idx);

                    // enqueue write job (move to avoid copy)
                    WriteJob w;
                    w.outObj = j.outObj;
                    w.vertsXYZ = std::move(verts);
                    w.indices  = std::move(idx);

                    if (!writeQ.push(std::move(w))) break;

                    // restore vectors for reuse next loop (avoid realloc churn)
                    verts.clear();
                    idx.clear();
                }
            } catch (...) {
                setExceptionOnce(std::current_exception());
                jobQ.close();
                writeQ.close();
            }
        });
    }

    // --- producer: enumerate jobs ---
    try {
        for (const auto& entry : std::filesystem::directory_iterator(tilesDir)) {
            if (exPtr) break;               // stop if any thread failed
            if (!entry.is_directory()) continue;

            const std::string folderName = entry.path().filename().string();
            uint32_t tileX = 0, tileY = 0;
            if (!parseTileXY(folderName, tileX, tileY)) continue;

            const std::string tilePath = entry.path().string();

            for (uint32_t lod = 0; lod < args.lodCount; lod++) {
                const uint32_t N = TILE0_SIZE >> lod;
                if (N < 2) break;

                const std::string hPath = tilePath + "/lod" + std::to_string(lod) + ".height.raw";
                if (!fileExists(hPath)) continue;

                const std::string outObj =
                    args.outDir + "/" + folderName + "_lod" + std::to_string(lod) + ".obj";

                ExportJob j;
                j.hPath = hPath;
                j.outObj = outObj;
                j.N = N;
                j.spacing = args.spacing;
                j.heightScale = args.heightScale;
                j.tileX = tileX;
                j.tileY = tileY;

                if (!jobQ.push(std::move(j))) break;
            }
        }
    } catch (...) {
        setExceptionOnce(std::current_exception());
    }

    // --- shutdown ---
    jobQ.close();                 // no more jobs
    for (auto& th : workers) th.join();

    writeQ.close();               // no more writes after workers finish
    writer.join();

    // if any thread failed, rethrow it here
    if (exPtr) std::rethrow_exception(exPtr);

    std::cout << "Exported " << exported.load() << " OBJ files to: " << args.outDir << "\n";


    // --- launch Blender  ---
    if (args.openBlender) {
        namespace fs = std::filesystem;

        if (args.blenderPath.empty())
            throw std::runtime_error("--open-blender set but --blender not provided");

        fs::path blenderExe = fs::path(args.blenderPath);
        fs::path pyPath = fs::absolute(fs::path(args.outDir) / "setup_scene.py");

        // Pick OBJ (tile_0_0_lod0 or first .obj)
        fs::path objPath = fs::path(args.outDir) / "tile_0_0_lod0.obj";
        if (!fs::exists(objPath)) {
            bool found = false;
            for (auto& e : fs::directory_iterator(args.outDir)) {
                if (e.is_regular_file() && e.path().extension() == ".obj") {
                    objPath = e.path();
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("No .obj files found in: " + args.outDir);
        }
        objPath = fs::absolute(objPath);

        //HARD FAIL early with clear message (instead of GetLastError=2)
        if (!fs::exists(blenderExe))
            throw std::runtime_error("Blender exe not found: " + blenderExe.string());
        if (!fs::exists(pyPath))
            throw std::runtime_error("setup_scene.py not found: " + pyPath.string());

        // Build command line
        std::string cmdLine =
            "\"" + blenderExe.string() + "\" --python \"" + pyPath.string() + "\" -- \"" + objPath.string() + "\"";

        std::cout << "Launching Blender:\n" << cmdLine << "\n";

    #ifdef _WIN32
        launchProcess(cmdLine);
    #else
        std::system(cmdLine.c_str());
    #endif
    }
    return 0;
}

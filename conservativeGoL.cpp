//////////////////////////////////////////////////////////////////////////////
// Program to simulate the conservative, probabilistic                      //
// version of Conway's Game of Life from                                    //
// https://journals.aps.org/pre/abstract/10.1103/PhysRevE.103.012132        //
// (Actually a slightly altered version that could be reduced to a 1D case  //
// where a neighborhood also includes the cells at -N-1,-N,-N+1 and         //
// symmetrically in the positive direction                                  //
// These systems are asymptotically identical.)                             //
//                                                                          //
// Requires CImg for saving simulation frames, and ffmpeg for               //
// encoding the frames as video output.                                     //
// Compile with minimum C++20                                               //
//                                                                          //
// Copyright Aapo Kössi (@aapo-kossi), 2023                                 //
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <bit>
#include <random>
#include <intrin.h>
#include <format>
#include <bitset>
#include <unordered_map>
#include "Cimg.h"

#define BITNESS 64
#define SNAPRES 512

using namespace std;
using namespace cimg_library;

enum Action {
    INCREMENT = 1,
    DECREMENT = -1
};

struct CounterNode {
    unsigned int ltotal;
    unsigned int rtotal;
    unsigned int location;
    bool final;
};

class CounterTree {
public:
    CounterTree(int N, vector<uint64_t> grid) {
        numLeaf = N*N/BITNESS;
        depth = log2(numLeaf);

        // build the tree starting from the bottom layer (leaves)
        vector<unsigned int> popcounts(N*N/BITNESS);
        for (unsigned int i=0; i<numLeaf; i++) {
            popcounts[i] = _popcnt64(grid[i]);
            nodes.push_back(CounterNode { 0, popcounts[i], i, true });
        }

        // add the narrower layers
        unsigned int layerWidth = numLeaf/2;
        int currentLayer = 1;
        int step = 2;
        while (layerWidth > 0) {
            vector<CounterNode> nLayer;
            // printf("Starting layer %u\n", currentLayer);
            for (int i=0; i<layerWidth; i++) {
                CounterNode lchild = nodes[i*2];
                CounterNode rchild = nodes[i*2+1];
                nLayer.push_back(CounterNode { lchild.ltotal + lchild.rtotal, rchild.ltotal + rchild.rtotal, 0, false });
                // printf("left: %u, right: %u\n", lchild.ltotal + lchild.rtotal, rchild.ltotal + rchild.rtotal);
            }
            nodes.insert(nodes.begin(), nLayer.begin(), nLayer.end());
            layerWidth /= 2;
            currentLayer++;
        }
    }

    // get the index of the leaf that has [n] set bits in total in leaves up to and including itself
    unsigned int getPosition(unsigned int n, unsigned int& totalBelow) {
        unsigned int treeIdx = 0;
        CounterNode current = nodes[treeIdx];
        unsigned int layerSize = 1;
        unsigned int layerIdx = 0;

        for (int i = 0; i < depth; i++) {
            unsigned int prevLayerIdx = layerIdx;
            bool selectRight = (totalBelow + current.ltotal) <= n;
            totalBelow += current.ltotal * selectRight;
            layerIdx = 2*layerIdx + selectRight;
            treeIdx += (layerSize - prevLayerIdx) + layerIdx;
            current = nodes[treeIdx];
            layerSize *= 2;
        }

        // index in the bottom layer of the tree is identical to the grid index
        return layerIdx;
    }

    // update the leaf at index idx and all its parents according to action
    void update(unsigned int idx, int action) {
        unsigned int treeIdx = 0;
        unsigned int layerSize = 1;
        unsigned int layerIdx = 0;
        unsigned int layerStart = 0;
        for (int i = 0; i < depth; i++) {
            layerIdx = idx * layerSize / numLeaf;
            treeIdx = layerStart + layerIdx;
            CounterNode& current = nodes[treeIdx];

            if ( (2 * idx * layerSize / numLeaf) % 2 ) {
                current.rtotal += action;
            } else {
                current.ltotal += action;
            }

            layerStart += layerSize;
            layerSize *= 2;
        }
        treeIdx = layerStart + layerIdx;
        CounterNode& current = nodes[treeIdx];
        current.rtotal += action;
    }

    // get the total number of set bits in the grid
    unsigned int total() {
        return nodes[0].ltotal + nodes[0].rtotal;
    }

    // get the leaf at index idx
    CounterNode leaf(unsigned int idx) {
        return nodes[numLeaf - 1 + idx];
    }

    // print all counters of the tree
    void printAll() {
        for (int i=0; i<numLeaf*2-1; i++) {
            printf("i=%i: l%u, r%u\n", i, nodes[i].ltotal, nodes[i].rtotal);
        }
    }
private:
    unsigned int numLeaf;
    unsigned int depth;
    vector<CounterNode> nodes;
};

// Construct a counter tree for tracking alive, unsatisfied counts
class AliveCounters: public CounterTree {
private:
    vector<uint64_t> getGridUD(int N, uint64_t* gridA, uint64_t* gridU) {
        vector<uint64_t> ret;
        for (int i=0; i<(N*N/BITNESS); i++) {
            ret.push_back(gridU[i] & gridA[i]);
        }
        return ret;
    }
public:
    AliveCounters(int N, uint64_t* gridA, uint64_t* gridU): CounterTree(N, getGridUD(N, gridA, gridU)) {}
};

// Construct a counter tree for tracking dead, unsatisfied counts
class DeadCounters: public CounterTree {
private:
    vector<uint64_t> getGridUD(int N, uint64_t* gridA, uint64_t* gridU) {
        vector<uint64_t> ret;
        for (int i=0; i<(N*N/BITNESS); i++) {
            ret.push_back(gridU[i] & (~gridA[i]));
        }
        return ret;
    }
public:
    DeadCounters(int N, uint64_t* gridA, uint64_t* gridU): CounterTree(N, getGridUD(N, gridA, gridU)) {}
};

unsigned mod(int x, int y) {
    int mod = x % y;
    if (mod < 0) {
        mod += y;
    }
    return mod;
}

// compile-time optimizeable version for fast mod of 64
unsigned mod64(int x) {
    return mod(x, BITNESS);
}


void mapAdd(std::unordered_map<uint64_t, int>& out, std::unordered_map<uint64_t, int>& in) {
    for (auto initer = in.begin(); initer != in.end(); ++initer) {
        out[initer->first] += initer->second;
    }
}

CImg<unsigned char> getFrame(int N, uint64_t* gridA, uint64_t* gridU) {
    unsigned char black[] = { 0,0,0 },  white[] = { 252,250,246 }, green[] = { 0,100,0 }, red[] = { 255,20,10 };
    unsigned char* color;
    unsigned char* colorArray = (unsigned char*)calloc(N*N*3, sizeof(char));
    int greenOffset = N*N;
    int blueOffset  = 2*N*N;
    for (int pixIdx=0; pixIdx<N*N; pixIdx++) {
        bool isAlive = 1U & (gridA[pixIdx/BITNESS] >> mod64(pixIdx));
        bool isSatisfied = 1U ^ (1U & (gridU[pixIdx/BITNESS] >> mod64(pixIdx)));

        if (isAlive) {
            if (isSatisfied) {
                color = white;
            } else {
                color = red;
            }
        } else {
            if (isSatisfied) {
                color = black;
            } else {
                color = green;
            }
        }
        colorArray[pixIdx              ] = color[0];
        colorArray[pixIdx + greenOffset] = color[1];
        colorArray[pixIdx + blueOffset ] = color[2];
    }
    uint64_t dim = N;
    CImg<unsigned char> img(colorArray,dim,dim,1U,3U);
    free(colorArray);
    if (N < (SNAPRES/2)) {
        img.resize(SNAPRES, SNAPRES);
    }
    return img;
}

void initialize(
    int N,
    std::bernoulli_distribution* d,
    std::mt19937* gen,
    uint64_t* grid
) {
    for (uint64_t i=0; i<(N*N); i++) {
        grid[i/BITNESS] |= (uint64_t)(*d)(*gen) << mod64(i);
    }

}

bool isUnsatisfied(uint64_t* grid, int N, int i) {
    int Nsquared = N*N;
    bool isAlive = 1U & (grid[i/BITNESS] >> mod64(i));
    uint64_t n = 0;
    const vector<int> neighborhood = { -1-N, -N, 1-N, -1, 1, N-1, N, N+1};
    for (uint64_t idx=0; idx < neighborhood.size(); idx++) {
        uint64_t j = mod(neighborhood[idx] + i, Nsquared);
        n += 1U & (grid[j/BITNESS] >> mod64(j));
    }

    if (isAlive) {
        return (n != 2) && (n != 3);
    }
    else {
        return n == 3;
    }
}

void print_numbers(int N, uint64_t* grid) {
    uint64_t n_alive = 0;
    for (int i=0; i< N*N/BITNESS; i++) {
        n_alive += _popcnt64(grid[i]);
    }
    printf("%u\n", n_alive);
}

void get_all_unsatisfied(
    int N,
    uint64_t* grid,
    uint64_t* gridUnsatisfied
) {

    #pragma omp parallel for
    for (uint64_t i=0; i<(N*N);i++) {
        bool state = isUnsatisfied(grid, N, i);
        gridUnsatisfied[i/BITNESS] |= (uint64_t)state << mod64(i);
    }
}

void update_unsatisfied(
    int N,
    CounterTree& treeUD,
    CounterTree& treeUA,
    uint64_t* grid,
    uint64_t* gridU,
    uint64_t* pos,
    uint64_t* bitPos
) {
    const vector<int> neighborhood{ -1-N, -N, 1-N, -1, 1, N-1, N, N+1};
    vector<uint64_t> nbrDPositions(neighborhood.size());
    vector<uint64_t> nbrAPositions(neighborhood.size());
    vector<uint64_t> nbrDbitPositions(neighborhood.size());
    vector<uint64_t> nbrAbitPositions(neighborhood.size());
    vector<bool> nbrDisUnsatisfied(neighborhood.size());
    vector<bool> nbrAisUnsatisfied(neighborhood.size());

    // aggregate different neighbors that have identical position to avoid unnecessary updates
    std::unordered_map<uint64_t, int> totalIncrementsA;
    std::unordered_map<uint64_t, int> totalIncrementsD;
    std::unordered_map<uint64_t, uint64_t> updatedGridUElements;

    int gridLength = N*N/BITNESS;

    // this doesn't parallelize very well as we are adding new keys, but it is convenient
    // to do the reduction inside the parallel block.
    #pragma omp declare reduction(mapAdd : std::unordered_map<uint64_t, int> :  \
            mapAdd(omp_out, omp_in))                                            \
            initializer(omp_priv=omp_orig)

    #pragma omp parallel for reduction(mapAdd:totalIncrementsA, totalIncrementsD)
    for (int idx=0; idx < 8; idx++) {
        uint64_t i = mod(pos[0]*BITNESS + __builtin_ctzll(bitPos[0]) + neighborhood[idx], N*N);
        uint64_t j = mod(pos[1]*BITNESS + __builtin_ctzll(bitPos[1]) + neighborhood[idx], N*N);

        nbrDPositions[idx] = i / BITNESS;
        nbrAPositions[idx] = j / BITNESS;

        nbrDbitPositions[idx] = std::rotl(bitPos[0], neighborhood[idx]);
        nbrAbitPositions[idx] = std::rotl(bitPos[1], neighborhood[idx]);

        nbrDisUnsatisfied[idx] = isUnsatisfied(grid, N, i);
        nbrAisUnsatisfied[idx] = isUnsatisfied(grid, N, j);

        // add new unsatisfied state, subtract previous
        int nbrDIncrement = nbrDisUnsatisfied[idx] - !!(gridU[nbrDPositions[idx]] & nbrDbitPositions[idx]);
        int nbrAIncrement = nbrAisUnsatisfied[idx] - !!(gridU[nbrAPositions[idx]] & nbrAbitPositions[idx]);

        if (grid[nbrDPositions[idx]] & nbrDbitPositions[idx]) {
            // this neighbor of dead cell is alive
            totalIncrementsA[nbrDPositions[idx]] += nbrDIncrement;
        } else {
            // this neighbor of dead cell is dead
            totalIncrementsD[nbrDPositions[idx]] += nbrDIncrement;
        }

        if (grid[nbrAPositions[idx]] & nbrAbitPositions[idx]) {
            // this neighbor of alive cell is alive
            totalIncrementsA[nbrAPositions[idx]] += nbrAIncrement;
        } else {
            // this neighbor of alive cell is dead
            totalIncrementsD[nbrAPositions[idx]] += nbrAIncrement;
        }

    }
    
    for (auto& iterIncrD : totalIncrementsD) {
        if (iterIncrD.second != 0) {
            treeUD.update(iterIncrD.first, iterIncrD.second);
        }
    }

    for (auto& iterIncrA : totalIncrementsA) {
        if (iterIncrA.second != 0) {
            treeUA.update(iterIncrA.first, iterIncrA.second);
        }
    }

    for (int idx=0; idx < 8; idx++) {

        gridU[nbrDPositions[idx]] = (
            gridU[nbrDPositions[idx]] & ~nbrDbitPositions[idx])
            | (nbrDbitPositions[idx]*nbrDisUnsatisfied[idx]);

        gridU[nbrAPositions[idx]] = (
            gridU[nbrAPositions[idx]] & ~nbrAbitPositions[idx])
            | (nbrAbitPositions[idx]*nbrAisUnsatisfied[idx]);

    }
}

void unsatisfiedIdxToGridIdx(
    int N,
    uint64_t* gridA,
    uint64_t* gridU,
    uint64_t& nd,
    uint64_t& na,
    CounterTree& treeUD,
    CounterTree& treeUA,
    uint64_t* pos,
    uint64_t* bitPos
) {
    unsigned int belowCountUD = 0;
    unsigned int belowCountUA = 0;
    pos[0] = treeUD.getPosition(nd, belowCountUD);
    pos[1] = treeUA.getPosition(na, belowCountUA);
    bitPos[0] = _pdep_u64(1ULL << (nd - belowCountUD), (gridU[pos[0]] & (~gridA[pos[0]])));
    bitPos[1] = _pdep_u64(1ULL << (na - belowCountUA), (gridU[pos[1]] & gridA[pos[1]]));
}

void step(
    int N,
    CounterTree& treeUD,
    CounterTree& treeUA,
    uint64_t* grid,
    uint64_t* gridUnsatisfied,
    uint64_t* pos,
    uint64_t* bitPos,
    mt19937& gen
) {
    uniform_int_distribution<> dD(0, treeUD.total() - 1);
    uniform_int_distribution<> dA(0, treeUA.total() - 1);
    uint64_t idxd = dD(gen);
    uint64_t idxa = dA(gen);
    unsatisfiedIdxToGridIdx(
        N,
        grid,
        gridUnsatisfied,
        idxd,
        idxa,
        treeUD,
        treeUA,
        pos,
        bitPos
    ); // populate positions of cells to be swapped

    // swap cells
    grid[pos[0]] ^= bitPos[0];
    grid[pos[1]] ^= bitPos[1];
    treeUD.update(pos[0], DECREMENT);
    treeUA.update(pos[1], DECREMENT);


    //check if swapped cells are unsatisfied, update counts
    if (isUnsatisfied(grid, N, pos[0]*BITNESS + __builtin_ctzll(bitPos[0]))) {
        treeUA.update(pos[0], INCREMENT);
    } else {
        gridUnsatisfied[pos[0]] ^= bitPos[0];
    }
    if (isUnsatisfied(grid, N, pos[1]*BITNESS + __builtin_ctzll(bitPos[1]))) {
        treeUD.update(pos[1], INCREMENT);
    } else {
        gridUnsatisfied[pos[1]] ^= bitPos[1];
    }

    // update grid of unsatisfied cells and their counts
    update_unsatisfied(N, treeUD, treeUA, grid, gridUnsatisfied, pos, bitPos);
}

int main(int argc, char** argv) {

    char* filename;

    if (argc < 7) {
        printf("Required arguments: grid-size, p-dead, n-iters, n-video-frames, video-freq, rng-seed\n");
        printf("Optional argument: output video file name");
        return 1;
    } else if (argc < 8) {
        filename = (char*)"GoL_video.raw";
    } else {
        filename = argv[7];
    }
    int N = atoi(argv[1]);
    float p = atof(argv[2]);
    uint64_t numIter = atof(argv[3]);
    uint64_t numFrames = atoi(argv[4]);
    float freq = atof(argv[5]);
    uint64_t seed = atoi(argv[6]);

    mt19937 gen(seed);
    bernoulli_distribution d(1-p);
    printf(
        "Running conservative Game of Life.\n"
        "Grid size: %i\n"
        "Initialization probability of cell being dead: %f\n"
        "Maximum number of iterations: %i\n"
        "Maximum numbe of generated video frames: %i\n"
        "Animation frequency in units of 1/dt: %f\n",
        N, p, numIter, numFrames, freq);

    uint64_t* gridAlive = (uint64_t*)calloc(N*N/BITNESS + 1, sizeof(uint64_t));
    uint64_t* gridUnsatisfied = (uint64_t*)calloc(N*N/BITNESS + 1, sizeof(uint64_t));

    initialize(N, &d, &gen, gridAlive);
    get_all_unsatisfied(N, gridAlive, gridUnsatisfied);
    DeadCounters treeUD(N, gridAlive, gridUnsatisfied);
    AliveCounters treeUA(N, gridAlive, gridUnsatisfied);

    CImgList<unsigned char> animation;
    getFrame(N, gridAlive, gridUnsatisfied).move_to(animation);

    uint64_t i = 0;
    uint64_t pos[2];
    uint64_t bitPos[2];
    float frameTime = 0;
    float totalTime = 0;
    uint64_t currentFrame = 1; // The first simulation frame is already plotted
    float maxFrameTime = 1.0f/freq;

    for (i; i<numIter; i++) {
        if (!treeUD.total() || !treeUA.total()) {
            break;
        }
        float diff = frameTime - maxFrameTime;
        if (currentFrame < (numFrames-1) && (diff >= 0)) {
            getFrame(N, gridAlive, gridUnsatisfied).move_to(animation);
            frameTime = diff;
            currentFrame += 1;
        }
        float stepTime = 1.0f/(treeUD.total() + treeUA.total());
        step(N, treeUD, treeUA, gridAlive, gridUnsatisfied, pos, bitPos, gen);
        frameTime += stepTime;
        totalTime += stepTime;
    }

    getFrame(N, gridAlive, gridUnsatisfied).move_to(animation);

    printf("simulation ended at iteration %u\nTotal simulation time %f\n", i, totalTime);
    animation.save_ffmpeg_external(filename, 10, "libx264", 4048);
    animation.save("videos/GoL.bmp");
    free(gridAlive);
    free(gridUnsatisfied);


    return 0;
}
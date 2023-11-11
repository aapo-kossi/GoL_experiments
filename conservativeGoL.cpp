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
#include <iomanip>
#include <bit>
#include <random>
#include <immintrin.h>
#include <format>
#include <bitset>
#include <unordered_map>
#include "CImg.h"

#define BITNESS 64
#define SNAPRES 512

using namespace std;
using namespace cimg_library;

constexpr unsigned char staticGetUnsatisfied(uint16_t bitPattern) {
    unsigned char out = 0;
    const std::array<int, 8> neighbors = {-6,-5,-4,-1,1,4,5,6};
    for (int checked = 0; checked < 3; checked++) {
        bool isAlive = (bitPattern>>(6 + checked)) & 1;
        unsigned char nnCount = 0;
        for (int nn : neighbors) {
            nnCount += (bitPattern>>(6 + checked + nn)) & 1;
        }
        out |= ((!isAlive && (nnCount == 3)) || (isAlive && ((nnCount < 2) ||(nnCount > 3))))<<checked;
    }

    return out;
}

constexpr auto unsatisfiedLUT = [] {
    constexpr auto LUT_Size = 0x7FFF;
    std::array<unsigned char, LUT_Size> arr = {};

    for (int i = 0; i < LUT_Size; ++i)
    {
        arr[i] = staticGetUnsatisfied(i);
    }

    return arr;
}();

/***********************************************\
|  test lookup with the following bit patterns: |
|                                               |
|  1. 11111                 2. 10110            |
|     00000                    01000            |
|     00000                    00110            |
|  (all unsatisfied)        (all satisfied)     |
|                                               |
\***********************************************/
static_assert(unsatisfiedLUT[31744] == 0x07);
static_assert(unsatisfiedLUT[12365] == 0x00);

enum Action {
    INCREMENT = 1,
    DECREMENT = -1
};

struct CounterNode {
    unsigned int ltotal;
    unsigned int rtotal;
};

class CounterTree {
private:
    // get a reference to the leaf at index idx
    CounterNode& _leaf(unsigned int idx) {
        return nodes[numLeaf - 1 + idx];
    }

public:
    CounterTree(int N, vector<uint64_t> grid) {
        unsigned int trueNumLeaf = N*N/BITNESS + !!(N*N%BITNESS);
        depth = ceil(log2(trueNumLeaf));

        int accumulated_size = 0;
        for (int size = 1, i = 0; i<=depth; size *= 2, i++) {
            layerStarts.push_back(accumulated_size);
            layerSizes.push_back(size);
            accumulated_size += size;
        }
        numLeaf = (accumulated_size + 1)/2;

        // build the tree starting from the bottom layer (leaves)
        for (unsigned int i=0; i<numLeaf; i++) {
            unsigned int popcount;
            if (i < trueNumLeaf) {
                popcount = _popcnt64(grid[i]);
            } else {
                popcount = 0;
            }
            nodes.push_back(CounterNode { 0, popcount});
        }

        // add the narrower layers
        unsigned int layerWidth = numLeaf/2;
        int currentLayer = 1;
        int step = 2;
        while (layerWidth > 0) {
            vector<CounterNode> nLayer;
            for (int i=0; i<layerWidth; i++) {
                CounterNode lchild = nodes[i*2];
                CounterNode rchild = nodes[i*2+1];
                nLayer.push_back(CounterNode { lchild.ltotal + lchild.rtotal, rchild.ltotal + rchild.rtotal});
            }
            nodes.insert(nodes.begin(), nLayer.begin(), nLayer.end());
            layerWidth /= 2;
            currentLayer++;
        }
    }

    // get the total number of set bits in the grid
    unsigned int total() {
        return nodes[0].ltotal + nodes[0].rtotal;
    }

    // get the leaf at index idx
    CounterNode leaf(unsigned int idx) {
        return nodes[numLeaf - 1 + idx];
    }

    // O(log2(N))get the index of the leaf that has [n] set bits in total in leaves up to and including itself
    unsigned int getPosition(unsigned int n, unsigned int& totalBelow) {
        unsigned int treeIdx = 0;
        CounterNode current = nodes[treeIdx];

        for (int i = 0; i < depth; i++) {
            bool selectRight = ((totalBelow + current.ltotal) <= n) || !current.ltotal;
            totalBelow += current.ltotal * selectRight;
            treeIdx = 2*treeIdx + 1 +selectRight;
            current = nodes[treeIdx];
        }

        // index in the bottom layer of the tree is identical to the grid index
        return treeIdx + 1 - numLeaf;
    }

    // update the leaf at index idx and all its parents according to action (embarassingly parallel loop!)
    void update(unsigned int idx, int action) {
        for (int i = 0; i < depth; i++) {
            unsigned int layerIdx = idx >> (depth - i);
            unsigned int treeIdx = (1 << i) - 1 + layerIdx;
            CounterNode& current = nodes[treeIdx];

            if ( (idx >> (depth - i - 1)) & 1 ) {
                current.rtotal += action;
            } else {
                current.ltotal += action;
            }

        }
        CounterNode& last = _leaf(idx);
        last.rtotal += action;
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
    vector<unsigned int> layerSizes;
    vector<unsigned int> layerStarts;
    vector<CounterNode> nodes;
};

// Construct a counter tree for tracking alive, unsatisfied counts
class AliveCounters: public CounterTree {
private:
    vector<uint64_t> getGridUD(int N, uint64_t* gridA, uint64_t* gridU) {
        vector<uint64_t> ret(N*N/BITNESS+1);

        for (int i=0; i<(N*N/BITNESS+1); i++) {
            ret[i] = gridU[i] & gridA[i];
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
        vector<uint64_t> ret(N*N/BITNESS+1);

        for (int i=0; i<(N*N/BITNESS+1); i++) {
            ret[i] = gridU[i] & (~gridA[i]);
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

inline uint16_t getNNsUnsatisfied(uint64_t* grid, int N, uint64_t bitPos, int idx) {
    const int affectingRadius = 2;
    const int nRows = 2*affectingRadius + 1;
    int maxIdx = N*N;
    
    // build bit 5x5 bit pattern around index idx into the 32 bit nnnpacked
    // TODO: fix for grids that dont have cell count multiple of BITNESS
    int bitIdx = mod64(idx);
    uint64_t bitMask = 0;
    for (int i = -affectingRadius; i<= affectingRadius; i++) {
        bitMask |= std::rotl(bitPos, i);
    }
    uint64_t masked_neighbor;
    uint32_t nnnpacked = 0;
    for (int i=0; i<nRows; i++) {
        int n = (i-2)*N;
        int iBitIdx = mod64(bitIdx + n);
        uint64_t itermask = std::rotl(bitMask, n);
        if (iBitIdx < affectingRadius) {
            masked_neighbor = (grid[mod(idx + n, maxIdx) / BITNESS] & itermask) << (affectingRadius  - iBitIdx    )\
                                | (grid[mod(idx + n-BITNESS, maxIdx) / BITNESS] & itermask) >> (BITNESS + iBitIdx - affectingRadius);
        } else if (iBitIdx >= (BITNESS - affectingRadius)) {
            masked_neighbor = (grid[mod(idx + n, maxIdx) / BITNESS] & itermask) >> (iBitIdx - affectingRadius)\
                                | (grid[mod(idx + n+BITNESS, maxIdx) / BITNESS] & itermask) << (BITNESS - iBitIdx + affectingRadius);
        } else {
            masked_neighbor = (grid[mod(idx + n, maxIdx) / BITNESS] & itermask) >> (iBitIdx - affectingRadius);
        }
        nnnpacked |= std::rotl(masked_neighbor, i*nRows);
    }

    // look up which cells of the 3x3 nearest neighbors are unsatisfied, row by row
    uint16_t result = unsatisfiedLUT[nnnpacked & 0x7FFF];
    for (int row = 1; row < (nRows-2); row++) {
        uint16_t pattern = (nnnpacked >> (5*row)) & 0x7FFF;
        result |= unsatisfiedLUT[pattern] << (3*row);
    }
    return result;
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

    for (uint64_t i=0; i<(N*N);i++) {
        bool state = isUnsatisfied(grid, N, i);
        gridUnsatisfied[i/BITNESS] |= (uint64_t)state << mod64(i);
    }
}

inline void update_unsatisfied(
    int N,
    CounterTree& treeUD,
    CounterTree& treeUA,
    uint64_t* grid,
    uint64_t* gridU,
    uint64_t* pos,
    uint64_t* bitPos
) {
    array<int, 9> neighborhood = {-N-1, -N, -N+1, -1, 0, 1, N-1, N, N+1};

    int idxD = pos[0]*BITNESS + __builtin_ctzll(bitPos[0]);
    int idxA = pos[1]*BITNESS + __builtin_ctzll(bitPos[1]);
    uint16_t updatedUofD = getNNsUnsatisfied(grid, N, bitPos[0], idxD);
    uint16_t updatedUofA = getNNsUnsatisfied(grid, N, bitPos[1], idxA);

    // aggregate different neighbors that have identical position to avoid unnecessary updates
    std::unordered_map<uint64_t, int> totalIncrementsA(12);
    std::unordered_map<uint64_t, int> totalIncrementsD(12);

    int gridLength = N*N/BITNESS;

    for (int idx=0; idx < 9; idx++) {
        int i = mod(idxD + neighborhood[idx], N*N);
        int j = mod(idxA + neighborhood[idx], N*N);

        int nbrDPosition = i / BITNESS;
        int nbrAPosition = j / BITNESS;

        uint64_t nbrDbitPosition = std::rotl(bitPos[0], neighborhood[idx]);
        uint64_t nbrAbitPosition = std::rotl(bitPos[1], neighborhood[idx]);

        // add new unsatisfied state, subtract previous
        int nbrDIncrement = ((updatedUofD >> idx) & 1) - !!(gridU[nbrDPosition] & nbrDbitPosition);
        int nbrAIncrement = ((updatedUofA >> idx) & 1) - !!(gridU[nbrAPosition] & nbrAbitPosition);

        gridU[nbrDPosition] = (
            gridU[nbrDPosition] & ~nbrDbitPosition)
            | (nbrDbitPosition*!!(updatedUofD & (1 << idx)));

        gridU[nbrAPosition] = (
            gridU[nbrAPosition] & ~nbrAbitPosition)
            | (nbrAbitPosition*!!(updatedUofA & (1 << idx)));

        if (grid[nbrDPosition] & nbrDbitPosition) {
            // this neighbor of dead cell is alive
            totalIncrementsA[nbrDPosition] += nbrDIncrement;
        } else {
            // this neighbor of dead cell is dead
            totalIncrementsD[nbrDPosition] += nbrDIncrement;
        }

        if (grid[nbrAPosition] & nbrAbitPosition) {
            // this neighbor of alive cell is alive
            totalIncrementsA[nbrAPosition] += nbrAIncrement;
        } else {
            // this neighbor of alive cell is dead
            totalIncrementsD[nbrAPosition] += nbrAIncrement;
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
}

inline void unsatisfiedIdxToGridIdx(
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
    gridUnsatisfied[pos[0]] ^= bitPos[0];
    gridUnsatisfied[pos[1]] ^= bitPos[1];

    // update grid of unsatisfied cells and their counts
    update_unsatisfied(N, treeUD, treeUA, grid, gridUnsatisfied, pos, bitPos);
}

// progress bar function mostly by ChatGPT
void displayProgressBar(int i, int n, float measure) {
    // Calculate the percentage of completion
    float progress = static_cast<float>(i) / n;

    // Determine the width of the progress bar (here, it's set to 50 characters)
    int barWidth = 50;

    // Calculate the number of characters to represent completed progress
    int completedWidth = static_cast<int>(progress * barWidth);

    // Display the progress bar
    std::cout << "[";
    for (int j = 0; j < completedWidth; ++j) {
        std::cout << "=";
    }
    for (int j = completedWidth; j < barWidth; ++j) {
        std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(2) << progress * 100.0 << "% ("
              << i << "/" << n << ") " << std::fixed << measure << " simulation time" << "\r";
    std::cout.flush();

}

void kahanSum(float &accumulator,float value,float &correction) {
    float correctedValue = value-correction;
    value = accumulator + correctedValue;
    correction = (value-accumulator)-correctedValue;
    accumulator = value;
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
        "Maximum number of generated video frames: %i\n"
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

    // we may need such a large amount of steps that incrementing total time results in precision loss
    // To counteract this, we use Kahan summation for the increments
    float cFrameTime = 0;
    float cTotalTime = 0;

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

        kahanSum(frameTime, stepTime, cFrameTime);
        kahanSum(totalTime, stepTime, cTotalTime);

        if (!(i % 50000)) {
            displayProgressBar(i, numIter, totalTime);
        }
    }
    displayProgressBar(i, numIter, totalTime);
    std::cout << std::endl; // don't overwrite the progress bar

    getFrame(N, gridAlive, gridUnsatisfied).move_to(animation);

    printf("simulation ended at iteration %u\nTotal simulation time %f\n", i, totalTime);
    animation.save_ffmpeg_external(filename, 10, "libx264", 4048);
    animation.save("videos/GoL.bmp");
    free(gridAlive);
    free(gridUnsatisfied);


    return 0;
}

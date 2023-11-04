//////////////////////////////////////////////////////////////////////////////
// Program to simulate the conservative, probabilistic                      //
// version of Conway's Game of Life from                                    //
// https://journals.aps.org/pre/abstract/10.1103/PhysRevE.103.012132        //
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
#include <immintrin.h>
#include <format>
#include <bitset>
#include "CImg.h"

#define BITNESS 64
#define AVXBITNESS 512
#define SNAPRES 512

using namespace std;
using namespace cimg_library;

union U512u64 {
    __m512i v;
    uint64_t a[8];
};

// Extract the 64 bit integer in index idx of avx512i vector v
uint64_t extract512(__m512i v, int idx) {
    __m512d vd = _mm512_castsi512_pd(v);
    __m512i shuffled = _mm512_castsi128_si512( _mm_cvtsi32_si128(idx) );
    return bit_cast<uint64_t>(_mm512_cvtsd_f64( _mm512_permutexvar_pd(shuffled, vd) ));
}

// Modulo that always returns the first non-negative value
unsigned mod(int x, int y) {
    int mod = x % y;
    if (mod < 0) {
        mod += y;
    }
    return mod;
}

// compile-time optimizeable version for fast mod of BITNESS
unsigned mod64(int x) {
    return mod(x, BITNESS);
}

// Calculate pixel intensities for given configuration cells
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
        img.resize(SNAPRES,SNAPRES);
    }
    return img;
}

// Randomly place alive cells on an N*N grid represented by an array of uint64_t
void initialize(
    int N,
    std::bernoulli_distribution* d,
    std::mt19937* gen,
    uint64_t* grid
) {
    for (uint64_t i=0; i<(N*N); i++) {
        grid[i/BITNESS] |= static_cast<uint64_t>((*d)(*gen)) << mod64(i);
    }

}

// Check whether a single cell is unsatisfied
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

// Helper
void print_numbers(int N, uint64_t* grid) {
    uint64_t n_alive = 0;
    for (int i=0; i< N*N/BITNESS; i++) {
        n_alive += _popcnt64(grid[i]);
    }
    printf("%u\n", n_alive);
}

// Simple loop to initialize the grid of unsatisfied cells
void get_all_unsatisfied(
    int N,
    uint64_t* grid,
    uint64_t* gridUnsatisfied,
    uint64_t& countD,
    uint64_t& countA
) {
    for (uint64_t i=0; i<(N*N);i++) {
        uint64_t state = static_cast<uint64_t>(isUnsatisfied(grid, N, i));
        gridUnsatisfied[i/BITNESS] |= state << mod64(i);
        countD += state & !((grid[i/BITNESS] >> mod64(i)) & 1);
        countA += state &  ((grid[i/BITNESS] >> mod64(i)) & 1);
    }
}

// O(1) complexity loop over each neighboring cell of the swapped cells,
// check if they are satisfied or not
void update_unsatisfied(
    int N,
    uint64_t& countUD,
    uint64_t& countUA,
    uint64_t* grid,
    uint64_t* gridU,
    uint64_t* pos,
    uint64_t* bitPos
) {
    // Define grid indices of a Moore neighborhood
    const vector<int> neighborhood{ -1-N, -N, 1-N, -1, 1, N-1, N, N+1};
    int gridLength = N*N/BITNESS;

    // Loop over neighbors
    for (uint64_t idx=0; idx < 8; idx++) {

        // Get grid indices of the neighbors of the swapped cells
        uint64_t i = mod(pos[0]*BITNESS + __builtin_ctzll(bitPos[0]) + neighborhood[idx], N*N);
        uint64_t j = mod(pos[1]*BITNESS + __builtin_ctzll(bitPos[1]) + neighborhood[idx], N*N);

        // Get array indices and bit positions of the swapped cells
        uint64_t iPosD = pos[0];
        if ((neighborhood[idx] < 0) && !(bitPos[0] >> -neighborhood[idx])) {
            iPosD--;
        } else if ((neighborhood[idx] > 0) && !(bitPos[0] << neighborhood[idx])) {
            iPosD++;
        }
        iPosD = mod(iPosD, gridLength);

        uint64_t iPosA = pos[1];
        if ((neighborhood[idx] < 0) && !(bitPos[1] >> -neighborhood[idx])) {
            iPosA--;
        } else if ((neighborhood[idx] > 0) && !(bitPos[1] << neighborhood[idx])) {
            iPosA++;
        }
        iPosA = mod(iPosA, gridLength);

        uint64_t iBitPosD = std::rotl(bitPos[0], neighborhood[idx]);
        uint64_t iBitPosA = std::rotl(bitPos[1], neighborhood[idx]);

        // Check if the cells are satisfied
        bool ui = isUnsatisfied(grid, N, i);
        bool uj = isUnsatisfied(grid, N, j);


        // counters: add new unsatisfied state, subtract previous
        int dNeighborIncr = ui - !!(gridU[iPosD] & iBitPosD);
        int aNeighborIncr = uj - !!(gridU[iPosA] & iBitPosA);
        if (grid[iPosD] & iBitPosD) {
            countUA += dNeighborIncr;
        } else {
            countUD += dNeighborIncr;
        }
        if (grid[iPosA] & iBitPosA) {
            countUA += aNeighborIncr;
        } else {
            countUD += aNeighborIncr;
        }

        // Set the grid bit of the cells to their new state
        gridU[iPosD] = (gridU[iPosD] & ~iBitPosD) | (iBitPosD*ui);
        gridU[iPosA] = (gridU[iPosA] & ~iBitPosA) | (iBitPosA*uj);

    }
}

// O(N^2) complexity, this function dominates runtime for large grids
// This is why slight optimization efforts are taken to make it faster
void unsatisfiedIdxToGridIdx(
    int N,
    uint64_t* gridA,
    uint64_t* gridU,
    uint64_t& nd,
    uint64_t& na,
    uint64_t* pos,
    uint64_t* bitPos
) {

    // declarations
    uint64_t numUD = 0;
    uint64_t numUA = 0;
    bool done = false;
    bool foundA = false;
    bool foundD = false;
    int i = 0;
    int incr = AVXBITNESS / BITNESS;

    while (!done) {

        // load 8 consecutive elements into 64 byte AVX-512 registers
        __m512i aliveBlock = _mm512_load_epi64(&gridA[i]);
        __m512i unsatisfiedBlock = _mm512_load_epi64(&gridU[i]);

        // get unsatisfied dead cells from the block
        __m512i blockUD = _mm512_andnot_epi64(aliveBlock, unsatisfiedBlock);

        // population counts of u,d cells in each 8 byte uint
        U512u64 numUDblock = { _mm512_popcnt_epi64(blockUD) };

        // total population count in the register
        uint64_t totalUDblock = _mm512_reduce_add_epi64(numUDblock.v);

        // Enter if population count passes our index to be swapped nd
        if (!foundD && (numUD + totalUDblock > nd)) {

            // find the exact position inside the block
            int k = 0;
            uint64_t accumulator = numUDblock.a[0];
            foundD = (numUD + accumulator) > nd;
            while (!foundD) {
                k++;
                accumulator += numUDblock.a[k];
                foundD = (numUD + accumulator) > nd;
            }
            // record the position of the uint containing the nd'th u,d cell
            pos[0] = i + k;

            // record the bit location inside the uint where the cell is
            uint64_t subblockUD = extract512(blockUD, k);
            bitPos[0] = _pdep_u64(1UL << (numUD + accumulator - nd - 1), subblockUD);
        }
        numUD += totalUDblock;

        // now do the above for unsatisfied alive cells
        __m512i blockUA = _mm512_and_epi64(unsatisfiedBlock, aliveBlock);
        U512u64 numUAblock = { _mm512_popcnt_epi64(blockUA) };
        uint64_t totalUAblock = _mm512_reduce_add_epi64(numUAblock.v);
        if (!foundA && (numUA + totalUAblock > na)) {
            int k = 0;
            uint64_t accumulator = numUAblock.a[0];
            foundA = (numUA + accumulator) > na;
            while (!foundA) {
                k++;
                accumulator += numUAblock.a[k];
                foundA = (numUA + accumulator) > na;
            }
            pos[1] = i + k;
            uint64_t subblockUA = extract512(blockUA, k);
            bitPos[1] = _pdep_u64(1UL << (numUA + accumulator - na - 1), subblockUA);
        }
        numUA += totalUAblock;

        // If we have found both the u,d and u,a cell, we can exit
        done = foundA && foundD;
        i += incr;
    }
}

void step(
    int N,
    uint64_t& countUD,
    uint64_t& countUA,
    uint64_t* grid,
    uint64_t* gridUnsatisfied,
    uint64_t* pos,
    uint64_t* bitPos,
    mt19937& gen
) {

    // sample one unsatisfied dead, one unsatisfied alive cell
    uniform_int_distribution<> dD(0, countUD - 1);
    uniform_int_distribution<> dA(0, countUA - 1);
    uint64_t idxd = dD(gen);
    uint64_t idxa = dA(gen);

    // populate positions of cells to be swapped
    unsatisfiedIdxToGridIdx(N, grid, gridUnsatisfied, idxd, idxa, pos, bitPos);

    // swap cells
    grid[pos[0]] ^= bitPos[0];
    grid[pos[1]] ^= bitPos[1];
    countUD--;
    countUA--;

    //check if swapped cells are unsatisfied, update counts
    // (these can only be unsatisfied if we swapped neighboring cells)
    if (isUnsatisfied(grid, N, pos[0]*BITNESS + __builtin_ctzll(bitPos[0]))) {
        countUA++;
    } else {
        gridUnsatisfied[pos[0]] ^= bitPos[0];
    }
    if (isUnsatisfied(grid, N, pos[1]*BITNESS + __builtin_ctzll(bitPos[1]))) {
        countUD++;
    } else {
        gridUnsatisfied[pos[1]] ^= bitPos[1];
    }

    // update grid of unsatisfied cells and their counts
    update_unsatisfied(N, countUD, countUA, grid, gridUnsatisfied, pos, bitPos);
}

int main(int argc, char** argv) {

    char* filename;

    if (argc < 7) {
        printf("Required arguments: grid-size, p-dead, n-iters, n-video-frames, video-freq, rng-seed\n");
        printf("Optional argument: output video file name");
        return 1;
    } else if (argc < 8) {
        filename = const_cast<char*>("GoL_video.raw");
    } else {
        filename = argv[7];
    }

    // interpret command line arguments
    int N = atoi(argv[1]);
    float p = atof(argv[2]);
    uint64_t numIter = atof(argv[3]);
    uint64_t numFrames = atoi(argv[4]);
    float freq = atof(argv[5]);
    uint64_t seed = atoi(argv[6]);

    // create random generator and initial cell distribution
    mt19937 gen(seed);
    bernoulli_distribution d(1-p);

    // print run information
    printf(
        "Running conservative Game of Life.\n"
        "Grid size: %i\n"
        "Initialization probability of cell being dead: %f\n"
        "Maximum number of iterations: %i\n"
        "Maximum numbe of generated video frames: %i\n"
        "Animation frequenof 1/dt: %f\n",
        N, p, numIter, numFrames, freq);

    //declarations
    int allocGridSize = 64 * (1 + N*N/(BITNESS*8));

    uint64_t* gridAlive = (uint64_t*)_mm_malloc(allocGridSize, 64);
    memset(gridAlive, 0, allocGridSize);

    uint64_t* gridUnsatisfied = (uint64_t*)_mm_malloc(allocGridSize, 64);
    memset(gridUnsatisfied, 0, allocGridSize);

    uint64_t countUD = 0;
    uint64_t countUA = 0;

    // place intial cells
    initialize(N, &d, &gen, gridAlive);

    // calculate which cells are initially unsatisfied
    // and get their total count
    get_all_unsatisfied(N, gridAlive, gridUnsatisfied, countUD, countUA);

    // Create container for system snapshots, save the initial configuration
    CImgList<unsigned char> animation;
    getFrame(N, gridAlive, gridUnsatisfied).move_to(animation);

    // Loop variable declarations
    uint64_t i = 0;
    uint64_t pos[2];
    uint64_t bitPos[2];
    float frameTime = 0;
    float totalTime = 0;
    uint64_t currentFrame = 1; // The first simulation frame is already plotted
    float maxFrameTime = 1.0f/freq;

    // main loop for iterating the system
    for (i; i<numIter; i++) {

        // exit if one type of unsatisfied cell runs out
        if (!countUD || !countUA) {
            break;
        }

        // possibly save the current state for drawing
        float diff = frameTime - maxFrameTime;
        if (currentFrame < (numFrames-1) && (diff >= 0)) {
            getFrame(N, gridAlive, gridUnsatisfied).move_to(animation);
            frameTime = diff;
            currentFrame += 1;
        }

        // advance the system a single time step
        step(N, countUD, countUA, gridAlive, gridUnsatisfied, pos, bitPos, gen);

        // normalize the time step
        float stepTime = 1.0f/(countUD + countUA);
        frameTime += stepTime;
        totalTime += stepTime;
    }

    // Always save the final state of the system
    getFrame(N, gridAlive, gridUnsatisfied).move_to(animation);

    printf("simulation ended at iteration %u\nTotal simulation time %f\n", i, totalTime);

    // write output as video and separate frames
    animation.save_ffmpeg_external(filename, 10, "libx264", 4048);
    animation.save("videos/GoL.bmp");

    free(gridAlive);
    free(gridUnsatisfied);

    return 0;
}

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
#include <intrin.h>
#include <format>
#include <bitset>
#include "Cimg.h"

#define BITNESS 64

using namespace std;
using namespace cimg_library;

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

bool isUnsatisfied(uint64_t* grid, int N, int i) {       // verified working as intended! yay
    int Nsquared = N*N;
    bool isAlive = 1U & (grid[i/BITNESS] >> mod64(i));
    uint64_t n = 0;
    const vector<int> neighborhood = { -1-N, -N, 1-N, -1, 1, N-1, N, N+1}; // this can wrap wrong :((((, creates 1 offset between left and right edges, insignificant
    for (uint64_t idx=0; idx < neighborhood.size(); idx++) {
        uint64_t j = mod(neighborhood[idx] + i, Nsquared);
        n += 1U & (grid[j/BITNESS] >> mod64(j));
    }

    if (isAlive) {
        // printf("alive cell, %i\n", n);
        return (n != 2) && (n != 3);
    }
    else {
        // printf("dead cell, %i\n", n);
        return n == 3;
    }
}

void print_numbers(int N, uint64_t* grid) {
    uint64_t n_alive = 0;
    for (int i=0; i< N*N/BITNESS; i++) {
        n_alive += __popcnt64(grid[i]);
    }
    printf("%u\n", n_alive);
}

void get_all_unsatisfied(
    int N,
    uint64_t* grid,
    uint64_t* gridUnsatisfied,
    uint64_t& countD,
    uint64_t& countA
) {
    for (uint64_t i=0; i<(N*N);i++) {
        bool state = isUnsatisfied(grid, N, i);
        gridUnsatisfied[i/BITNESS] |= (uint64_t)state << mod64(i);
        countD += state & !((grid[i/BITNESS] >> mod64(i)) & 1);
        countA += state &  ((grid[i/BITNESS] >> mod64(i)) & 1);
    }
}

void update_unsatisfied(
    int N,
    uint64_t& countUD,
    uint64_t& countUA, 
    uint64_t* grid,
    uint64_t* gridU,
    uint64_t* pos,
    uint64_t* bitPos
) {
    const vector<int> neighborhood{ -1-N, -N, 1-N, -1, 1, N-1, N, N+1};
    int gridLength = N*N/BITNESS;
    for (uint64_t idx=0; idx < 8; idx++) {
        uint64_t i = mod(pos[0]*BITNESS + __builtin_ctzll(bitPos[0]) + neighborhood[idx], N*N);
        uint64_t j = mod(pos[1]*BITNESS + __builtin_ctzll(bitPos[1]) + neighborhood[idx], N*N);
        
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

        bool ui = isUnsatisfied(grid, N, i);
        bool uj = isUnsatisfied(grid, N, j);
        
        // printf("i = %u, ui = %d, bpi = %i, pi = %i\n", i, ui, __builtin_ctzll(iBitPosD), iPosD);
        // printf("j = %d, uj = %d, bpj = %i, pj = %i\n", j, uj, __builtin_ctzll(iBitPosA), iPosA);

        // add new unsatisfied state, subtract previous
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
        
        // printf("countUD = %u, countUA = %u\n", countUD, countUA);

        // std::bitset<64> set(gridU[0]);
        // update the grid
        // std::cout << set << std::endl;
        gridU[iPosD] = (gridU[iPosD] & ~iBitPosD) | (iBitPosD*ui);
        gridU[iPosA] = (gridU[iPosA] & ~iBitPosA) | (iBitPosA*uj);
        // set = gridU[0];
        // std::cout << set << std::endl;

    }
}

// O(N^2) complexity, this function dominates runtime for large grids
void unsatisfiedIdxToGridIdx(
    int N,
    uint64_t* gridA,
    uint64_t* gridU,
    uint64_t& nd,
    uint64_t& na,
    uint64_t* pos,
    uint64_t* bitPos
) {

    uint64_t numUD = 0;
    uint64_t numUA = 0;
    bool done = false;
    bool foundA = false;
    bool foundD = false;
    int i = 0;

    while (!done) {
        // if (i > N*N/BITNESS) {
        //     break;
        // }
        uint64_t blockUD = (~gridA[i]) & gridU[i];
        uint64_t blockUA = gridA[i] & gridU[i];
        uint64_t numUDblock = __popcnt64(blockUD);
        uint64_t numUAblock = __popcnt64(blockUA);
        if (!foundD && (numUD + numUDblock > nd)) {
            pos[0] = i;
            bitPos[0] = _pdep_u64(1UL << (nd - numUD), blockUD);
            foundD = true;
            // printf("numUD = %u, numUDblock = %u, blockUD = %llu, nd = %u, bitPos = %llu\n", numUD, numUDblock, blockUD, nd, bitPos[0]);
        }
        if (!foundA && (numUA + numUAblock > na)) {
            pos[1] = i;
            bitPos[1] = _pdep_u64(1UL << (na - numUA), blockUA);
            foundA = true;
            // printf("numUA = %u, numUAblock = %u, blockUA = %llu, na = %u, bitPos = %llu\n", numUA, numUAblock, blockUA, na, bitPos[1]);
        }
        numUD += numUDblock;
        numUA += numUAblock;
        done = foundA && foundD;
        i++;
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
    uniform_int_distribution<> dD(0, countUD - 1);
    uniform_int_distribution<> dA(0, countUA - 1);
    uint64_t idxd = dD(gen);
    uint64_t idxa = dA(gen);
    unsatisfiedIdxToGridIdx(N, grid, gridUnsatisfied, idxd, idxa, pos, bitPos); // populate positions of cells to be swapped

    // swap cells
    grid[pos[0]] ^= bitPos[0];
    grid[pos[1]] ^= bitPos[1];
    countUD--;
    countUA--;


    //check if swapped cells are unsatisfied, update counts
    if (isUnsatisfied(grid, N, pos[0]/BITNESS + __builtin_ctzll(bitPos[0]))) {
        countUA++;
    } else {
        gridUnsatisfied[pos[0]] ^= bitPos[0];
    }
    if (isUnsatisfied(grid, N, pos[1]/BITNESS + __builtin_ctzll(bitPos[1]))) {
        countUD++;
    } else {
        gridUnsatisfied[pos[1]] ^= bitPos[1];
    }


    // printf("idxd = %u\n", idxd);
    // printf("idxa = %u\n", idxa);
    // printf("pos0 = %u\n", pos[0]);
    // printf("pos1 = %u\n", pos[1]);
    // printf("bp0 = %llu\n", __builtin_ctzll(bitPos[0]));
    // printf("bp1 = %llu\n", __builtin_ctzll(bitPos[1]));


    // update grid of unsatisfied cells and their counts
    update_unsatisfied(N, countUD, countUA, grid, gridUnsatisfied, pos, bitPos);
    // printf("%u, %u\n", countUD, countUA);
    // print_numbers(N, grid);
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
    uint64_t countUD = 0;
    uint64_t countUA = 0;
    initialize(N, &d, &gen, gridAlive);

    get_all_unsatisfied(N, gridAlive, gridUnsatisfied, countUD, countUA);

    // printf("%u,%u\n", countUD, countUA);
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
        if (!countUD || !countUA) {
            break;
        }
        float diff = frameTime - maxFrameTime;
        if (currentFrame < (numFrames-1) && (diff >= 0)) {
            getFrame(N, gridAlive, gridUnsatisfied).move_to(animation);
            frameTime = diff;
            currentFrame += 1;
        }
        step(N, countUD, countUA, gridAlive, gridUnsatisfied, pos, bitPos, gen);
        float stepTime = 1.0f/(countUD + countUA);
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
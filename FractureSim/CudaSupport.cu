#include "CudaSupport.h"
#include "stdio.h"


// GPU Error check helper
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPU ASSERT: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace CudaSupport {

const unsigned int MaxBlockSize = 512;

unsigned int blockSize;
unsigned int numBlocks;

// Device constants - initialized once for usage
__device__ __constant__ unsigned int numOfParticles;
__device__ __constant__ unsigned int hashBinsMaxSize;
__device__ __constant__ unsigned int hashBinsNum;
__device__ __constant__ unsigned int hashBinsNumHalf;
__device__ __constant__ double gridCellSize;
__device__ __constant__ unsigned int p1 = 73856093;
__device__ __constant__ unsigned int p2 = 19349663;
__device__ __constant__ unsigned int p3 = 83492791;
__device__ __constant__ double dt;
__device__ __constant__ double half_dt;
__device__ __constant__ double Kc;
__device__ __constant__ double3 gravity;
__device__ __constant__ double stiffness;
__device__ __constant__ double3 x_offset;
__device__ __constant__ double3 y_offset;
__device__ __constant__ double3 z_offset;
__device__ __constant__ double collisionThreshold;
__device__ __constant__ double lambda;



/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Start of device helpers of CUDA kernels //////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ double3 operator+(const double3& a, const double3& b)
{
	double3 out;
	out.x = a.x + b.x;
	out.y = a.y + b.y;
	out.z = a.z + b.z;
	return out;
}

__device__ __forceinline__ double3 operator-(const double3& a, const double3& b)
{
	double3 out;
	out.x = a.x - b.x;
	out.y = a.y - b.y;
	out.z = a.z - b.z;
	return out;
}

__device__ __forceinline__ double3 operator*(const double3& a, const double b)
{
	double3 out;
	out.x = a.x * b;
	out.y = a.y * b;
	out.z = a.z * b;
	return out;
}

__device__ __forceinline__ double3 operator*(const double b, const double3& a)
{
	double3 out;
	out.x = a.x * b;
	out.y = a.y * b;
	out.z = a.z * b;
	return out;
}

__device__ __forceinline__ double norm(const double3& v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// End device helpers CUDA kernels //////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Start of CUDA kernels ////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

__global__
void initializeKernel(
	unsigned int* hashCounts)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < hashBinsNum; i += stride) {
		hashCounts[i] = 0;
	}
}


__device__
unsigned int spatialHash(
	const double3& pos)
{
	long long int i = floor(pos.x / gridCellSize);
	long long int j = floor(pos.y / gridCellSize);
	long long int k = floor(pos.z / gridCellSize);

	return (((i * p1) ^ (j * p2) ^ (k ^ p3)) % hashBinsNumHalf) + hashBinsNumHalf;
}


__global__
void collectKernel(
	double3* positions,
	unsigned int* hashTable,
	unsigned int* hashCounts)
{
	// Fill the hash table
	unsigned int hash;
	unsigned int idx;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < numOfParticles; i += stride) {
		hash = spatialHash(positions[i]);
		idx = hash * hashBinsMaxSize + atomicAdd(&hashCounts[hash], 1);
		hashTable[idx] = i;
	}
}


__global__
void detectCollisionsKernel(
	double3* positions,
	double3* forces,
	unsigned int* hashTable,
	unsigned int* hashCounts)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	// Collision detection
	for (int i = index; i < numOfParticles; i += stride) {
		// Get cells to check for each particle
		unsigned int cellsToCheck[8];
		unsigned int cellsToCheck_duplicates[8];
		double3 position = positions[i];

		// Hash all AABB vertices
		cellsToCheck_duplicates[0] = spatialHash(position - x_offset - y_offset - z_offset);
		cellsToCheck_duplicates[1] = spatialHash(position + x_offset - y_offset - z_offset);
		cellsToCheck_duplicates[2] = spatialHash(position - x_offset + y_offset - z_offset);
		cellsToCheck_duplicates[3] = spatialHash(position - x_offset - y_offset + z_offset);
		cellsToCheck_duplicates[4] = spatialHash(position + x_offset + y_offset - z_offset);
		cellsToCheck_duplicates[5] = spatialHash(position + x_offset - y_offset + z_offset);
		cellsToCheck_duplicates[6] = spatialHash(position - x_offset + y_offset + z_offset);
		cellsToCheck_duplicates[7] = spatialHash(position + x_offset + y_offset + z_offset);

		unsigned int numCellsToCheck = 0;

		bool dupl;
		for (int i = 0; i < 8; ++i) {
			dupl = false;
			for (int j = 0; j < numCellsToCheck; ++j) {
				if (cellsToCheck_duplicates[i] == cellsToCheck[j]) {
					dupl = true;
					break;
				}
			}
			if (!dupl) {
				cellsToCheck[numCellsToCheck++] = cellsToCheck_duplicates[i];
			}
		}
		
		// Check all the cells - if they are colliding, compute response
		unsigned int nextCell, start;
		for (int j = 0; j < numCellsToCheck; ++j) {
			nextCell = cellsToCheck[j];
			start = nextCell * hashBinsMaxSize;
			for (int k = start; k < start + hashCounts[nextCell]; ++k) {
				if (hashTable[k] != i) {
					double3 diff = positions[i] - positions[hashTable[k]];
					double distance = norm(diff);
					if (distance < 1e-9) continue;
					if (distance < collisionThreshold) {
						//printf("Particles %d and %d are colliding!\n", i, hashTable[k]);
						forces[i] =
							forces[i] + (Kc * pow(distance - collisionThreshold, 2) / distance) * diff;
					}
				}
			}
		}
	}
}


__global__
void advanceVelocitiesKernel(
	double3* velocities,
	double3* forces,
	double massInv)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < numOfParticles; i += stride) {
		velocities[i] = velocities[i] + half_dt * massInv * forces[i];
	}
}


__global__
void advancePositionsKernel(
	double3* positions,
	double3* velocities)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < numOfParticles; i += stride) {
		positions[i] = positions[i] + dt * velocities[i];
	}
}


__global__
void addBodyForcesKernel(
	double3* positions,
	double3* velocities,
	double3* forces,
	double particleMass)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < numOfParticles; i += stride) {
		if (positions[i].y < 0) {
			// Restore position, otherwise particle might remain stuck under the floor (?)
			//positions[i].y = 0;
			// Reflect velocity
			velocities[i].y *= -1;
		}
		forces[i] = particleMass * gravity;
	}
}


__global__
void addSpringForcesKernel(
	double3* positions,
	double3* forces,
	int* adjs,
	unsigned int* adjsCounts,
	unsigned int* adjsStarts,
	double* restLengths,
	double* taus)
{
	int start, end;
	double epsilon, distance;
	double3 diff;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < numOfParticles; i += stride) {
		start = adjsStarts[i];
		end = start + adjsCounts[i];
		for (int j = start; j < end; ++j) {
			if (adjs[j] != -1) {
				diff = positions[i] - positions[adjs[j]];
				distance = norm(diff);
				if (distance <= 1e-9) continue;
				epsilon = (distance / restLengths[j]) - 1;
				if (epsilon > taus[i]) {
					//printf("The spring between %d and %d broke!\n", i, adjs[j]);
					adjs[j] = -1;
					continue;
				}

				if (epsilon != 0) {
					forces[i] = forces[i] + diff * (-1 * stiffness * epsilon / distance);
				}
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// End of CUDA kernels //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Start of simulation interface ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void initializeSimulationParameters(
	unsigned int numOfParticles_host,
	unsigned int hashBinsNum_host,
	unsigned int hashBinsMaxSize_host,
	double gridCellSize_host,
	double dt_host,
	double Kc_host,
	double stiffness_host,
	double avgtau_host,
	double collisionThreshold_host,
	double lambda_host,
	thrust::device_vector<double3>& velocities,
	thrust::device_vector<double3>& forces,
	thrust::device_vector<double>& taus,
	thrust::device_vector<unsigned int>& hashTable,
	thrust::device_vector<unsigned int>& hashCounts)
{
	cudaMemcpyToSymbol(numOfParticles, &numOfParticles_host, sizeof(unsigned int));
	cudaMemcpyToSymbol(hashBinsMaxSize, &hashBinsMaxSize_host, sizeof(unsigned int));
	cudaMemcpyToSymbol(hashBinsNum, &hashBinsNum_host, sizeof(unsigned int));
	unsigned int hashBinsNumHalf_host = hashBinsNum_host / 2;
	cudaMemcpyToSymbol(hashBinsNumHalf, &hashBinsNumHalf_host, sizeof(unsigned int));
	cudaMemcpyToSymbol(gridCellSize, &gridCellSize_host, sizeof(double));
	cudaMemcpyToSymbol(dt, &dt_host, sizeof(double));
	double half_dt_host = dt_host / 2;
	cudaMemcpyToSymbol(half_dt, &half_dt_host, sizeof(double));
	cudaMemcpyToSymbol(Kc, &Kc_host, sizeof(double));
	double3 gravity_host = make_double3(0, -9.81, 0);
	cudaMemcpyToSymbol(gravity, &gravity_host, sizeof(double3));
	cudaMemcpyToSymbol(stiffness, &stiffness_host, sizeof(double));
	cudaMemcpyToSymbol(collisionThreshold, &collisionThreshold_host, sizeof(double));
	cudaMemcpyToSymbol(lambda, &lambda_host, sizeof(double));

	// Number of threads per block
	blockSize = (numOfParticles_host > MaxBlockSize ? MaxBlockSize : numOfParticles_host);
	// Number of blocks (to avoid overlapping)
	numBlocks = (numOfParticles_host + blockSize - 1) / blockSize;

	double3 x_offset_host = make_double3(lambda_host / 2, 0, 0);
	double3 y_offset_host = make_double3(0, lambda_host / 2, 0);
	double3 z_offset_host = make_double3(0, 0, lambda_host / 2);
	cudaMemcpyToSymbol(x_offset, &x_offset_host, sizeof(double3));
	cudaMemcpyToSymbol(y_offset, &y_offset_host, sizeof(double3));
	cudaMemcpyToSymbol(z_offset, &z_offset_host, sizeof(double3));

	double3 zeroVector = make_double3(0, 0, 0);
	velocities.resize(numOfParticles_host);
	thrust::fill(thrust::device, velocities.begin(), velocities.end(), zeroVector);
	forces.resize(numOfParticles_host);
	thrust::fill(thrust::device, forces.begin(), forces.end(), zeroVector);
	taus.resize(numOfParticles_host);
	thrust::fill(thrust::device, taus.begin(), taus.end(), avgtau_host);
	hashTable.resize(hashBinsMaxSize_host * hashBinsNum_host);
	thrust::fill(thrust::device, hashTable.begin(), hashTable.end(), 0);
	hashCounts.resize(hashBinsNum_host);
	thrust::fill(thrust::device, hashCounts.begin(), hashCounts.end(), 0);
}


void iterate(
	thrust::host_vector<double3>& positionsHost,
	thrust::device_vector<double3>& positions,
	thrust::device_vector<double3>& velocities,
	thrust::device_vector<double3>& forces,
	thrust::host_vector<int>& adjsHost,
	thrust::device_vector<int>& adjs,
	thrust::device_vector<unsigned int>& adjsCounts,
	thrust::device_vector<unsigned int>& adjsStarts,
	thrust::device_vector<double>& restLengths,
	thrust::device_vector<double>& taus,
	thrust::device_vector<unsigned int>& hashTable,
	thrust::device_vector<unsigned int>& hashCounts,
	unsigned int numberOfIterations,
	double particleMass,
	double particleMassInv)
{
	// Get raw pointers to pass to kernels
	unsigned int* hashTable_ptr = thrust::raw_pointer_cast(hashTable.data());
	unsigned int* hashCounts_ptr = thrust::raw_pointer_cast(hashCounts.data());
	double3* positions_ptr = thrust::raw_pointer_cast(positions.data());
	double3* forces_ptr = thrust::raw_pointer_cast(forces.data());
	double3* velocities_ptr = thrust::raw_pointer_cast(velocities.data());
	int* adjs_ptr = thrust::raw_pointer_cast(adjs.data());
	unsigned int* adjsCounts_ptr = thrust::raw_pointer_cast(adjsCounts.data());
	unsigned int* adjsStarts_ptr = thrust::raw_pointer_cast(adjsStarts.data());
	double* restLengths_ptr = thrust::raw_pointer_cast(restLengths.data());
	double* taus_ptr = thrust::raw_pointer_cast(taus.data());

	for (int i = 0; i < numberOfIterations; ++i) {
		// Initialize hash bins for the next iteration
		initializeKernel<<<numBlocks, blockSize>>>(hashCounts_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// Collect - assign each particle to hash bin
		collectKernel<<<numBlocks, blockSize>>>(positions_ptr, hashTable_ptr, hashCounts_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// Detect collisions and compute response
		detectCollisionsKernel<<<numBlocks, blockSize>>>(positions_ptr, forces_ptr, hashTable_ptr, hashCounts_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// Advance velocities by half-time step (first Velocity-Verlet update)
		advanceVelocitiesKernel<<<numBlocks, blockSize>>>(velocities_ptr, forces_ptr, particleMassInv);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// Advance positions by one timestep (second Velocity-Verlet update
		advancePositionsKernel<<<numBlocks, blockSize>>>(positions_ptr, velocities_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// Add body forces: gravity + collision with the floor
		addBodyForcesKernel<<<numBlocks, blockSize>>>(positions_ptr, velocities_ptr, forces_ptr, particleMass);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// Add spring forces
		addSpringForcesKernel<<<numBlocks, blockSize>>>(
			positions_ptr, forces_ptr, adjs_ptr, adjsCounts_ptr, adjsStarts_ptr, restLengths_ptr, taus_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// Advance velocities (third Velocitiy-Verlet update)
		advanceVelocitiesKernel<<<numBlocks, blockSize>>>(velocities_ptr, forces_ptr, particleMassInv);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

	}

	// Synchronize GPU and CPU before copying the data back
	gpuErrchk(cudaDeviceSynchronize());

	thrust::copy(positions.begin(), positions.end(), positionsHost.begin());
	thrust::copy(adjs.begin(), adjs.end(), adjsHost.begin());
}


void resetVelocitiesAndForces(
	thrust::device_vector<double3>& velocities,
	thrust::device_vector<double3>& forces
)
{
	double3 zeroVector = make_double3(0, 0, 0);
	thrust::fill(thrust::device, velocities.begin(), velocities.end(), zeroVector);
	thrust::fill(thrust::device, forces.begin(), forces.end(), zeroVector);
}

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// End of simulation interface //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Start of kernel unit-testing helpers /////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


	void* allocateDeviceMemory(unsigned int size)
	{
		void* ptr;
		cudaMalloc(&ptr, size);
		return ptr;
	}

	void freeDeviceMemory(void* ptr) {
		cudaFree(ptr);
	}

	void copyToDevice(void* devPtr, void* dataPtr, unsigned int size) {
		cudaMemcpy(devPtr, dataPtr, size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}

	void copyFromDevice(void* dataPtr, void* devPtr, unsigned int size) {
		cudaMemcpy(dataPtr, devPtr, size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}

	void initialize(
		thrust::device_vector<unsigned int>& hashCounts) 
	{
		unsigned int* hashCounts_ptr = thrust::raw_pointer_cast(&hashCounts[0]);
		initializeKernel<<<numBlocks, blockSize>>>(hashCounts_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	void collect(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<unsigned int>& hashTable,
		thrust::device_vector<unsigned int>& hashCounts)
	{
		double3* positions_ptr = thrust::raw_pointer_cast(&positions[0]);
		unsigned int* hashTable_ptr = thrust::raw_pointer_cast(&hashTable[0]);
		unsigned int* hashCounts_ptr = thrust::raw_pointer_cast(&hashCounts[0]);
		collectKernel<<<numBlocks, blockSize>>>(positions_ptr, hashTable_ptr, hashCounts_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	void detectCollisions(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<double3>& forces,
		thrust::device_vector<unsigned int>& hashTable,
		thrust::device_vector<unsigned int>& hashCounts)
	{
		double3* positions_ptr = thrust::raw_pointer_cast(positions.data());
		double3* forces_ptr = thrust::raw_pointer_cast(forces.data());
		unsigned int* hashTable_ptr = thrust::raw_pointer_cast(hashTable.data());
		unsigned int* hashCounts_ptr = thrust::raw_pointer_cast(hashCounts.data());
		detectCollisionsKernel<<<numBlocks, blockSize>>>(positions_ptr, forces_ptr, hashTable_ptr, hashCounts_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	void advanceVelocities(
		thrust::device_vector<double3>& velocities,
		thrust::device_vector<double3>& forces,
		double massInv)
	{
		double3* velocities_ptr = thrust::raw_pointer_cast(&velocities[0]);
		double3* forces_ptr = thrust::raw_pointer_cast(&forces[0]);
		advanceVelocitiesKernel<<<numBlocks, blockSize>>>(velocities_ptr, forces_ptr, massInv);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	void advancePositions(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<double3>& velocities)
	{
		double3* positions_ptr = thrust::raw_pointer_cast(&positions[0]);
		double3* velocities_ptr = thrust::raw_pointer_cast(&velocities[0]);
		advancePositionsKernel<<<numBlocks, blockSize>>>(positions_ptr, velocities_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	void addBodyForces(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<double3>& velocities,
		thrust::device_vector<double3>& forces,
		double mass)
	{
		double3* positions_ptr = thrust::raw_pointer_cast(&positions[0]);
		double3* velocities_ptr = thrust::raw_pointer_cast(&velocities[0]);
		double3* forces_ptr = thrust::raw_pointer_cast(&forces[0]);
		addBodyForcesKernel<<<numBlocks, blockSize>>>(positions_ptr, velocities_ptr, forces_ptr, mass);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	void addSpringForces(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<double3>& forces,
		thrust::device_vector<int>& adjs,
		thrust::device_vector<unsigned int>& adjsCounts,
		thrust::device_vector<unsigned int>& adjsStarts,
		thrust::device_vector<double>& restLengths,
		thrust::device_vector<double>& taus) 
	{
		double3* positions_ptr = thrust::raw_pointer_cast(&positions[0]);
		double3* forces_ptr = thrust::raw_pointer_cast(&forces[0]);
		int* adjs_ptr = thrust::raw_pointer_cast(&adjs[0]);
		unsigned int* adjsCounts_ptr = thrust::raw_pointer_cast(&adjsCounts[0]);
		unsigned int* adjsStarts_ptr = thrust::raw_pointer_cast(&adjsStarts[0]);
		double* restLengths_ptr = thrust::raw_pointer_cast(&restLengths[0]);
		double* taus_ptr = thrust::raw_pointer_cast(&taus[0]);
		addSpringForcesKernel<<<numBlocks, blockSize>>>(
			positions_ptr, forces_ptr, adjs_ptr, adjsCounts_ptr, adjsStarts_ptr, restLengths_ptr, taus_ptr);
#if ERRCHECK_AND_SYNC
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// End of kernel unit-testing helpers ///////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


}
#pragma once

#include <vector_types.h>
#include <vector_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#define ERRCHECK_AND_SYNC 0

namespace CudaSupport
{	
/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Simulation interface /////////////////////////////////////////
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
		thrust::device_vector<unsigned int>& hashCounts);



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
		double particleMassInv);



	template<typename T>
	void thrustCopyHostToDev(
		thrust::device_vector<T>& deviceVector,
		thrust::host_vector<T>& hostVector
	)
	{
		deviceVector = hostVector;
	}

	void resetVelocitiesAndForces(
		thrust::device_vector<double3>& velocities,
		thrust::device_vector<double3>& forces
	);

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Helpers for unit testing /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

	void* allocateDeviceMemory(unsigned int size);



	void freeDeviceMemory(void* free);



	void copyToDevice(void* devPtr, void* dataPtr, unsigned int size);



	void copyFromDevice(void* dataPtr, void* devPtr, unsigned int size);


	void initialize(
		thrust::device_vector<unsigned int>& hashCounts);


	void collect(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<unsigned int>& hashTable,
		thrust::device_vector<unsigned int>& hashCounts);


	void detectCollisions(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<double3>& forces,
		thrust::device_vector<unsigned int>& hashTable,
		thrust::device_vector<unsigned int>& hashCounts);


	void advanceVelocities(
		thrust::device_vector<double3>& velocities,
		thrust::device_vector<double3>& forces,
		double massInv);


	void advancePositions(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<double3>& velocities);


	void addBodyForces(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<double3>& velocities,
		thrust::device_vector<double3>& forces,
		double mass);


	void addSpringForces(
		thrust::device_vector<double3>& positions,
		thrust::device_vector<double3>& forces,
		thrust::device_vector<int>& adjs,
		thrust::device_vector<unsigned int>& adjsCounts,
		thrust::device_vector<unsigned int>& adjsStarts,
		thrust::device_vector<double>& restLengths,
		thrust::device_vector<double>& taus);

}
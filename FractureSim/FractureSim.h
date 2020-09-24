#pragma once

#include "Simulation.h"

#include "parameters.h"

#include "CudaSupport.h"

#include "helpers.h"

#define DEBUG 0

#include <fstream>

//#define DEBUG_MSS_GENERATION

class FractureSim : public Simulation {
public:

	enum class ModelingGeometry {
		CUBES,
		TETRAHEDRA,
		VORONOI_POLYTOPES
	};

	FractureSim() : Simulation() {
		init();
	}

	~FractureSim() {
		recWriter.close();
	}

	virtual void init() override;

	// Initialize device-side data structure to run the simulation
	void initializeSimOnGPU();

	void initializeRecFile();

	void writeFrameRecord();

	void readMeshFile(const std::string&);

	// Reset members of Simulation
	virtual void resetMembers() override;

	virtual void updateRenderGeometry() override;

	virtual bool advance() override;

	virtual void renderRenderGeometry(
		igl::opengl::glfw::Viewer& viewer) override;

	void setMassOfGlass(double m) 
	{ 
		m_massOfOriginalGlass = m; 
	}

private:

	double m_massOfOriginalGlass;
	unsigned int m_iterationsPerFrame;
	double m_internalTimeStep;

	double3 m_gravity;
	double3 m_gravityForce;

	/// Parameters for string placement
	double m_lambda;							// Length scale
	double m_delta;								// Neighborhood radius
	double m_stiffness;
	double m_avgtau;

	/// Material parameters
	double K; 									// Bulk modulus
	double G; 									// Fracture energy of the materials

	// Particles
	thrust::host_vector<double3> m_originalPositions;
	thrust::host_vector<double3> m_positions;
	thrust::device_vector<double3> m_positionsDev;
	thrust::device_vector<double3> m_velocities;
	thrust::device_vector<double3> m_forces;
	thrust::device_vector<double> m_epsilon_min;
	thrust::device_vector<double> m_taus;
	
	// Springs
	thrust::host_vector<int> m_originalAdjs;
	thrust::host_vector<int> m_adjs;
	thrust::device_vector<int> m_adjsDev;
	thrust::host_vector<unsigned> m_adjsCounts;
	thrust::device_vector<unsigned int> m_adjsCountsDev;
	thrust::host_vector<unsigned int> m_adjsStarts;
	thrust::device_vector<unsigned int> m_adjsStartsDev;
	thrust::host_vector<double> m_restLengths;
	thrust::device_vector<double> m_restLengthsDev;

	// Collision detection
	thrust::device_vector<unsigned int> m_hashTable;
	thrust::device_vector<unsigned int> m_hashBinsCounts;


	// Always collision detection
	double m_collisionThreshold;
	double m_Kc;
	unsigned int m_hashBinsSize;
	unsigned int m_numHashBins;
	unsigned int m_hashTableSize;
	double m_gridCellSize;

	double m_particleMass;
	double m_particleMassInv;
	size_t m_numParticles;

	std::vector<RigidObject> m_crashedObjects;	// Objects that have to be removed (cracked into smaller ones)
	std::vector<RigidObject> m_originalObjects;	// Original objects

	std::vector<Eigen::MatrixXd> m_renderVs;	// vertex positions for rendering
	std::vector<Eigen::MatrixXi> m_renderFs;	// face indices for rendering

	unsigned int frameNo{ 0 };
	bool m_dumpToOff{ false };

	std::ofstream recWriter;
	int maxBonds;
};
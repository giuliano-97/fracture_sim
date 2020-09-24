#include "FractureSim.h" 
#include "KDTreeVectorOfVectorsAdaptor.h"

#include <random>
#include <sstream>
#include <fstream>

typedef std::vector< Eigen::Vector3d > vectorOfPositions;
typedef KDTreeVectorOfVectorsAdaptor< vectorOfPositions, double >  KDTreeVectorOfPositions;

const std::string cubeMeshPath = "cube.off";

using namespace d3_helpers;

static const Eigen::RowVector3d floorColor = { 0.5, 0.5, 0.5 };
static const double3 x_offset = make_double3(g_lambda / 2, 0, 0);
static const double3 y_offset = make_double3(0, g_lambda / 2, 0);
static const double3 z_offset = make_double3(0, 0, g_lambda / 2);

void FractureSim::init()
{
	m_gravity = make_double3(0, -9.81, 0);
	m_massOfOriginalGlass = 1000000.0; // mass of entire glass
	m_iterationsPerFrame = 1000;
	m_internalTimeStep = g_timestep / m_iterationsPerFrame;

	m_dumpToOff = false;

	m_originalObjects.clear();

	std::string path = "../../data/tet_vase.mesh";
	readMeshFile(path);

	// Mass springs system parameters
	K = g_K;
	G = g_G;

	// Compute spring constants
	m_delta = g_N * m_lambda;

	m_stiffness = 18 * K / (M_PI * pow(m_delta, 4));
	m_avgtau = sqrt(5 * G / (K * m_delta));
	m_Kc = g_Kc;
	m_collisionThreshold = m_lambda / 2;

	// Collision detection parameters
	m_gridCellSize = m_lambda * 2;						// Larger than min interparticle distance and aabb size
	m_hashBinsSize = 128;								// Max size of hash table cell
	m_numHashBins = 10086;								// Number of hash bins
	m_hashTableSize = m_numHashBins * m_hashBinsSize;

	// Particles properties
	m_numParticles = m_originalPositions.size();
	m_particleMass = 10;
	m_particleMassInv = 1 / m_particleMass;

	// Initialize the rest of the simulation params
	initializeSimOnGPU();

	// Initialize rendering geometry
	m_objects = m_originalObjects;

	// Initialize .rec record file
	initializeRecFile();
}


void FractureSim::initializeSimOnGPU()
{
	// Particles state data structures
	m_positions = m_originalPositions;		// We will use this two as a proxy to get position data and reset
	CudaSupport::thrustCopyHostToDev(m_positionsDev, m_positions);

	// Adjacencies to compute spring forces
	m_adjs = m_originalAdjs;
	CudaSupport::thrustCopyHostToDev(m_adjsDev, m_adjs);	// We will use this to get connectivity data
	CudaSupport::thrustCopyHostToDev(m_adjsCountsDev, m_adjsCounts);
	CudaSupport::thrustCopyHostToDev(m_adjsStartsDev, m_adjsStarts);
	CudaSupport::thrustCopyHostToDev(m_restLengthsDev, m_restLengths);

	//// Initialize all the rest on the CUDA side
	CudaSupport::initializeSimulationParameters(
		m_numParticles,
		m_numHashBins,
		m_hashBinsSize,
		m_gridCellSize,
		m_internalTimeStep,
		m_Kc,
		m_stiffness,
		m_avgtau,
		m_collisionThreshold,
		m_lambda,
		m_velocities,
		m_forces,
		m_taus,
		m_hashTable,
		m_hashBinsCounts
	);

}


void FractureSim::resetMembers()
{
	m_crashedObjects = m_objects;
	m_objects = m_originalObjects;
	CudaSupport::thrustCopyHostToDev(m_positionsDev, m_originalPositions);
	CudaSupport::thrustCopyHostToDev(m_adjsDev, m_originalAdjs);
	CudaSupport::resetVelocitiesAndForces(m_velocities, m_forces);
}


void FractureSim::updateRenderGeometry() 
{
	for (size_t i = 0; i < m_objects.size(); i++) {
		RigidObject& o = m_objects[i];
		if (o.getID() < 0) {
			m_renderVs.emplace_back();
			m_renderFs.emplace_back();
		}

		m_objects[i].getMesh(m_renderVs[i], m_renderFs[i]);
	}
}

bool FractureSim::advance() {

	CudaSupport::iterate(
		m_positions,
		m_positionsDev,
		m_velocities,
		m_forces,
		m_adjs,
		m_adjsDev,
		m_adjsCountsDev,
		m_adjsStartsDev,
		m_restLengthsDev,
		m_taus,
		m_hashTable,
		m_hashBinsCounts,
		m_iterationsPerFrame,
		m_particleMass,
		m_particleMassInv);


	// Update geometry primitives
	for (int i = 0; i < m_objects.size(); ++i) {
		m_objects[i].setPosition(m_positions[i].x, m_positions[i].y, m_positions[i].z);
	}


	// advance time
	m_time += m_dt;
	m_step++;

	if (m_step % 10 == 0) {
		writeFrameRecord();
	}
	return false;
}


void FractureSim::renderRenderGeometry(igl::opengl::glfw::Viewer& viewer)
{
	// Step 1: erase crashed meshes in viewer(on GPU?), they are no longer existing in the scene
	for (size_t i = 0; i < m_crashedObjects.size(); i++) {
		RigidObject& co = m_crashedObjects[i];
		if (co.getID() < 0) {
			// the object doesn't exist in renderer
		}
		else {
			size_t meshIndex = viewer.mesh_index(co.getID());
			viewer.erase_mesh(meshIndex);
		}
	}
	m_crashedObjects.clear();

	// Step 2: update geometries to render
	for (size_t i = 0; i < m_objects.size(); i++) {
		RigidObject& o = m_objects[i];
		if (o.getID() < 0) {
			int new_id = 0;
			if (i > 0) {
				new_id = viewer.append_mesh();
				o.setID(new_id);
			}
			else {
				o.setID(new_id);
			}

			// show faces instead of edges of meshes
			size_t meshIndex = viewer.mesh_index(o.getID());
			viewer.data_list[meshIndex].show_lines = false;
			viewer.data_list[meshIndex].show_faces = true;

			viewer.data_list[meshIndex].set_face_based(true);
			viewer.data_list[meshIndex].point_size = 2.0f;
			viewer.data_list[meshIndex].clear();
		}
		size_t meshIndex = viewer.mesh_index(o.getID());

		viewer.data_list[meshIndex].set_mesh(m_renderVs[i], m_renderFs[i]);
		viewer.data_list[meshIndex].compute_normals();

		Eigen::MatrixXd color;
		o.getColors(color);
		viewer.data_list[meshIndex].set_colors(color);
	}

}

void FractureSim::readMeshFile(const std::string& path) {
	std::ifstream file(path.c_str());
	if (!file.is_open()) {
		throw(std::runtime_error("FractureSim::readMeshFile(): invalid file path"));
	}

	int num, numtetra, useless;
	file >> num >> numtetra >> useless;
	float x, y, z;

	vectorOfPositions initialPositions;
	int particleLocalId = 0;
	int particleIdOffset = m_originalPositions.size();
	int renderIndexOffset = m_originalObjects.size();

	for (int i = 0; i < num; i++) {
		file >> x >> y >> z;

		double nx = x * (sqrt(3) * 0.5) - y * 0.5;
		double ny = x * 0.5 + y * (sqrt(3) * 0.5);
		x = nx;
		y = ny;

		y += 0.022;

		int renderIndex = renderIndexOffset + particleLocalId;

		Eigen::Vector3d position{ x,y,z };
		m_originalPositions.push_back(make_double3(x, y, z));
		initialPositions.push_back(position);

		//have something to show on the screen
		if (i < 10) {
			m_originalObjects.push_back(RigidObject(cubeMeshPath));
			m_originalObjects[renderIndex].setScale(0.05);
			m_originalObjects[renderIndex].setPosition(position);
			particleLocalId++;
		}
	}

	int i0, i1, i2, i3, t;
	double3 tmpV[4];
	m_lambda = 1;
	for (int i = 0; i < numtetra; i++) {
		file >> i0 >> i1 >> i2 >> i3 >> t >> t >> t >> t;
		tmpV[0] = m_originalPositions[i0];
		tmpV[1] = m_originalPositions[i1];
		tmpV[2] = m_originalPositions[i2];
		tmpV[3] = m_originalPositions[i3];
		
		for (int j = 0; j < 3; j++) {
			for (int k = j + 1; k < 4; k++) {
				double dist = norm(tmpV[j] - tmpV[k]);
				if (dist < m_lambda)
					m_lambda = dist;
			}
		}
	}
	std::cout << "M_LAMBDA:" << m_lambda << std::endl;
	m_lambda *= 0.7;
	file.close();

	m_delta = g_N * m_lambda;

	KDTreeVectorOfPositions kdtree_index(
		3					/* Dimension of sample space*/,
		initialPositions,	/* points */
		10					/* max leaf */);

	// For each particle, make a radius search with radius = delta
	// NOTE 1: nanoflann radius search uses squared distances
	// NOTE 2: we have to add a small epsilon to delta^2 make a border-inclusive radius search
	const double search_radius = pow(m_delta, 2) + 1e-04;

	// Data structures for nanoflann radius search
	std::vector<std::pair<size_t, double> > ret_matches;
	std::vector<std::vector<unsigned int>> neighbors(initialPositions.size());
	nanoflann::SearchParams params;

	// Place springs
	for (size_t i = 0; i < initialPositions.size(); ++i) {
		// Reset search params
		ret_matches.clear();
		Eigen::Vector3d query_pt = initialPositions[i];
		int particleGlobalIndex = particleIdOffset + i;

#ifdef DEBUG_MSS_GENERATION
		std::cout << "Placing the springs for particle ("
			<< d3_helpers::double3_toString(m_originalPositions[particleGlobalIndex]) << "):\n";
#endif

		// Do radius search - the result will be a vector of pairs of the type
		// <index of neighbor in initialPositions, L^2 distance of neighbor>
		const size_t nMatches = kdtree_index.index->radiusSearch(&query_pt[0],
			search_radius,
			ret_matches,
			params);

		// Place springs between query_pt and all points found by radius search
		for (int j = 0; j < ret_matches.size(); ++j) {
			size_t localNeighborIndex = ret_matches[j].first;
			size_t globalNeighborIndex = particleIdOffset + localNeighborIndex;

			// Check if a spring has already been placed
			if (localNeighborIndex != i) {
				// Add neighbor
				neighbors[i].push_back(globalNeighborIndex);
			}
		}
	}

	// Now that we have determined the neighbors of each particle, we generate the 1D adj. list
	for (int i = 0; i < neighbors.size(); ++i) {
		unsigned int globalParticleIndex = i + particleIdOffset;
		// Add adjancecies
		for (unsigned int idx : neighbors[i]) {
			m_originalAdjs.push_back(idx);
			double dist =
				d3_helpers::norm(m_originalPositions[i] - m_originalPositions[idx]);
			m_restLengths.push_back(dist);
		}

		// Add adjancencies count and starting point
		if (globalParticleIndex != 0) {
			m_adjsStarts.push_back(m_adjsStarts.back() + m_adjsCounts.back());
		}
		else {
			m_adjsStarts.push_back(0);
		}

		unsigned int neighbors_cnt = neighbors[i].size();
		m_adjsCounts.push_back(neighbors_cnt);
	}
}

void FractureSim::initializeRecFile() {
	// Initialize record file output stream
	recWriter = std::ofstream("../../data/temp.rec", std::ios::binary | std::ios::out);
	if (!recWriter.is_open()) {
		// ERROR: recWriter is not opened
		throw(std::runtime_error("FractureSim::Init(): rec output file open failed"));
	}

	int n_particles = m_positions.size();
	float lambda = m_lambda;

	maxBonds = 0;
	for (size_t i = 0; i < m_adjsCounts.size(); i++) {
		if (maxBonds < m_adjsCounts[i]) {
			maxBonds = m_adjsCounts[i];
		}
	}

	int object_size = 12 + 4 + 12 + 12 + 4 * maxBonds;

	recWriter << "# object_size = " << object_size
		<< ", n_particles = " << n_particles
		<< ", lambda = " << lambda
		<< ", maxbonds = " << maxBonds << std::endl;

	writeFrameRecord();
}

void FractureSim::writeFrameRecord() {
	// 3 * 4 * n_particles : position
	for (size_t i = 0; i < m_positions.size(); i++) {
		float x = (float)(m_positions[i].x);
		float y = (float)(m_positions[i].y);
		float z = (float)(m_positions[i].z);
		recWriter.write((char*)&x, sizeof(float));
		recWriter.write((char*)&y, sizeof(float));
		recWriter.write((char*)&z, sizeof(float));
	}

	// 4 * n_particles : #neighbors
	for (size_t i = 0; i < m_adjsCounts.size(); i++) {
		int x = (int)m_adjsCounts[i];
		recWriter.write((char*)&x, sizeof(int));
	}

	// 3 * 4 * n_particles : velocities
	// 3 * 4 * n_particles : forces
	for (size_t i = 0; i < m_positions.size(); i++) {
		float x = 0.0f;
		recWriter.write((char*)&x, sizeof(float));
		recWriter.write((char*)&x, sizeof(float));
		recWriter.write((char*)&x, sizeof(float));
		recWriter.write((char*)&x, sizeof(float));
		recWriter.write((char*)&x, sizeof(float));
		recWriter.write((char*)&x, sizeof(float));
	}

	// 4 * maxbonds * n_particles : connectivity
	for (size_t i = 0; i < m_positions.size(); i++) {
		int numOriginalNeighbor = m_adjsCounts[i];
		int offset = m_adjsStarts[i];
		int countCurrentNeighbors = 0;
		for (int j = offset; j < offset + numOriginalNeighbor; j++) {
			int neighborIndex = m_adjs[j];
			if (neighborIndex == -1) {
				// broken spring;
				continue;
			}
			recWriter.write((char*)&neighborIndex, sizeof(int));
			countCurrentNeighbors++;
		}
		// fill rest of row with zeros
		for (int j = countCurrentNeighbors; j < maxBonds; j++) {
			int x = 0;
			recWriter.write((char*)&x, sizeof(int));
		}
	}

}




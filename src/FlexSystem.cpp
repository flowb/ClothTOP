#include "FlexSystem.h"

void ErrorCallback(NvFlexErrorSeverity, const char* msg, const char* file, int line)
{
	printf("Flex: %s - %s:%d\n", msg, file, line);
}

void VMesh::GetBounds(Vector3& outMinExtents, Vector3& outMaxExtents) const
{
	Point3 minExtents(FLT_MAX);
	Point3 maxExtents(-FLT_MAX);

	// calculate face bounds
	for (uint32_t i = 0; i < m_positions.size(); ++i)
	{
		const Point3& a = Point3(m_positions[i].x, m_positions[i].y, m_positions[i].z);

		minExtents = Min(a, minExtents);
		maxExtents = Max(a, maxExtents);
	}

	outMinExtents = Vector3(minExtents);
	outMaxExtents = Vector3(maxExtents);
}

SimBuffers::SimBuffers(NvFlexLibrary* l) :
	positions(l), positionsGpu(l, 0, eNvFlexBufferDevice), restPositions(l), velocities(l), phases(l), densities(l),
	anisotropy1(l, 0, eNvFlexBufferDevice), anisotropy2(l, 0, eNvFlexBufferDevice), anisotropy3(l, 0, eNvFlexBufferDevice), normals(l), normalsGpu(l, 0, eNvFlexBufferDevice), smoothPositions(l, 0, eNvFlexBufferDevice),
	diffusePositions(l), diffuseVelocities(l), diffuseCount(l), activeIndices(l),
	shapeGeometry(l), shapePositions(l), shapeRotations(l), shapePrevPositions(l),
	shapePrevRotations(l), shapeFlags(l), rigidOffsets(l), rigidIndices(l), rigidMeshSize(l),
	rigidCoefficients(l), rigidPlasticThresholds(l), rigidPlasticCreeps(l), rigidRotations(l), rigidTranslations(l),
	rigidLocalPositions(l), rigidLocalNormals(l), inflatableTriOffsets(l),
	inflatableTriCounts(l), inflatableVolumes(l), inflatableCoefficients(l),
	inflatablePressures(l), springIndices(l), springLengths(l),
	springStiffness(l), triangles(l), triangleNormals(l), uvs(l), uvsGpu(l,0, eNvFlexBufferDevice),
	contactPlanes(l), contactVelocities(l), contactIndices(l), contactCounts(l), trianglesGpu(l, 0, eNvFlexBufferDevice), triangleNormalsGpu(l, 0, eNvFlexBufferDevice)/*, fieldCollider(l, 0, eNvFlexBufferDevice)*/
{

}

SimBuffers::~SimBuffers() 
{
	// particles
	positions.destroy();
	positionsGpu.destroy();
	restPositions.destroy();
	velocities.destroy();
	phases.destroy();
	densities.destroy();
	anisotropy1.destroy();
	anisotropy2.destroy();
	anisotropy3.destroy();
	normals.destroy();
	normalsGpu.destroy();
	diffusePositions.destroy();
	diffuseVelocities.destroy();
	diffuseCount.destroy();
	smoothPositions.destroy();
	activeIndices.destroy();

	// convexes
	shapeGeometry.destroy();
	shapePositions.destroy();
	shapeRotations.destroy();
	shapePrevPositions.destroy();
	shapePrevRotations.destroy();
	shapeFlags.destroy();

	// rigids
	rigidOffsets.destroy();
	rigidIndices.destroy();
	rigidMeshSize.destroy();
	rigidCoefficients.destroy();
	rigidPlasticThresholds.destroy();
	rigidPlasticCreeps.destroy();
	rigidRotations.destroy();
	rigidTranslations.destroy();
	rigidLocalPositions.destroy();
	rigidLocalNormals.destroy();

	// springs
	springIndices.destroy();
	springLengths.destroy();
	springStiffness.destroy();

	// inflatables
	inflatableTriOffsets.destroy();
	inflatableTriCounts.destroy();
	inflatableVolumes.destroy();
	inflatableCoefficients.destroy();
	inflatablePressures.destroy();

	// triangles
	triangles.destroy();
	trianglesGpu.destroy();
	triangleNormals.destroy();
	triangleNormalsGpu.destroy();
	uvs.destroy();
	uvsGpu.destroy();

}

void SimBuffers::MapBuffers() 
{
	// particles
	positions.map();
	positionsGpu.map();
	restPositions.map();
	velocities.map();
	phases.map();
	densities.map();
	anisotropy1.map();
	anisotropy2.map();
	anisotropy3.map();
	normals.map();
	normalsGpu.map();
	diffusePositions.map();
	diffuseVelocities.map();
	diffuseCount.map();
	smoothPositions.map();
	activeIndices.map();

	// convexes
	shapeGeometry.map();
	shapePositions.map();
	shapeRotations.map();
	shapePrevPositions.map();
	shapePrevRotations.map();
	shapeFlags.map();

    // rigids
	rigidOffsets.map();
	rigidIndices.map();
	rigidMeshSize.map();
	rigidCoefficients.map();
	rigidPlasticThresholds.map();
	rigidPlasticCreeps.map();
	rigidRotations.map();
	rigidTranslations.map();
	rigidLocalPositions.map();
	rigidLocalNormals.map();

	springIndices.map();
	springLengths.map();
	springStiffness.map();

	// inflatables
	inflatableTriOffsets.map();
	inflatableTriCounts.map();
	inflatableVolumes.map();
	inflatableCoefficients.map();
	inflatablePressures.map();

	// triangles
	triangles.map();
	trianglesGpu.map();
	triangleNormals.map();
	triangleNormalsGpu.map();
	uvs.map();
	uvsGpu.map();

	//contacts
	contactPlanes.map();
	contactVelocities.map();
	contactIndices.map();
	contactCounts.map();

}

void SimBuffers::UnmapBuffers() 
{
	// particles
	positions.unmap();
	positionsGpu.unmap();
	restPositions.unmap();
	velocities.unmap();
	phases.unmap();
	densities.unmap();
	anisotropy1.unmap();
	anisotropy2.unmap();
	anisotropy3.unmap();
	normals.unmap();
	normalsGpu.unmap();
	diffusePositions.unmap();
	diffuseVelocities.unmap();
	diffuseCount.unmap();
	smoothPositions.unmap();
	activeIndices.unmap();

	// convexes
	shapeGeometry.unmap();
	shapePositions.unmap();
	shapeRotations.unmap();
	shapePrevPositions.unmap();
	shapePrevRotations.unmap();
	shapeFlags.unmap();

	// rigids
	rigidOffsets.unmap();
	rigidIndices.unmap();
	rigidMeshSize.unmap();
	rigidCoefficients.unmap();
	rigidPlasticThresholds.unmap();
	rigidPlasticCreeps.unmap();
	rigidRotations.unmap();
	rigidTranslations.unmap();
	rigidLocalPositions.unmap();
	rigidLocalNormals.unmap();

	// springs
	springIndices.unmap();
	springLengths.unmap();
	springStiffness.unmap();

	// inflatables
	inflatableTriOffsets.unmap();
	inflatableTriCounts.unmap();
	inflatableVolumes.unmap();
	inflatableCoefficients.unmap();
	inflatablePressures.unmap();

	// triangles
	triangles.unmap();
	trianglesGpu.unmap();
	triangleNormals.unmap();
	triangleNormalsGpu.unmap();
	uvs.unmap();
	uvsGpu.unmap();

	//contacts
	contactPlanes.unmap();
	contactVelocities.unmap();
	contactIndices.unmap();
	contactCounts.unmap();

}

void SimBuffers::InitBuffers() 
{
	// particles
	positions.resize(0);
	positionsGpu.resize(0);
	restPositions.resize(0);
	velocities.resize(0);
	phases.resize(0);
	diffusePositions.resize(0);
	diffuseVelocities.resize(0);
	diffuseCount.resize(0);

	normals.resize(0);
	normalsGpu.resize(0);

	// rigids
	rigidOffsets.resize(0);
	rigidIndices.resize(0);
	rigidMeshSize.resize(0);
	rigidRotations.resize(0);
	rigidTranslations.resize(0);
	rigidCoefficients.resize(0);
	rigidPlasticThresholds.resize(0);
	rigidPlasticCreeps.resize(0);
	rigidLocalPositions.resize(0);
	rigidLocalNormals.resize(0);
	
	// springs
	springIndices.resize(0);
	springLengths.resize(0);
	springStiffness.resize(0);
	
	// triangles
	triangles.resize(0);
	triangleNormals.resize(0);
	uvs.resize(0);
	uvsGpu.resize(0);

    // shapes
	shapeGeometry.resize(0);
	shapePositions.resize(0);
	shapeRotations.resize(0);
	shapePrevPositions.resize(0);
	shapePrevRotations.resize(0);
	shapeFlags.resize(0);

	//contacts
	contactPlanes.resize(0);
	contactVelocities.resize(0);
	contactIndices.resize(0);
	contactCounts.resize(0);
}

FlexSystem::FlexSystem()
{	
	g_profile = false;
	
	nEmitter = 0;
	nVolumeBoxes = 0;

	maxParticles = 9;
	g_maxDiffuseParticles = 60;
	numDiffuse = 64;

	g_solver = NULL;

	time1 = 0;
	time2 = 0;
	time3 = 0;
	time4 = 0;

	memset(&g_timers, 0, sizeof(g_timers));

	cursor = 0;

	nFields = 0;
	clothMesh0 = nullptr;
	g_triangleCollisionMesh = nullptr;
}

FlexSystem::~FlexSystem()
{
	if (g_solver)
	{
		if(g_buffers)
			delete g_buffers;

		if (clothMesh0)
			delete clothMesh0;

		if (g_triangleCollisionMesh)
			delete g_triangleCollisionMesh;
		
		NvFlexDestroySolver(g_solver);
		NvFlexShutdown(g_flexLib);
	}

	NvFlexDeviceDestroyCudaContext();
}

void FlexSystem::getSimTimers() 
{
	if (g_profile) {
		memset(&g_timers, 0, sizeof(g_timers));
		NvFlexGetTimers(g_solver, &g_timers);
		simLatency = g_timers.total;
	}
	else
		simLatency = NvFlexGetDeviceLatency(g_solver, &g_GpuTimers.computeBegin, &g_GpuTimers.computeEnd, &g_GpuTimers.computeFreq);
}

void FlexSystem::initSystem() 
{
	//int g_device = -1;
	g_device = NvFlexDeviceGetSuggestedOrdinal();

	// Create an optimized CUDA context for Flex and set it on the 
	// calling thread. This is an optional call, it is fine to use 
	// a regular CUDA context, although creating one through this API
	// is recommended for best performance.
	bool success = NvFlexDeviceCreateCudaContext(g_device);

	if (!success)
	{
		printf("Error creating CUDA context.\n");
	}

	NvFlexInitDesc desc;
	desc.deviceIndex = g_device;
	desc.enableExtensions = true;
	desc.renderDevice = 0;
	desc.renderContext = 0;
	desc.computeType = eNvFlexCUDA;

	g_flexLib = NvFlexInit(NV_FLEX_VERSION, ErrorCallback, &desc);
}

void FlexSystem::initParams() 
{
	g_params.gravity[0] = 0.0f;
	g_params.gravity[1] = -9.8f;
	g_params.gravity[2] = 0.0f;

	g_params.wind[0] = 0.0f;
	g_params.wind[1] = 0.0f;
	g_params.wind[2] = 0.0f;
	g_params.drag = 0.0f;
	g_params.lift = 0.0f;

	g_params.radius = 0.15f;
	g_params.viscosity = 0.0f;
	g_params.dynamicFriction = 0.0f;
	g_params.staticFriction = 0.0f;
	g_params.particleFriction = 0.0f; // scale friction between particles by default
	g_params.freeSurfaceDrag = 0.0f;
	g_params.drag = 0.0f;
	g_params.lift = 0.0f;
	g_params.numIterations = 3;
	g_params.fluidRestDistance = 0.0f;
	g_params.solidRestDistance = 0.0f;

	g_params.anisotropyScale = 1.0f;
	g_params.anisotropyMin = 0.1f;
	g_params.anisotropyMax = 2.0f;
	g_params.smoothing = 1.0f;

	g_params.dissipation = 0.0f;
	g_params.damping = 0.0f;
	g_params.particleCollisionMargin = 0.0f;
	g_params.shapeCollisionMargin = 0.0f;
	g_params.collisionDistance = 0.0f;
	g_params.sleepThreshold = 0.0f;
	g_params.shockPropagation = 0.0f;
	g_params.restitution = 0.001f;

	g_params.maxSpeed = FLT_MAX;
	g_params.maxAcceleration = 100.0f;	// approximately 10x gravity

	g_params.relaxationMode = eNvFlexRelaxationLocal;
	g_params.relaxationFactor = 1.0f;
	g_params.solidPressure = 1.0f;
	g_params.adhesion = 0.0f;
	g_params.cohesion = 0.1f;
	g_params.surfaceTension = 0.0f;
	g_params.vorticityConfinement = 80.0f;
	g_params.buoyancy = 1.0f;
	g_params.diffuseThreshold = 100.0f;
	g_params.diffuseBuoyancy = 1.0f;
	g_params.diffuseDrag = 0.8f;
	g_params.diffuseBallistic = 16;
	g_params.diffuseLifetime = 2.0f;

	g_params.numPlanes = 0;

	callback = NULL;
}

void FlexSystem::initScene() 
{
	RandInit();

	cursor = 0;

	if (g_solver)
	{

		if (g_buffers)
			delete g_buffers;

		NvFlexDestroySolver(g_solver);
		g_solver = nullptr;

		/*	if (g_triangleCollisionMesh) {
				delete g_triangleCollisionMesh;
				g_triangleCollisionMesh = nullptr;
			}

			if (clothMesh0) {
				delete clothMesh0;
				clothMesh0 = nullptr;
			}*/
	}

	// alloc buffers
	g_buffers = new SimBuffers(g_flexLib);

	// map during initialization
	g_buffers->MapBuffers();
	g_buffers->InitBuffers();

	g_rectEmitters.resize(0);
	g_volumeBoxes.resize(0);

	initParams();

	g_numSubsteps = 2;

	g_sceneLower = FLT_MAX;
	g_sceneUpper = -FLT_MAX;

	// initialize solver desc
	NvFlexSetSolverDescDefaults(&g_solverDesc);

	ClearShapes();

	maxParticles = 64;
	g_maxDiffuseParticles = 0;
	g_maxNeighborsPerParticle = 96;
	g_maxContactsPerParticle = 6;

}

void FlexSystem::postInitScene()
{
	g_solverDesc.featureMode = eNvFlexFeatureModeSimpleSolids;// eNvFlexFeatureModeSimpleFluids;

	g_params.fluidRestDistance = g_params.radius*0.65f;

	if (g_params.solidRestDistance == 0.0f)
		g_params.solidRestDistance = g_params.radius;

	// if fluid present then we assume solid particles have the same radius
	if (g_params.fluidRestDistance > 0.0f)
		g_params.solidRestDistance = g_params.fluidRestDistance;

	// set collision distance automatically based on rest distance if not alraedy set
	if (g_params.collisionDistance == 0.0f)
		g_params.collisionDistance = Max(g_params.solidRestDistance, g_params.fluidRestDistance)*0.5f;

	// default particle friction to 10% of shape friction
	if (g_params.particleFriction == 0.0f)
		g_params.particleFriction = g_params.dynamicFriction*0.1f;

	// add a margin for detecting contacts between particles and shapes
	if (g_params.shapeCollisionMargin == 0.0f)
		g_params.shapeCollisionMargin = g_params.collisionDistance*0.5f;

	g_maxDiffuseParticles = 5000;


	for (int i = 0; i < nVolumeBoxes; i++) 
	{
		CreateCenteredParticleGrid(Point3(g_volumeBoxes[i].mPos.x, g_volumeBoxes[i].mPos.y, g_volumeBoxes[i].mPos.z), g_volumeBoxes[i].mRot, Point3(g_volumeBoxes[i].mSize.x, g_volumeBoxes[i].mSize.y, g_volumeBoxes[i].mSize.z), g_params.fluidRestDistance, Vec3(0.0f), 1, false, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid), g_params.fluidRestDistance*0.01f);
	}

	g_params.anisotropyScale = 1.0f;

	//*******CREATE TRIANGLE MESH

	if (g_triangleCollisionMesh) {
		triangleCollisionMeshId = CreateTriangleMesh(g_triangleCollisionMesh);
	}

	uint32_t numParticles = g_buffers->positions.size(); //non zero if init volume boxes

	//**** IF PARTICLE GRID

	if (g_buffers->positions.size()) {
		g_buffers->activeIndices.resize(numParticles);
		for (size_t i = 0; i < g_buffers->activeIndices.size(); ++i)
			g_buffers->activeIndices[i] = i;
	}
	g_inactiveIndices.resize(maxParticles - numParticles);

	for (size_t i = 0; i < g_inactiveIndices.size(); ++i)
		g_inactiveIndices[i] = i + numParticles;

	
	// for fluid rendering these are the Laplacian smoothed positions
	g_buffers->smoothPositions.resize(maxParticles);

	g_buffers->normals.resize(0);
	g_buffers->normals.resize(maxParticles);
	g_buffers->normalsGpu.resize(0);
	g_buffers->normalsGpu.resize(maxParticles);

	// resize particle buffers to fit
	g_buffers->positions.resize(maxParticles);
	g_buffers->positionsGpu.resize(maxParticles);
	//g_buffers->restPositions.resize(maxParticles);
	g_buffers->velocities.resize(maxParticles);
	g_buffers->phases.resize(maxParticles);
	g_buffers->densities.resize(maxParticles);
	g_buffers->anisotropy1.resize(maxParticles);
	g_buffers->anisotropy2.resize(maxParticles);
	g_buffers->anisotropy3.resize(maxParticles);

	// diffuse
	g_buffers->diffusePositions.resize(g_maxDiffuseParticles);
	g_buffers->diffuseVelocities.resize(g_maxDiffuseParticles);
	g_buffers->diffuseCount.resize(g_maxDiffuseParticles);

	// triangles
	g_buffers->triangles.resize(maxParticles);
	g_buffers->trianglesGpu.resize(maxParticles);
	g_buffers->triangleNormals.resize(maxParticles);
	g_buffers->triangleNormalsGpu.resize(maxParticles);
	g_buffers->uvs.resize(maxParticles);
	g_buffers->uvsGpu.resize(maxParticles);

	// contacts
	const int maxContactsPerParticle = 6;
	g_buffers->contactPlanes.resize(maxParticles * maxContactsPerParticle);
	g_buffers->contactVelocities.resize(maxParticles * maxContactsPerParticle);
	g_buffers->contactIndices.resize(maxParticles);
	g_buffers->contactCounts.resize(maxParticles);

	// save rest positions
	g_buffers->restPositions.resize(g_buffers->positions.size());
	size_t j;
	for ( j= 0; j < g_buffers->positions.size(); ++j)
		g_buffers->restPositions[j] = g_buffers->positions[j];
	
	g_solverDesc.maxParticles = maxParticles;
	g_solverDesc.maxDiffuseParticles = g_maxDiffuseParticles;
	g_solverDesc.maxNeighborsPerParticle = g_maxNeighborsPerParticle;
	g_solverDesc.maxContactsPerParticle = g_maxContactsPerParticle;

	g_solver = NvFlexCreateSolver(g_flexLib, &g_solverDesc);

	// ######################### WARM UP ######################################

	g_buffers->UnmapBuffers();

	if (g_buffers->activeIndices.size()) {
		NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, NULL);
	}

	if (g_buffers->positions.size()) {
		NvFlexSetParticles(g_solver, g_buffers->positions.buffer, NULL);
		NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, NULL);
		NvFlexSetPhases(g_solver, g_buffers->phases.buffer, NULL);

		NvFlexSetNormals(g_solver, g_buffers->normals.buffer, NULL);
	}
	
	// rigids
	if (g_buffers->rigidOffsets.size())
	{
		NvFlexSetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer, g_buffers->rigidLocalPositions.buffer, g_buffers->rigidLocalNormals.buffer, g_buffers->rigidCoefficients.buffer, g_buffers->rigidPlasticThresholds.buffer, g_buffers->rigidPlasticCreeps.buffer, g_buffers->rigidRotations.buffer, g_buffers->rigidTranslations.buffer, g_buffers->rigidOffsets.size() - 1, g_buffers->rigidIndices.size());
	}
	
	// triangles
	if (g_buffers->triangles.size())
		NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangles.size() / 3);

	// springs
	if (g_buffers->springIndices.size())
	{
		assert((g_buffers->springIndices.size() & 1) == 0);
		assert((g_buffers->springIndices.size() / 2) == g_buffers->springLengths.size());
		NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer, g_buffers->springStiffness.buffer, g_buffers->springLengths.size());
	}

	// inflatables
	if (g_buffers->inflatableTriOffsets.size())
	{
		NvFlexSetInflatables(g_solver, g_buffers->inflatableTriOffsets.buffer, g_buffers->inflatableTriCounts.buffer, g_buffers->inflatableVolumes.buffer, g_buffers->inflatablePressures.buffer, g_buffers->inflatableCoefficients.buffer, g_buffers->inflatableTriOffsets.size());
	}

	setShapes();

	NvFlexSetParams(g_solver, &g_params);

	NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

	// ###################################################################################

	// Forcefield
	// free previous callback, todo: destruction phase for tests
	if (callback)
		NvFlexExtDestroyForceFieldCallback(callback);

	// create new callback
	callback = NvFlexExtCreateForceFieldCallback(g_solver);
}

void FlexSystem::AddBox(Vec3 halfEdge, Vec3 center, Quat quat, bool dynamic)
{
	// transform
	g_buffers->shapePositions.push_back(Vec4(center.x, center.y, center.z, 0.0f));
	g_buffers->shapeRotations.push_back(quat);

	g_buffers->shapePrevPositions.push_back(g_buffers->shapePositions.back());
	g_buffers->shapePrevRotations.push_back(g_buffers->shapeRotations.back());

	NvFlexCollisionGeometry geo;
	geo.box.halfExtents[0] = halfEdge.x;
	geo.box.halfExtents[1] = halfEdge.y;
	geo.box.halfExtents[2] = halfEdge.z;

	g_buffers->shapeGeometry.push_back(geo);
	g_buffers->shapeFlags.push_back(NvFlexMakeShapeFlags(eNvFlexShapeBox, dynamic));
}

void FlexSystem::AddSphere(float radius, Vec3 position, Quat rotation)
{
	NvFlexCollisionGeometry geo;
	geo.sphere.radius = radius;
	g_buffers->shapeGeometry.push_back(geo);

	g_buffers->shapePositions.push_back(Vec4(position, 0.0f));
	g_buffers->shapeRotations.push_back(rotation);

	g_buffers->shapePrevPositions.push_back(g_buffers->shapePositions.back());
	g_buffers->shapePrevRotations.push_back(g_buffers->shapeRotations.back());

	int flags = NvFlexMakeShapeFlags(eNvFlexShapeSphere, false);
	g_buffers->shapeFlags.push_back(flags);
}

NvFlexTriangleMeshId FlexSystem::CreateTriangleMesh(VMesh* m)
{
	/*if (!m)
		return 0;*/

	Vec3 lower, upper;

	lower = m->minExtents;
	upper = m->maxExtents;

	NvFlexVector<Vec4> positions(g_flexLib);
	NvFlexVector<int> indices(g_flexLib);

	positions.assign((Vec4*)&m->m_positions[0], m->m_positions.size());
	indices.assign((int*)&m->m_indices[0], m->m_indices.size());

	positions.unmap();
	indices.unmap();

	NvFlexTriangleMeshId flexMesh = NvFlexCreateTriangleMesh(g_flexLib);
	NvFlexUpdateTriangleMesh(g_flexLib, flexMesh, positions.buffer, indices.buffer, m->GetNumVertices(), m->GetNumFaces(), (float*)&lower, (float*)&upper);

	return flexMesh;
}

void FlexSystem::UpdateTriangleMesh(VMesh* m, NvFlexTriangleMeshId flexMeshId)
{
	Vec3 lower, upper;

	lower = m->minExtents;
	upper = m->maxExtents;

	NvFlexVector<Vec4> positions(g_flexLib);
	NvFlexVector<int> indices(g_flexLib);

	positions.assign((Vec4*)&m->m_positions[0], m->m_positions.size());
	indices.assign((int*)&m->m_indices[0], m->m_indices.size());

	positions.unmap();
	indices.unmap();

	NvFlexUpdateTriangleMesh(g_flexLib, flexMeshId, positions.buffer, indices.buffer, m->GetNumVertices(), m->GetNumFaces(), (float*)&lower, (float*)&upper);
}

void FlexSystem::AddTriangleMesh(NvFlexTriangleMeshId mesh, Vec3 translation, Quat rotation, Vec3 prevTrans, Quat prevRot, Vec3 scale)
{
	//Vec3 lower, upper;
	//NvFlexGetTriangleMeshBounds(g_flexLib, mesh, lower, upper);

	NvFlexCollisionGeometry geo;
	geo.triMesh.mesh = mesh;
	geo.triMesh.scale[0] = scale.x;
	geo.triMesh.scale[1] = scale.y;
	geo.triMesh.scale[2] = scale.z;

	g_buffers->shapePositions.push_back(Vec4(translation, 0.0f));
	g_buffers->shapeRotations.push_back(Quat(rotation));
	g_buffers->shapePrevPositions.push_back(Vec4(prevTrans, 0.0f));
	g_buffers->shapePrevRotations.push_back(Quat(prevRot));
	g_buffers->shapeGeometry.push_back((NvFlexCollisionGeometry&)geo);
	g_buffers->shapeFlags.push_back(NvFlexMakeShapeFlags(eNvFlexShapeTriangleMesh, false));
}

void FlexSystem::GetParticleBounds(Vec3& lower, Vec3& upper)
{
	lower = Vec3(FLT_MAX);
	upper = Vec3(-FLT_MAX);

	for (size_t i=0; i < g_buffers->positions.size(); ++i)
	{
		lower = Min(Vec3(g_buffers->positions[i]), lower);
		upper = Max(Vec3(g_buffers->positions[i]), upper);
	}
}

void FlexSystem::CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter=0.005f)
{

	for (int x=0; x < dimx; ++x)
	{
		for (int y=0; y < dimy; ++y)
		{
			for (int z=0; z < dimz; ++z)
			{

				Vec3 position = lower + Vec3(float(x), float(y), float(z))*radius + RandomUnitVector()*jitter;

				g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
				g_buffers->velocities.push_back(velocity);
				g_buffers->phases.push_back(phase);
			}
		}
	}


}

void FlexSystem::CreateCenteredParticleGrid(Point3 center, Vec3 rotation, Point3 size, float restDistance, Vec3 velocity, float invMass, bool rigid, int phase, float jitter)
{

	int dx = int(ceilf(size.x / restDistance));
	int dy = int(ceilf(size.y / restDistance));
	int dz = int(ceilf(size.z / restDistance));

	for (int x=0; x < dx; ++x)
	{
		for (int y=0; y < dy; ++y)
		{
			for (int z=0; z < dz; ++z)
			{
				Point3 position = restDistance*Point3(float(x) - 0.5*(dx-1), float(y) - 0.5*(dy-1), float(z) - 0.5*(dz-1)) + RandomUnitVector()*jitter;
				position = TranslationMatrix(center) * TransformMatrix(Rotation(rotation.y, rotation.z, rotation.x), Point3(0.f))*position;

				if(cursor<maxParticles-1) {

					g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
					g_buffers->velocities.push_back(velocity);
					g_buffers->phases.push_back(phase);

					cursor++;
				}
			}
		}
	}
}

void FlexSystem::CreateSpring(int i, int j, float stiffness, float give = 0.0f)
{
	g_buffers->springIndices.push_back(i);
	g_buffers->springIndices.push_back(j);
	g_buffers->springLengths.push_back((1.0f + give) * Length(Vec3(g_buffers->positions[i]) - Vec3(g_buffers->positions[j])));
	g_buffers->springStiffness.push_back(stiffness);
}

inline int GridIndex(int x, int y, int dx) { return y * dx + x; }

void FlexSystem::CreateSpringGrid(Vec3 lower, int dx, int dy, int dz, float radius, int phase, float stretchStiffness, float bendStiffness, float shearStiffness, Vec3 velocity, float invMass)
{
	int baseIndex = int(g_buffers->positions.size());

	for (int z = 0; z < dz; ++z)
	{
		for (int y = 0; y < dy; ++y)
		{
			for (int x = 0; x < dx; ++x)
			{
				Vec3 position = lower + radius * Vec3(float(x), float(z), float(y));

				g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
				g_buffers->velocities.push_back(velocity);
				g_buffers->phases.push_back(phase);

				if (x > 0 && y > 0)
				{
					g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y - 1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dx));

					g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y, dx));

					g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
					g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
				}
			}
		}
	}

	// horizontal
	for (int y = 0; y < dy; ++y)
	{
		for (int x = 0; x < dx; ++x)
		{
			int index0 = y * dx + x;

			if (x > 0)
			{
				int index1 = y * dx + x - 1;
				CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
			}

			if (x > 1)
			{
				int index2 = y * dx + x - 2;
				CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
			}

			if (y > 0 && x < dx - 1)
			{
				int indexDiag = (y - 1) * dx + x + 1;
				CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
			}

			if (y > 0 && x > 0)
			{
				int indexDiag = (y - 1) * dx + x - 1;
				CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
			}
		}
	}

	// vertical
	for (int x = 0; x < dx; ++x)
	{
		for (int y = 0; y < dy; ++y)
		{
			int index0 = y * dx + x;

			if (y > 0)
			{
				int index1 = (y - 1) * dx + x;
				CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
			}

			if (y > 1)
			{
				int index2 = (y - 2) * dx + x;
				CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
			}
		}
	}
}

void FlexSystem::AddSDF(NvFlexDistanceFieldId sdf, Vec3 translation, Quat rotation, float width)
{
	NvFlexCollisionGeometry geo;
	geo.sdf.field = sdf;
	geo.sdf.scale = width;

	g_buffers->shapePositions.push_back(Vec4(translation, 0.0f));
	g_buffers->shapeRotations.push_back(Quat(rotation));
	g_buffers->shapePrevPositions.push_back(Vec4(translation, 0.0f));
	g_buffers->shapePrevRotations.push_back(Quat(rotation));
	g_buffers->shapeGeometry.push_back((NvFlexCollisionGeometry&)geo);
	g_buffers->shapeFlags.push_back(NvFlexMakeShapeFlags(eNvFlexShapeSDF, false));
}

void FlexSystem::ClearShapes()
{
	g_buffers->shapeGeometry.resize(0);
	g_buffers->shapePositions.resize(0);
	g_buffers->shapeRotations.resize(0);
	g_buffers->shapePrevPositions.resize(0);
	g_buffers->shapePrevRotations.resize(0);
	g_buffers->shapeFlags.resize(0);
}

void FlexSystem::setShapes(){

	if(g_buffers->shapeFlags.size()){
	NvFlexSetShapes(
				g_solver,
				g_buffers->shapeGeometry.buffer,
				g_buffers->shapePositions.buffer,
				g_buffers->shapeRotations.buffer,
				g_buffers->shapePrevPositions.buffer,
				g_buffers->shapePrevRotations.buffer,
				g_buffers->shapeFlags.buffer,
				g_buffers->shapeFlags.size());
		}
}

void FlexSystem::emission(){

	size_t e=0;

	for (; e < nEmitter; ++e)
	{
		if (!g_rectEmitters[e].mEnabled)
			continue;

		Vec3 emitterRot = g_rectEmitters[e].mRot;
		Point3 emitterSize = g_rectEmitters[e].mSize;
		Point3 emitterPos = g_rectEmitters[e].mPos;

		float r = g_params.fluidRestDistance;
		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);

		float numSlices = (g_rectEmitters[e].mSpeed / r)*g_dt;

		// whole number to emit
		int n = int(numSlices + g_rectEmitters[e].mLeftOver);
				
		if (n)
			g_rectEmitters[e].mLeftOver = (numSlices + g_rectEmitters[e].mLeftOver)-n;
		else
			g_rectEmitters[e].mLeftOver += numSlices;

		//int circle = 1;

		int disc = g_rectEmitters[e].mDisc;

		int dy = 0;

				
		int dx = int(ceilf(emitterSize.x / g_params.fluidRestDistance));
		if(disc)
			dy = int(ceilf(emitterSize.x / g_params.fluidRestDistance));
		else
			dy = int(ceilf(emitterSize.y / g_params.fluidRestDistance));
		Mat44	tMat = TransformMatrix(Rotation(emitterRot.y, emitterRot.z, emitterRot.x), emitterPos);


		for (int z=0; z < n; ++z)
				{
			for (int x=0; x < dx; ++x)
			{
				for (int y=0; y < dy; ++y)
				{
						
					Point3 position = g_params.fluidRestDistance*Point3(float(x) - 0.5*(dx-1), float(y) - 0.5*(dy-1), float(z)) + RandomUnitVector()*g_params.fluidRestDistance*0.01f;

					int keep = 1;

					if(disc){
						if(position.x*position.x + position.y*position.y>0.25*emitterSize.x*emitterSize.x)
							keep=0;
					}

					if(g_rectEmitters[e].mNoise){
						Point3 scaledP = position*g_rectEmitters[e].mNoiseFreq + Point3(0,0,g_rectEmitters[e].mNoiseOffset);
						const float kNoise = Perlin3D(scaledP.x, scaledP.y, scaledP.z, 1, 0.25f);

						if(kNoise<g_rectEmitters[e].mNoiseThreshold)
							keep=0;

					}

					if(keep) {

						position = tMat*position;

						Vec3 vel = Vec3(0,0,g_rectEmitters[e].mSpeed);
						vel = tMat*vel;

						g_buffers->positions[cursor] = Vec4(Vec3(position), 1.0f);
						g_buffers->velocities[cursor] = vel;
						g_buffers->phases[cursor] = phase;

						if(g_buffers->activeIndices.size()<maxParticles)
							g_buffers->activeIndices.push_back(cursor);

						if(cursor<maxParticles-1)
							cursor++;
						else
							cursor = 0;

						
					}//end dist
				}
			}
		}

	}
}

void FlexSystem::update() {

		activeParticles = NvFlexGetActiveCount(g_solver);
		
		time1 = GetSeconds();

		emission();

		time2 = GetSeconds();

		g_buffers->UnmapBuffers();

		NvFlexCopyDesc copyDesc;
		copyDesc.dstOffset = 0;
		copyDesc.srcOffset = 0;

		if (g_buffers->activeIndices.size()) {

			copyDesc.elementCount = g_buffers->activeIndices.size();

			NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, &copyDesc);
			NvFlexSetActiveCount(g_solver, g_buffers->activeIndices.size());
		}

		if (g_buffers->positions.size()) {

			copyDesc.elementCount = g_buffers->positions.size();

			NvFlexSetParticles(g_solver, g_buffers->positions.buffer, &copyDesc);
			NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, &copyDesc);
			NvFlexSetPhases(g_solver, g_buffers->phases.buffer, &copyDesc);

			NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, &copyDesc);

		}

		// dynamic triangles
		if (g_buffers->triangles.size())
		{
			NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangles.size() / 3);
		}

		// inflatables
		if (g_buffers->inflatableTriOffsets.size())
		{
			NvFlexSetInflatables(g_solver, g_buffers->inflatableTriOffsets.buffer, g_buffers->inflatableTriCounts.buffer, g_buffers->inflatableVolumes.buffer, g_buffers->inflatablePressures.buffer, g_buffers->inflatableCoefficients.buffer, g_buffers->inflatableTriOffsets.size());
		}

		// Forcefield
		if (/*forcefield != NULL && */callback != nullptr && nFields > 0)
			NvFlexExtSetForceFields(callback, &forcefield, nFields);


		setShapes();

		NvFlexSetParams(g_solver, &g_params);

		NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

		if(g_buffers->positions.size()){

			copyDesc.elementCount = g_buffers->positions.size();

			NvFlexGetParticles(g_solver, g_buffers->positions.buffer, &copyDesc);
			NvFlexGetParticles(g_solver, g_buffers->positionsGpu.buffer, &copyDesc);

			NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, &copyDesc);
			NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer, g_buffers->anisotropy3.buffer, &copyDesc);

			NvFlexGetVelocities(g_solver, g_buffers->velocities.buffer, &copyDesc);
			NvFlexGetNormals(g_solver, g_buffers->normalsGpu.buffer, NULL);
		}

		if (g_buffers->triangles.size()){
			NvFlexGetDynamicTriangles(g_solver, g_buffers->trianglesGpu.buffer, g_buffers->triangleNormalsGpu.buffer, g_buffers->triangles.size() / 3);
		}

		activeParticles = NvFlexGetActiveCount(g_solver);
}
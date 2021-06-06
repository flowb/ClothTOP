#pragma once

#include <stddef.h>
#include <vector>

#include <NvFlex.h>
#include <NvFlexExt.h>
#include <NvFlexDevice.h>

#include <core/types.h>
#include <core/maths.h>
#include <core/platform.h>
#include <core/mesh.h>
#include <core/perlin.h>
#include <core/cloth.h>

#include "TOP_CPlusPlusBase.h"

using namespace std;

inline float sqr(float x) { return x*x; }

struct Emitter
{
	Emitter() : mSpeed(0.0f), mEnabled(false), mLeftOver(0.0f), mWidth(8) {}

	Vec3 mPos;
	Vec3 mDir;
	Vec3 mRight;
	float mSpeed;
	bool mEnabled;
	float mLeftOver;
	int mWidth;
};

struct RectEmitter
{
	RectEmitter() : mSpeed(0.0f), mEnabled(false), mLeftOver(0.0f),
		mDisc(0), mNoise(0), mNoiseThreshold(0.0f), mNoiseFreq(0.0f), mNoiseOffset(0.0f) {}

	Point3 mPos;
	Point3 mSize;
	Vec3 mRot;

	float mSpeed;
	bool mEnabled;
	float mLeftOver;

	int mDisc;
	int mNoise;
	float mNoiseThreshold;
	float mNoiseFreq;
	float mNoiseOffset;

};

struct VolumeBox
{
	VolumeBox() {}

	Point3 mPos;
	Point3 mSize;
	Vec3 mRot;
};

struct GpuTimers
{
	unsigned long long computeBegin;
	unsigned long long computeEnd;
	unsigned long long computeFreq;
};

struct VMesh
{
	uint32_t GetNumVertices() const { return uint32_t(m_positions.size()); }
	uint32_t GetNumFaces() const { return uint32_t(m_indices.size()) / 3; }

	void GetBounds(Vector3& minExtents, Vector3& maxExtents) const;

	Vector3 minExtents;
	Vector3 maxExtents;

	std::vector<Vector4> m_positions;
	std::vector<uint32_t> m_indices;
};

struct SimBuffers
{
	// particles
	NvFlexVector<Vec4> positions;
	NvFlexVector<Vec4> positionsGpu;
	NvFlexVector<Vec4> restPositions;
	NvFlexVector<Vec3> velocities;
	NvFlexVector<int> phases;
	NvFlexVector<float> densities;
	NvFlexVector<Vec4> anisotropy1;
	NvFlexVector<Vec4> anisotropy2;
	NvFlexVector<Vec4> anisotropy3;
	NvFlexVector<Vec4> normals;
	NvFlexVector<Vec4> normalsGpu;
	NvFlexVector<Vec4> smoothPositions;
	NvFlexVector<Vec4> diffusePositions;
	NvFlexVector<Vec4> diffuseVelocities;
	NvFlexVector<int> diffuseCount;

	NvFlexVector<int> activeIndices;

	// convexes
	NvFlexVector<NvFlexCollisionGeometry> shapeGeometry;
	NvFlexVector<Vec4> shapePositions;
	NvFlexVector<Quat> shapeRotations;
	NvFlexVector<Vec4> shapePrevPositions;
	NvFlexVector<Quat> shapePrevRotations;
	NvFlexVector<int> shapeFlags;

	// inflatables
	NvFlexVector<int> inflatableTriOffsets;
	NvFlexVector<int> inflatableTriCounts;
	NvFlexVector<float> inflatableVolumes;
	NvFlexVector<float> inflatableCoefficients;
	NvFlexVector<float> inflatablePressures;

	// springs
	NvFlexVector<int> springIndices;
	NvFlexVector<float> springLengths;
	NvFlexVector<float> springStiffness;

	// triangles
	NvFlexVector<int> triangles;
	NvFlexVector<int> trianglesGpu;
	NvFlexVector<Vec3> triangleNormals;
	NvFlexVector<Vec3> triangleNormalsGpu;
	NvFlexVector<Vec3> uvs;
	NvFlexVector<Vec3> uvsGpu;

	//contacts
	NvFlexVector<Vec4> contactPlanes;
	NvFlexVector<Vec4> contactVelocities;
	NvFlexVector<int> contactIndices;
	NvFlexVector<unsigned int> contactCounts;

	SimBuffers(NvFlexLibrary* l);
	~SimBuffers();

	void MapBuffers();
	void UnmapBuffers();
	void InitBuffers();
};

// Singleton  DP
class FlexSystem 
{
public:
	static FlexSystem& getInstance()
	{
		static FlexSystem myObject;
		return myObject;
	}

	FlexSystem(FlexSystem const&) = delete;
	void operator=(FlexSystem const&) = delete;

	void initScene();
	void initSystem();
	void postInitScene();
	void update();
	void ClearShapes();
	void getSimTimers();

	void updateParams(const OP_Inputs* inputs);
	void initTriangleMesh(const OP_Inputs* inputs);
	void initClothMesh(const OP_Inputs* inputs);
	void updateTriangleMesh(const OP_Inputs* inputs);
	void updatePlanes(const OP_Inputs* inputs);
	void updateSpheresCols(const OP_Inputs* inputs);
	void updateBoxesCols(const OP_Inputs* inputs);
	void updateCloths(const OP_Inputs* inputs);

	void AddTriangleMesh(NvFlexTriangleMeshId mesh, Vec3 translation, Quat rotation, Vec3 prevTrans, Quat prevRot, Vec3 scale);

	NvFlexParams g_params;
	NvFlexSolver* g_solver;
	SimBuffers* g_buffers;
	int activeParticles;
	int maxParticles;
	vector<int> g_inactiveIndices;
	float simLatency;
	int cursor;
	int nFields;

	NvFlexExtForceField* forcefieldRadial;
	VMesh* g_triangleCollisionMesh;
	Vec3 curMeshTrans;
	Quat curMeshRot;
	Vec3 previousMeshTrans;
	Quat previousMeshRot;
	GpuTimers g_GpuTimers;
	NvFlexTriangleMeshId triangleCollisionMeshId;
	NvFlexTimers g_timers;

private:
	FlexSystem();
	~FlexSystem();

	void initParams();

	void GetParticleBounds(Vec3& lower, Vec3& upper);
	void CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter);
	void CreateCenteredParticleGrid(Point3 position, Vec3 rotation, Point3 size, float restDistance, Vec3 velocity, float invMass, bool rigid, int phase, float jitter = 0.005f);
	void CreateSpring(int i, int j, float stiffness, float give);
	void CreateSpringGrid(Vec3 lower, int dx, int dy, int dz, float radius, int phase, float stretchStiffness, float bendStiffness, float shearStiffness, Vec3 velocity, float invMass);

	void setShapes();
	void AddSphere(float radius, Vec3 position, Quat rotation);
	void AddBox(Vec3 halfEdge = Vec3(2.0f), Vec3 center = Vec3(0.0f), Quat quat = Quat(), bool dynamic = false);

	NvFlexTriangleMeshId CreateTriangleMesh(VMesh* m);
	void UpdateTriangleMesh(VMesh* m, NvFlexTriangleMeshId flexMeshId);

	int deformingMesh;

	ClothMesh* clothMesh0;

	NvFlexExtForceFieldCallback* callback;
	NvFlexLibrary* g_flexLib;
	NvFlexParams g_defaultParams;

	bool g_profile;
	int numDiffuse;

	vector<RectEmitter> g_rectEmitters;
	vector<VolumeBox> g_volumeBoxes;

	int g_numSubsteps;
	float g_dt;

	Vec3 g_sceneLower;
	Vec3 g_sceneUpper;

	int g_maxDiffuseParticles;
	int g_maxContactsPerParticle;
	unsigned char g_maxNeighborsPerParticle;

	int nEmitter;
	int nVolumeBoxes;

	NvFlexSolverDesc g_solverDesc;
	int g_device;
};

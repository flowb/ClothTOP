#pragma once

#include <stddef.h>

#include <NvFlex.h>
#include <vector>

#include <core/types.h>
#include <core/maths.h>


#include <core/platform.h>
#include <core/mesh.h>

#include <core/perlin.h>

#include <NvFlexExt.h>
#include <NvFlexDevice.h>

#include <core/cloth.h>


//#include "TOP_CPlusPlusBase.h"
//#include "CPlusPlus_Common.h"


using namespace std;

inline float sqr(float x) { return x*x; }

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

struct Emitter
{
	Emitter() : mSpeed(0.0f), mEnabled(false), mLeftOver(0.0f), mWidth(8)   {}

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
	RectEmitter() : mSpeed(0.0f), mEnabled(false), mLeftOver(0.0f), mDisc(0), mNoise(0)  {}

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
	
	VolumeBox()  {}

	Point3 mPos;
	Point3 mSize;
	Vec3 mRot;

};

struct GpuTimers
{
	/*unsigned long long renderBegin;
	unsigned long long renderEnd;
	unsigned long long renderFreq;*/
	unsigned long long computeBegin;
	unsigned long long computeEnd;
	unsigned long long computeFreq;

	/*static const int maxTimerCount = 4;
	double timers[benchmarkEndFrame][maxTimerCount];
	int timerCount[benchmarkEndFrame];*/
};

struct SimBuffers
{
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

	// rigids
	NvFlexVector<int> rigidOffsets;
	NvFlexVector<int> rigidIndices;
	NvFlexVector<int> rigidMeshSize;
	NvFlexVector<float> rigidCoefficients;
	NvFlexVector<float> rigidPlasticThresholds;
	NvFlexVector<float> rigidPlasticCreeps;
	NvFlexVector<Quat> rigidRotations;
	NvFlexVector<Vec3> rigidTranslations;
	NvFlexVector<Vec3> rigidLocalPositions;
	NvFlexVector<Vec4> rigidLocalNormals;

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

	//fields
	//NvFlexVector<float> fieldCollider;

	SimBuffers(NvFlexLibrary* l);
	~SimBuffers();

	void MapBuffers();
	void UnmapBuffers();
	void InitBuffers();

};

class FlexSystem {

	public:

	FlexSystem();
	~FlexSystem();

	void initSystem();
	void initParams();

	void getSimTimers();

	void GetParticleBounds(Vec3& lower, Vec3& upper);
	void CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter);
	void CreateCenteredParticleGrid(Point3 position, Vec3 rotation, Point3 size, float restDistance, Vec3 velocity, float invMass, bool rigid, int phase, float jitter = 0.005f);

	// springs 
	void CreateSpring(int i, int j, float stiffness, float give);

	//Vec3 SampleSDFGrad(const float* sdf, int dim, int x, int y, int z);
	//void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter, Vec3 skinOffset, float skinExpand, Vec4 color, float springStiffness);
	//void CreateParticleShape(const char* filename, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter, Vec3 skinOffset, float skinExpand, Vec4 color, float springStiffness);
	//inline std::string GetFilePathByPlatform(const char* path);
	//void CalculateRigidLocalPositions(const Vec4* restPositions, int numRestPositions, const int* offsets, const int* indices, int numRigids, Vec3* localPositions);

	void AddSDF(NvFlexDistanceFieldId sdf, Vec3 translation, Quat rotation, float width);

	void ClearShapes();
	void setShapes();

	void AddSphere(float radius, Vec3 position, Quat rotation);

	void AddBox(Vec3 halfEdge = Vec3(2.0f), Vec3 center = Vec3(0.0f), Quat quat = Quat(), bool dynamic = false);

	NvFlexTriangleMeshId CreateTriangleMesh(VMesh* m);
	void UpdateTriangleMesh(VMesh* m, NvFlexTriangleMeshId flexMeshId);
	void AddTriangleMesh(NvFlexTriangleMeshId mesh, Vec3 translation, Quat rotation, Vec3 prevTrans, Quat prevRot, Vec3 scale);

	void CreateSpringGrid(Vec3 lower, int dx, int dy, int dz, float radius, int phase, float stretchStiffness, float bendStiffness, float shearStiffness, Vec3 velocity, float invMass);

	void emission();
	void update();

	void initScene();
	void postInitScene();

	// triangle collision
	int deformingMesh;

	Vec3 curMeshTrans;
	Quat curMeshRot;

	Vec3 previousMeshTrans;
	Quat previousMeshRot;

	VMesh* g_triangleCollisionMesh;
	NvFlexTriangleMeshId triangleCollisionMeshId;

	// cloth
	ClothMesh* clothMesh0;

	//forcefield
	NvFlexExtForceField forcefield;
	NvFlexExtForceFieldCallback* callback;
	int nFields;

	NvFlexSolver* g_solver;
	NvFlexLibrary* g_flexLib;

	NvFlexParams g_params;
	NvFlexParams g_defaultParams;

	SimBuffers* g_buffers;

	vector<int> g_inactiveIndices;

	bool g_profile;

	int activeParticles;
	int maxParticles;
	int numDiffuse;


	vector<RectEmitter> g_rectEmitters;
	vector<VolumeBox> g_volumeBoxes;


	int g_numSubsteps;
	float g_dt;	// the time delta used for simulation

	Vec3 g_sceneLower;
	Vec3 g_sceneUpper;


	int g_maxDiffuseParticles;
	int g_maxContactsPerParticle;
	unsigned char g_maxNeighborsPerParticle;


	int nEmitter;
	int nVolumeBoxes;

	double time1;
	double time2;
	double time3;
	double time4;

	NvFlexTimers g_timers;

	float simLatency;

	int cursor;

	NvFlexSolverDesc g_solverDesc;

	GpuTimers g_GpuTimers;

	int g_device;

};

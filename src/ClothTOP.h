#include "TOP_CPlusPlusBase.h"
#include "FlexSystem.h"

#include <stdio.h>
#include <cuda_gl_interop.h>

using namespace std;

typedef unsigned int uint;
#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0);} }

class ClothTOP : public TOP_CPlusPlusBase
{
public:
	ClothTOP(const OP_NodeInfo* info, TOP_Context* context);
	virtual ~ClothTOP();

	virtual void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved1) override;
	virtual bool		getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void* reserved1) override;


	virtual void		execute(TOP_OutputFormatSpecs*,
		const OP_Inputs*,
		TOP_Context* context, void* reserved1) override;


	virtual int32_t		getNumInfoCHOPChans(void* reserved1) override;
	virtual void		getInfoCHOPChan(int32_t index,
		OP_InfoCHOPChan* chan,
		void* reserved1) override;

	virtual bool		getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved1) override;
	virtual void		getInfoDATEntries(int32_t index,
		int32_t nEntries,
		OP_InfoDATEntries* entries,
		void* reserved1) override;

	virtual void		getErrorString(OP_String* error, void* reserved1) override;

	virtual void		setupParameters(OP_ParameterManager* manager, void* reserved1) override;
	virtual void		pulsePressed(const char* name, void* reserved1) override;

	// FlexSystem members
	void updateParams(const OP_Inputs* inputs);
	void initTriangleMesh(const OP_Inputs* inputs);
	void initClothMesh(const OP_Inputs* inputs);
	void updateTriangleMesh(const OP_Inputs* inputs);
	void updatePlanes(const OP_Inputs* inputs);
	void updateSpheresCols(const OP_Inputs* inputs);
	void updateBoxesCols(const OP_Inputs* inputs);
	void updateCloths(const OP_Inputs* inputs);

	float	maxVel;
	int		activeCount;
	int		maxParticles;
	int		activeIndicesSize;
	int		inactiveIndicesSize;
	float	timer;
	int		numPos;
	int		numQuads;
	int		numFaces;

	FlexSystem* FlexSys;

	void setupParamsCustom(OP_ParameterManager* manager);
	void setupParamsSolver(OP_ParameterManager* manager);
	void setupParamsParts(OP_ParameterManager* manager);
	void setupParamsCollisions(OP_ParameterManager* manager);
	void setupParamsCloths(OP_ParameterManager* manager);

	// CUDA-GL interop
	cudaGraphicsResource* cuGlResource_outColorBuffer1;
	cudaGraphicsResource* cuGlResource_outColorBuffer2;
	cudaArray* cuArray_outColorBuffer1;
	cudaArray* cuArray_outColorBuffer2;

private:
	const OP_NodeInfo		*myNodeInfo;
	int						 myExecuteCount;
	const char* myError;
};

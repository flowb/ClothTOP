uniform float uBumpScale;
uniform vec4 uBaseColor;
uniform float uMetallic;
uniform float uRoughness;
uniform float uReflectance;
uniform float uSpecularLevel;
uniform float uAmbientOcclusion;
uniform vec3 uRimColor[1];
uniform vec3 uRimDir[1];
uniform float uRimWidth[1];
uniform float uRimStrength[1];
uniform float uShadowStrength;
uniform vec3 uShadowColor;

uniform sampler2D sPos;
uniform sampler2D sNorm;

in float primID;
in float idx;
in vec4 T;

//out vec3 ioUVUnwrapCoord;

out Vertex
{
	vec4 color;
	mat3 tangentToWorld;
	vec3 worldSpacePos;
	vec2 texCoord0;
	flat int cameraIndex;
} oVert;

void main()
{
	
	vec3 texcoord = TDInstanceTexCoord(uv[0]);
	oVert.texCoord0.st = texcoord.st;

	vec4 worldSpacePos = texture(sPos, texcoord.st);
	worldSpacePos.w = 1.0f;
	vec3 uvUnwrapCoord = TDInstanceTexCoord(TDUVUnwrapCoord());

	gl_Position = TDWorldToProj(worldSpacePos);
	//gl_Position = worldSpacePos;

	//ioUVUnwrapCoord = uvUnwrapCoord;

	int cameraIndex = TDCameraIndex();
	oVert.cameraIndex = cameraIndex;
	oVert.worldSpacePos.xyz = worldSpacePos.xyz;
	oVert.color = TDInstanceColor(Cd);
	vec3 worldSpaceNorm = texture(sNorm, texcoord.st).xyz;

	vec3 worldSpaceTangent = TDDeformNorm(T.xyz);
	worldSpaceTangent.xyz = normalize(worldSpaceTangent.xyz);

	mat3 tangentToWorld = TDCreateTBNMatrix(worldSpaceNorm, worldSpaceTangent, T.w);
	tangentToWorld[2][0] = worldSpaceNorm.x;
	tangentToWorld[2][1] = worldSpaceNorm.y;
	tangentToWorld[2][2] = worldSpaceNorm.z;

	oVert.tangentToWorld = tangentToWorld ;
}

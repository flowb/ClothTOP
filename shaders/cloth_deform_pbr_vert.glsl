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

in float primID;
in float idx;
in vec4 T;

out vec3 ioUVUnwrapCoord;

out Vertex
{
	vec4 color;
	mat3 tangentToWorld;
	vec3 worldSpacePos;
	vec2 texCoord0;
	flat int cameraIndex;
	flat float id;
} oVert;

void main()
{
	{
		vec3 texcoord = TDInstanceTexCoord(uv[0]);
		oVert.texCoord0.st = texcoord.st;
	}
	// First deform the vertex and normal
	// TDDeform always returns values in world space
	vec4 worldSpacePos = TDDeform(P);
	vec3 uvUnwrapCoord = TDInstanceTexCoord(TDUVUnwrapCoord());
	// Let the geometry shader do the conversion to projection space.
	gl_Position = worldSpacePos;

	ioUVUnwrapCoord = uvUnwrapCoord;

	int cameraIndex = TDCameraIndex();
	oVert.cameraIndex = cameraIndex;
	oVert.worldSpacePos.xyz = worldSpacePos.xyz;
	oVert.color = TDInstanceColor(Cd);
	vec3 worldSpaceNorm = normalize(TDDeformNorm(N));

	oVert.id = primID;

	vec3 worldSpaceTangent = TDDeformNorm(T.xyz);
	worldSpaceTangent.xyz = normalize(worldSpaceTangent.xyz);

	oVert.tangentToWorld = TDCreateTBNMatrix(worldSpaceNorm, worldSpaceTangent, T.w);
}

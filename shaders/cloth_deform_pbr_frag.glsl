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

uniform sampler2D sNormalMap;
uniform sampler2D sBaseColorMap;
uniform sampler2D sMetallicMap;
uniform sampler2D sRoughnessMap;

in Vertex
{
	vec4 color;
	mat3 tangentToWorld;
	vec3 worldSpacePos;
	vec2 texCoord0;
	flat int cameraIndex;
} iVert;

// Output variable for the color
layout(location = 0) out vec4 oFragColor[TD_NUM_COLOR_BUFFERS];
void main()
{
	// This allows things such as order independent transparency
	// and Dual-Paraboloid rendering to work properly
	TDCheckDiscard();

	vec4 outcol = vec4(0.0, 0.0, 0.0, 0.0);
	vec3 diffuseSum = vec3(0.0, 0.0, 0.0);
	vec3 specularSum = vec3(0.0, 0.0, 0.0);

	vec2 texCoord0 = iVert.texCoord0.st;
	vec3 worldSpaceNorm = iVert.tangentToWorld[2];

	vec3 camSpaceNorm = mat3(uTDMats[iVert.cameraIndex].camForNormals) * worldSpaceNorm;
	camSpaceNorm = normalize(camSpaceNorm);


	// 0.08 is the value for dielectric specular that
	// Substance Designer uses for it's top-end.
	float specularLevel = 0.08 *uSpecularLevel;
	float metallic = uMetallic;

	float roughness = uRoughness;

	vec3 finalBaseColor = uBaseColor.rgb* iVert.color.rgb;

	vec4 baseColorMap = texture(sBaseColorMap, texCoord0.st);
	finalBaseColor = baseColorMap.rgb;

	float mappingFactor = 1.0f;

	vec4 metallicMapColor = texture(sMetallicMap, texCoord0.st);
	mappingFactor = metallicMapColor.r;
	metallic *= mappingFactor;

	vec4 roughnessMapColor = texture(sRoughnessMap, texCoord0.st);
	mappingFactor = roughnessMapColor.r;
	roughness *= mappingFactor;


	// A roughness of exactly 0 is not allowed
	roughness = max(roughness, 0.0001);

	vec3 pbrDiffuseColor = finalBaseColor * (1.0 - metallic);
	vec3 pbrSpecularColor = mix(vec3(specularLevel), finalBaseColor, metallic);

	vec3 viewVec = normalize(uTDMats[iVert.cameraIndex].camInverse[3].xyz - iVert.worldSpacePos.xyz );

	if (!TDFrontFacing(iVert.worldSpacePos.xyz, worldSpaceNorm.xyz))
	{
		worldSpaceNorm = -worldSpaceNorm;
		camSpaceNorm = -camSpaceNorm;
	}

	for (int i = 0; i < TD_NUM_LIGHTS; i++)
	{
		vec3 diffuseContrib = vec3(0);
		vec3 specularContrib = vec3(0);
		TDLightingPBR(diffuseContrib,
			specularContrib,
			i,
			pbrDiffuseColor,
			pbrSpecularColor,
			iVert.worldSpacePos.xyz,
			worldSpaceNorm,
			uShadowStrength, uShadowColor,
			viewVec,
			roughness
		);
		diffuseSum += diffuseContrib;
		specularSum += specularContrib;
	}

	for (int i = 0; i < TD_NUM_ENV_LIGHTS; i++)
	{
		vec3 diffuseContrib = vec3(0);
		vec3 specularContrib = vec3(0);
		TDEnvLightingPBR(diffuseContrib,
			specularContrib,
			i,
			pbrDiffuseColor,
			pbrSpecularColor,
			worldSpaceNorm,
			viewVec,
			roughness,
			uAmbientOcclusion
		);
		diffuseSum += diffuseContrib;
		specularSum += specularContrib;
	}

	outcol.rgb += specularSum  + diffuseSum;

	// Rim Light Setup
	float projVal;
	vec3 projNorm = vec3(camSpaceNorm.x, camSpaceNorm.y, 0.0);
	projVal = length(projNorm);
	if (projVal == 0.0)
		projNorm = vec3(0.0);
	else 
		projNorm /= projVal;

	// Rim Light 0
	{
		vec2 rimval = vec2((1.0 - dot(camSpaceNorm, vec3(0.0, 0.0, 1.0))), 0.5);
		rimval.r *= uRimStrength[0];
		projVal = dot(projNorm, uRimDir[0]);
		rimval *= clamp(((projVal * 0.5) - 0.5) + uRimWidth[0], 0.0, 1.0);
		outcol.rgb += rimval.r * uRimColor[0];
	}

	// Apply fog, this does nothing if fog is disabled
	outcol = TDFog(outcol, iVert.worldSpacePos.xyz, iVert.cameraIndex);

	// Alpha Calculation
	float alpha = uBaseColor.a * iVert.color.a ;

	// Dithering, does nothing if dithering is disabled
	outcol = TDDither(outcol);

	outcol.rgb *= alpha;
	//outcol.rgb  = worldSpaceNorm;

	// Modern GL removed the implicit alpha test, so we need to apply
	// it manually here. This function does nothing if alpha test is disabled.
	TDAlphaTest(alpha);

	outcol.a = alpha;
	oFragColor[0] = TDOutputSwizzle(outcol);


	// TD_NUM_COLOR_BUFFERS will be set to the number of color buffers
	// active in the render. By default we want to output zero to every
	// buffer except the first one.
	for (int i = 1; i < TD_NUM_COLOR_BUFFERS; i++)
	{
		oFragColor[i] = vec4(0.0);
	}
}

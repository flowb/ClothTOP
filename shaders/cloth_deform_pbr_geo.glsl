layout(triangle_strip, max_vertices = 3) out;
layout(triangles) in;

uniform sampler2D sPos;
uniform int primitiveIDIn;

in vec3 ioUVUnwrapCoord[];
in Vertex
{
	vec4 color;
	mat3 tangentToWorld;
	vec3 worldSpacePos;
	vec2 texCoord0;
	flat int cameraIndex;
	flat float id;
} iVert[];

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
	ivec2 texSize = textureSize(sPos, 0);
	
	float offset_indices = iVert[1].id*4.0f+1.0f;

	vec2 texCoord1 = vec2(int(mod(offset_indices,texSize.x)),int(offset_indices)/texSize.x);
	vec2 texCoord1Norm = vec2(texCoord1.x/texSize.x + 0.5/texSize.x, texCoord1.y/texSize.y + 0.5/texSize.y);

	vec3 triIndices = texture(sPos, texCoord1Norm).xyz;
	float triIndicesArray[3] = {triIndices.x, triIndices.y, triIndices.z};

	float offset_pos;
	vec2 texCoord2;
	vec4 vertexWorldPos;
	vec2 texCoord2Norm;
	vec3 worldSpaceNorm;
	mat3 tangentToWorld;

	for (int i = 0; i < 3; i++)
	{
		// fetch indices
		offset_pos = 4.0f*(triIndicesArray[i]);
		texCoord2 = vec2(mod(offset_pos,texSize.x),int(offset_pos)/texSize.x);
		texCoord2Norm = vec2(texCoord2.x/texSize.x + 0.5/texSize.x, texCoord2.y/texSize.y + 0.5/texSize.y);
		vertexWorldPos = vec4(texture(sPos, texCoord2Norm).xyz,1);

		// fetch smooth vertex normals
		offset_pos = 4.0f*(triIndicesArray[i]) + 2;
		texCoord2 = vec2(mod(offset_pos,texSize.x),int(offset_pos)/texSize.x);
		texCoord2Norm = vec2(texCoord2.x/texSize.x + 0.5/texSize.x, texCoord2.y/texSize.y + 0.5/texSize.y);
		worldSpaceNorm = texture(sPos, texCoord2Norm).zyx;

		// to fragment shader
		oVert.color = iVert[i].color;
		oVert.texCoord0 = iVert[i].texCoord0;
		oVert.cameraIndex = iVert[i].cameraIndex;
		oVert.worldSpacePos = vertexWorldPos.xyz;

		tangentToWorld = iVert[i].tangentToWorld;
		tangentToWorld[2][0] = worldSpaceNorm.x;
		tangentToWorld[2][1] = worldSpaceNorm.y;
		tangentToWorld[2][2] = worldSpaceNorm.z;

		oVert.tangentToWorld = tangentToWorld ;

		gl_Position = TDWorldToProj(vertexWorldPos, ioUVUnwrapCoord[i], iVert[i].cameraIndex);
		EmitVertex();
	}

	EndPrimitive();
}

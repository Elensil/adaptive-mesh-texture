#version 330
uniform int selected;
uniform sampler2D myColorMap;
in vec3 baryCoord;

in float R;			//resolution of the face

in vec3 eCw, eCh, vCw, vCh;	//store color indices for the 3 vertex (in the color 'texture') and for the 3 edges ((1,2),(2,3),(3,1))
in float fCw, fCh;			//face color index
out vec4 outputColour;
//convert index to texture coordinates, and get color value
vec4 getColorValue(in uint myIndex, in uint myWidth, in uint myHeight){
	uint Yu = uint((float(myIndex)+0.1f)/myWidth);			//absolute Y coordinate, as a uint
	float Xn = float(myIndex-myWidth*Yu)/myWidth+1.0f/(myWidth*2u);			//normalized coordinates
	float Yn = float(Yu)/myHeight + 1.0f/(myHeight*2u);
	return texture(myColorMap, vec2(Xn,1-Yn));
}

uint edgeCase(in uint v1I, in uint v2I, in int edgeI, in uint sampleI, in uint edgeRes, in uint faceRes, out float myWeight, out uint secondIndex){
	uint cEdgeI=uint(abs(edgeI))+1u;		//+1 since the 1st value is used to store the edge resolution
	if(edgeRes>faceRes){
	   edgeRes=faceRes;
	}
	float fragOffset = float(sampleI * edgeRes) / faceRes;		//translate offset into edge coordinates
	uint iFragOffset = uint(fragOffset+0.00000001);
	uint fragI = 0u;
	myWeight = fragOffset - float(iFragOffset);
	if(edgeRes==1u){
	   myWeight = 1.0-myWeight;
	   fragI = v1I;
	   secondIndex = v2I;
	}
	else{   
    	if (myWeight<=0.00001){						//we're on an edge sample
    		fragI = cEdgeI+iFragOffset-1u;
    		myWeight=1.0;
    	}else if ((1-myWeight)<=0.00001){			//we're on an edge sample (unlikely, but might happen because of float calculations?)
    		fragI = cEdgeI+iFragOffset;
    		myWeight=1.0;
    	}else {										//interpolation case
    		myWeight = 1.0-myWeight;
    		if (iFragOffset==0u){					//interpolate with 1st vertex
    			fragI = v1I;
    			secondIndex = cEdgeI;
    		} else if (iFragOffset==edgeRes-1u) {	//interpolate with 2nd vertex
    			fragI = cEdgeI+iFragOffset-1u;
    			secondIndex = v2I;
    		} else {								//interpolate between two edge samples
    			fragI = cEdgeI+iFragOffset-1u;
    			secondIndex = fragI+1u;
    		}
    	}
    }
	return fragI;
}


//This function takes a sampled point on the triangle, a face resolution and returns its index in the color map.
//For edges, the actual edge resolution might be less than that of the face. In that case, our sample does not have a color value, and must be
//interpolated between two edge samples. In that case, we return the index of one of the samples, and we store its weight in
// 'myWeight'. The second sample index is passed in 'secondIndex' (with weight [1-'myWeight']).
uint getIndex(in uvec3 vCiI, in ivec3 eCiI, in uint fCiI, in uvec2 myP, in uint uR, in uvec3 edgeRes, out float myWeight, out uint secondIndex){
	uint fragI = 0u;
	myWeight=1;
	if (myP.x==uR){							//1st vertex
		fragI = vCiI.x;
	}else if(myP.y==uR){					//2nd vertex
		fragI = vCiI.y;
	}else if((myP.x==0u)&&(myP.y==0u)){		//3rd vertex
		fragI = vCiI.z;
	}else if(myP.x==0u){					//(and myP.y>=1): 2nd edge [v3,v2]
		if ((eCiI.y<0.0)||(eCh.y<0.0)){		//If edge order is inverted, reverse edge index (in range (1,R-1) and vertices)
			fragI = edgeCase(vCiI.y,vCiI.z,eCiI.y, uR-myP.y, edgeRes.y,uR,myWeight,secondIndex);
		}else{
			fragI = edgeCase(vCiI.z,vCiI.y,eCiI.y, myP.y, edgeRes.y,uR,myWeight,secondIndex);
		}
	}else if (myP.y==0u){					//(and myP.x>=1): 3rd edge [v1,v3]
		if ((eCiI.z<0.0)||(eCh.z<0.0)){
			fragI = edgeCase(vCiI.z,vCiI.x,eCiI.z, myP.x, edgeRes.z,uR,myWeight,secondIndex);
		}else{
			fragI = edgeCase(vCiI.x,vCiI.z,eCiI.z, uR-myP.x, edgeRes.z,uR,myWeight,secondIndex);
		}
	}else if ((myP.x+myP.y)==uR){			//1st edge [v2,v1]
		if ((eCiI.x<0.0)||(eCh.x<0.0)){
			fragI = edgeCase(vCiI.x,vCiI.y,eCiI.x, myP.y, edgeRes.x,uR,myWeight,secondIndex);
		}else{
		    fragI = edgeCase(vCiI.y,vCiI.x,eCiI.x, myP.x, edgeRes.x,uR,myWeight,secondIndex);
		}
	}else{									//this is the 'face' case
		fragI = fCiI + (myP.x-1u)*uR - (myP.x*(myP.x+1u))/2u + 1u + (myP.y-1u);		//would be long to explain :(
		//Basically, we store the (R-2) values where Bx=1, then (R-3) values where Bx=2, and so on. This leads to this formula.
	}

	return fragI;
}


void main(void) {

	//texture size
	ivec2 texSize = textureSize(myColorMap,0);
	uint mapWidth = uint(texSize.x);
	uint mapHeight = uint(texSize.y);
	uint uR = uint(R+0.1);

	// Get real indices
	uvec3 vCiI = uvec3(vCh+0.1);
	vCiI = vCiI*mapWidth+uvec3(vCw+0.1);

	ivec3 eCiI = ivec3(abs(eCh)+0.1);
	eCiI = int(mapWidth)*eCiI;
	eCiI = eCiI + ivec3(abs(eCw)+0.1);
	if (eCw.x<0){
		eCiI.x=-eCiI.x;
	}
	if (eCw.y<0){
		eCiI.y=-eCiI.y;
	}
	if (eCw.z<0){
		eCiI.z=-eCiI.z;
	}

	uint fCiI = uint(fCh+0.1);
	fCiI = fCiI*mapWidth+uint(fCw+0.1);

	uint edgeRes1 = uint(255*getColorValue(uint(abs(eCiI.x)),mapWidth,mapHeight).x+0.1);
	uint edgeRes2 = uint(255*getColorValue(uint(abs(eCiI.y)),mapWidth,mapHeight).x+0.1);
	uint edgeRes3 = uint(255*getColorValue(uint(abs(eCiI.z)),mapWidth,mapHeight).x+0.1);

	uvec3 edgeRes = uvec3(edgeRes1,edgeRes2,edgeRes3);

	vec3 w = (R * baryCoord);		//fractional part of barycentric coords
	uvec3 B = uvec3(w);					//integer portion of barycentric coords
	w = w-vec3(B);
	float sumWeight = w.x+w.y+w.z;
	
	//interpolate
	uvec2 cP, cP2, cP3;
	vec3 weights;
	if (sumWeight<0.0001) {
		cP = B.xy;
		weights = vec3(1,0,0);
	} else if((sumWeight-1)<0.0001) {
			cP = B.xy + uvec2(1,0);
			cP2 = B.xy + uvec2(0,1);
			cP3 = B.xy;
			weights = w;

	}else{		//sum equals 2 (theoretically)
			cP = B.xy + uvec2(0,1);
			cP2 = B.xy + uvec2(1,0);
			cP3 = B.xy + uvec2(1,1);
			weights = vec3(1,1,1)-w;
	}

	/*
	//closest point
	if ((weights.x>=weights.y) && (weights.x>=weights.z)){
		weights = vec3(1,0,0);
	}else if ((weights.y>=weights.x) && (weights.y>=weights.z)){
		weights = vec3(0,1,0);
	}else{
		weights = vec3(0,0,1);
	}
	//*/

	float myW1, myW2, myW3;
	uint secI1, secI2, secI3;

	uint fragIndex = getIndex(vCiI, eCiI, fCiI, cP,uR, edgeRes, myW1, secI1);
	uint fragIndex2 = getIndex(vCiI, eCiI, fCiI, cP2, uR, edgeRes, myW2, secI2);
	uint fragIndex3 = getIndex(vCiI, eCiI, fCiI, cP3,uR, edgeRes, myW3, secI3);

	/*
	vec4 final_color = weights.x * getColorValue(fragIndex, mapWidth, mapHeight);
	final_color += weights.y * getColorValue(fragIndex2, mapWidth, mapHeight);
	final_color += weights.z * getColorValue(fragIndex3, mapWidth, mapHeight);
	/*/
	vec4 final_color = weights.x * myW1 * getColorValue(fragIndex, mapWidth, mapHeight);
		final_color += weights.y * myW2 * getColorValue(fragIndex2, mapWidth, mapHeight);
		final_color += weights.z * myW3 * getColorValue(fragIndex3, mapWidth, mapHeight);
		if (myW1<0.9999999){
			final_color += weights.x * (1-myW1) * getColorValue(secI1,mapWidth,mapHeight);
		}
		if (myW2<0.9999999){
			final_color += weights.y * (1-myW2) * getColorValue(secI2,mapWidth,mapHeight);
		}
		if (myW3<0.9999999){
			final_color += weights.z * (1-myW3) * getColorValue(secI3,mapWidth,mapHeight);
		}
	//*/

	//the usual blue tint for selected meshes
	final_color += float(selected)*vec4(0.1, 0.1, 1.0, 1.0);
	//gl_FragColor = final_color;
	outputColour = final_color;
	if(selected==2)
    {
        outputColour = vec4(0.0,0.0,0.0,1.0);
    }
}


#version 330
uniform mat4 modelview;
uniform mat4 projection;
layout (location=0) in vec3 position;
uniform sampler2D myColorMap;
in vec4 packedNormal;
// for mesh colors:


in uvec3 miscAttrib;
in uvec3 vCi;        //store color indices for the 3 vertex (in the color 'texture')
in ivec3 eCi;        //store color indices for the 3 edges ((1,2),(2,3),(3,1))
							//sign bit used for edge reading order
out vec3 baryCoord;		//barycentric coordinates, to be used in fragment shader
out float R;

out vec3 eCw, eCh, vCw, vCh;		// We try splitting color indices into two variables
							// since we lack precision in some cases, and using double is troublesome
out float fCw, fCh;		// The most obvious choice is to use texture coordinates
							// It messes up the computations in the fragment shader, though

void getXY(in uint myIndex, in uint myWidth, in uint myHeight, out float Xn, out float Yn){
	uint Yu = uint((float(myIndex)+0.5f)/myWidth);			//absolute Y coordinate, as a uint
	Xn = float(myIndex-myWidth*Yu);
	Yn = float(Yu);
}


void main(void) {
    

	//texture size
	ivec2 texSize = textureSize(myColorMap,0);
	uint mapWidth = uint(texSize.x);
	uint mapHeight = uint(texSize.y);

    gl_Position = projection * modelview * vec4(position,1.0);

    //gl_Normal = packedNormal.zyx;

    
    uint verBaryCoord = miscAttrib.x;
    R = float(miscAttrib.z);;

    getXY(miscAttrib.y,mapWidth,mapHeight,fCw,fCh);


    getXY(uint(abs(eCi.x)),mapWidth,mapHeight,eCw.x,eCh.x);
    getXY(uint(abs(eCi.y)),mapWidth,mapHeight,eCw.y,eCh.y);
    getXY(uint(abs(eCi.z)),mapWidth,mapHeight,eCw.z,eCh.z);

    getXY(vCi.x,mapWidth,mapHeight,vCw.x,vCh.x);
    getXY(vCi.y,mapWidth,mapHeight,vCw.y,vCh.y);
    getXY(vCi.z,mapWidth,mapHeight,vCw.z,vCh.z);

    if (eCi.x<0){
    	eCw.x = - eCw.x;
    	eCh.x = - eCh.x;
    }
    if (eCi.y<0){
		eCw.y = - eCw.y;
		eCh.y = - eCh.y;
	}
    if (eCi.z<0){
		eCw.z = - eCw.z;
		eCh.z = - eCh.z;
	}

    if (verBaryCoord<=1u){
    	baryCoord = vec3(1,0,0);
    }else if (verBaryCoord<3u){
    	baryCoord = vec3(0,1,0);
    }else{
    	baryCoord = vec3(0,0,1);
    }
    
    //gl_FrontColor = gl_Color;

}


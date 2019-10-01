Shader "Unlit/meshColor"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
	}

	//CGINCLUDE

	//ENDCG

	SubShader
	{
		//Tags { "RenderType"="Opaque" }
		//LOD 100

		Pass
		{
			CGPROGRAM
// Upgrade NOTE: excluded shader from DX11; has structs without semantics (struct v2f members edgeFaceCi,vCi,baryCoord)
#pragma exclude_renderers d3d11

            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;

            struct vertIn {
            	float3 vertex : POSITION;		
            	float2 miscAttrib1 : TEXCOORD0; //x shared between vertex num in triangle + face ind height. y: face res)
            	float4 vCw : TANGENT;			//includes face ind width as 4th component
            	float2 eCw12 : TEXCOORD1;
            	float2 eCwh3 : TEXCOORD2;
            	float3 vCh : NORMAL;
            	float2 eCh12 : TEXCOORD3;

            };

            struct v2f {
            	float4 pos : SV_POSITION;
            	float4 vCw : TANGENT;					//we add face ind width as 4th component
            	float4 vCh : TEXCOORD2;
            	float4 eCw : TEXCOORD0;					//face res as 4th component
            	float4 eCh : TEXCOORD1;					//face res as 4th component
            	float3 baryCoord : COLOR;
            	

			};

			float4 getColorValue(in int myIndex, in int myWidth, in int myHeight){
				int Yu = int((float(myIndex)+0.5F)/myWidth);		//absolute Y coordinate, as a int
				float Xn = float(myIndex-myWidth*Yu)/(myWidth)+1.0F/(2*myWidth);		//normalized coordinates
				float Yn = float(Yu)/(myHeight)+1.0F/(2*myHeight);
				return tex2D(_MainTex, float2(Xn,1-Yn));
			}

			int edgeCase(in int v1I, in int v2I, in int edgeI, in int sampleI, in int edgeRes, in int faceRes, out float myWeight, out int secondIndex){
				int cEdgeI=abs(edgeI)+1;		//+1 since the 1st value is used to store the edge resolution
				float fragOffset = float(sampleI * edgeRes) / faceRes;		//translate offset into edge coordinates
				int iFragOffset = int(fragOffset+0.00000001);
				int fragI = 0;
				myWeight = fragOffset - float(iFragOffset);
				if (myWeight<=0.00001){								//we're on an edge sample
					fragI = cEdgeI+iFragOffset-1;
					myWeight=1;
				}else if ((1-myWeight)<=0.00001){					//we're on an edge sample (unlikely, but might happen because of float calculations?)
					fragI = cEdgeI+iFragOffset;
					myWeight=1;
				}else{												//interpolation case
					myWeight = 1-myWeight;
					if (edgeRes==1){
						fragI = v1I;
						secondIndex = v2I;
					} else if(iFragOffset==0){								//interpolate with 1st vertex
						fragI = v1I;
						secondIndex = cEdgeI;
					} else if (iFragOffset==edgeRes-1) {			//interpolate with 2nd vertex
						fragI = cEdgeI+iFragOffset-1;
						secondIndex = v2I;
					} else {										//interpolate between two edge samples
						fragI = cEdgeI + iFragOffset-1;
						secondIndex = fragI+1;
					}
				}
				return fragI;
			}


			//This function takes a sampled point on the triangle, a face resolution and returns its index in the color map.
			//For edges, the actual edge resolution might be less than that of the face. In that case, our sample does not have a color value, and must be
			//interpolated between two edge samples. In that case, we return the index of one of the samples, and we store its weight in
			// 'myWeight'. The second sample index is passed in 'secondIndex' (with weight [1-'myWeight']).
			int getIndex(in int3 vCiI, in int3 eCiI, in int fCiI, in int2 myP, in int uR, in int3 edgeRes, out float myWeight, out int secondIndex){
				int fragI = 0;
				myWeight=1.0F;
				if (myP.x==uR){							//1st vertex
					fragI = vCiI.x;
					//fragI = vCiI.y;
				}else if(myP.y==uR){					//2nd vertex
					fragI = vCiI.y;
				}else if((myP.x==0)&&(myP.y==0)){		//3rd vertex
					fragI = vCiI.z;
					//fragI = vCiI.y;
				}else if(myP.x==0){					//(and myP.y>=1): 2nd edge [v3,v2]
					if (eCiI.y<0.0){		//If edge order is inverted, reverse edge index (in range (1,R-1) and vertices)
						fragI = edgeCase(vCiI.y,vCiI.z,eCiI.y, uR-myP.y, edgeRes.y,uR,myWeight,secondIndex);
						//fragI = vCiI.y;
					}else{
						fragI = edgeCase(vCiI.z,vCiI.y,eCiI.y, myP.y, edgeRes.y,uR,myWeight,secondIndex);
						//fragI = vCiI.z;
						//fragI = vCiI.y;
					}
				}else if (myP.y==0){					//(and myP.x>=1): 3rd edge [v1,v3]
					if (eCiI.z<0.0){
						fragI = edgeCase(vCiI.z,vCiI.x,eCiI.z, myP.x, edgeRes.z,uR,myWeight,secondIndex);
						//fragI = vCiI.z;
						//fragI = vCiI.y;
					}else{
						fragI = edgeCase(vCiI.x,vCiI.z,eCiI.z, uR-myP.x, edgeRes.z,uR,myWeight,secondIndex);
						//fragI = vCiI.x;
						//fragI = vCiI.y;
					}
				}else if ((myP.x+myP.y)==uR){			//1st edge [v2,v1]
					if (eCiI.x<0.0){
						fragI = edgeCase(vCiI.x,vCiI.y,eCiI.x, myP.y, edgeRes.x,uR,myWeight,secondIndex);
						//fragI = vCiI.x;
						//fragI = vCiI.y;
					}else{
						fragI = edgeCase(vCiI.y,vCiI.x,eCiI.x, myP.x, edgeRes.x,uR,myWeight,secondIndex);
						//fragI = vCiI.y;
					}
				}else{									//this is the 'face' case
					fragI = fCiI + (myP.x-1)*uR - (myP.x*(myP.x+1))/2 + 1 + (myP.y-1);		//would be long to explain :(
					//Basically, we store the (R-2) values where Bx=1, then (R-3) values where Bx=2, and so on. This leads to this formula.
					//fragI = vCiI.y;
				}
				//secondIndex=vCiI.y;
				//myWeight=1.0F;
				if(uR>2){
					//fragI=0;
				}
				if(uR==1){
					//fragI=10000;
				}

				return fragI;
			}





            v2f vert (vertIn v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);

                int mapWidth = int(_MainTex_TexelSize.z);
				int mapHeight = int(_MainTex_TexelSize.w);

                //barycentric coordinates
                // Old method
                int verBaryCoord = int((v.miscAttrib1.x+0.0)/mapHeight)+1;
                int fCh = int(v.miscAttrib1.x - (verBaryCoord-1) * mapHeight + 0.1);
                if(verBaryCoord<=1){
                	o.baryCoord = float3(1,0,0);
                }else if(verBaryCoord<3){
                	o.baryCoord = float3(0,1,0);
                }else{
                	o.baryCoord = float3(0,0,1);
                }

                // New method - Test
                //int fCh = int(floor(v.miscAttrib1.x+0.01));
                //float verBaryCoordF = v.miscAttrib1.x - fCh;

                //if(verBaryCoordF<=0.33){
                //	o.baryCoord = float3(1,0,0);
                //}else if(verBaryCoordF<=0.66){
                //	o.baryCoord = float3(0,1,0);
                //}else{
                //	o.baryCoord = float3(0,0,1);
                //}


                //vertices color indices
                o.vCh = float4(v.vCh,fCh);
                o.vCw = v.vCw;
                //edge and face color indices
                o.eCw = float4(v.eCw12.x, v.eCw12.y, v.eCwh3.x, v.miscAttrib1.y);
                o.eCh = float4(v.eCh12.x, v.eCh12.y, v.eCwh3.y, v.miscAttrib1.y);
                
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
				int mapWidth = int(_MainTex_TexelSize.z);
				int mapHeight = int(_MainTex_TexelSize.w);
				float R = i.eCw.w;
				int uR = int(R+0.1);

				// Get real indices
				//vertex ind
				int vCi1 = int(floor(i.vCh.x+0.1));
				int vCi2 = int(floor(i.vCh.y+0.1));
				int vCi3 = int(floor(i.vCh.z+0.1));
				int3 vCiI;
				if(false){
					vCiI = int3(vCi1, vCi2, vCi3);
					int vCw1 = int(floor(i.vCw.x+0.1));
					int vCw2 = int(floor(i.vCw.z+0.1));
					int vCw3 = int(floor(i.vCw.z+0.1));
					vCiI = vCiI*mapWidth+int3(vCw1,vCw2,vCw3);
				}else if(false){
					vCiI.x = vCi1 * mapWidth + int(floor(i.vCw.x+0.1));
					vCiI.y = vCi1 * mapWidth + int(floor(i.vCw.y+0.1));
					vCiI.z = vCi1 * mapWidth + int(floor(i.vCw.z+0.1));
				}else{
					vCiI.x = int(floor(vCi1 * mapWidth + i.vCw.x+0.1));
					vCiI.y = int(floor(vCi2 * mapWidth + i.vCw.y+0.1));
					vCiI.z = int(floor(vCi3 * mapWidth + i.vCw.z+0.1));
				}

				//edge ind
				int3 eCiI;
				eCiI.x = int(floor(i.eCh.x+0.5));
				eCiI.y = int(floor(i.eCh.y+0.5));
				eCiI.z = int(floor(i.eCh.z+0.5));

				//eCiI.x = eCiI.x * mapWidth + int(floor(i.eCw.x+0.1));
				//eCiI.y = eCiI.y * mapWidth + int(floor(i.eCw.y+0.1));
				//eCiI.z = eCiI.z * mapWidth + int(floor(i.eCw.z+0.1));

				eCiI.x = int(floor(eCiI.x * mapWidth + i.eCw.x+0.1));
				eCiI.y = int(floor(eCiI.y * mapWidth + i.eCw.y+0.1));
				eCiI.z = int(floor(eCiI.z * mapWidth + i.eCw.z+0.1));
				
				//face Ind
				int fCiI = int(i.vCh.w+0.1);
				fCiI = fCiI * mapWidth + int(i.vCw.w+0.1);

				int edgeRes1 = int(255*getColorValue(abs(eCiI.x),mapWidth,mapHeight).x+0.001);
				int edgeRes2 = int(255*getColorValue(abs(eCiI.y),mapWidth,mapHeight).x+0.001);
				int edgeRes3 = int(255*getColorValue(abs(eCiI.z),mapWidth,mapHeight).x+0.001);

				int3 edgeRes = int3(edgeRes1,edgeRes2,edgeRes3);

				float3 w = (uR * i.baryCoord);		//fractional part of barycentric coords
				int3 B = int3(w);					//integer portion of barycentric coords
				w = w-float3(B);
				float sumWeight = w.x+w.y+w.z;
				
				//interpolate
				int2 cP, cP2, cP3;
				float3 weights;
				if (sumWeight<0.0001) {
					cP = B.xy;
					weights = float3(1,0,0);
				} else if((sumWeight-1)<0.0001) {
						cP = B.xy + int2(1,0);
						cP2 = B.xy + int2(0,1);
						cP3 = B.xy;
						weights = w;

				}else{		//sum equals 2 (theoretically)
						cP = B.xy + int2(0,1);
						cP2 = B.xy + int2(1,0);
						cP3 = B.xy + int2(1,1);
						weights = float3(1,1,1)-w;
				}

				//closest point
				if(false){
					if ((weights.x>=weights.y) && (weights.x>=weights.z)){
						weights = float3(1,0,0);
					}else if ((weights.y>=weights.x) && (weights.y>=weights.z)){
						weights = float3(0,1,0);
					}else{
						weights = float3(0,0,1);
					}
				}
				//weights = float3(0.33,0.33,0.33);


				float myW1, myW2, myW3;
				int secI1, secI2, secI3;

				int fragIndex = getIndex(vCiI, eCiI, fCiI, cP,uR, edgeRes, myW1, secI1);
				int fragIndex2 = getIndex(vCiI, eCiI, fCiI, cP2, uR, edgeRes, myW2, secI2);
				int fragIndex3 = getIndex(vCiI, eCiI, fCiI, cP3,uR, edgeRes, myW3, secI3);

				float4 final_color = weights.x * myW1 * getColorValue(fragIndex, mapWidth, mapHeight);
					final_color += weights.y * myW2 * getColorValue(fragIndex2, mapWidth, mapHeight);
					final_color += weights.z * myW3 * getColorValue(fragIndex3, mapWidth, mapHeight);
					if (myW1<0.9999999F){
						final_color += weights.x * (1.0F-myW1) * getColorValue(secI1,mapWidth,mapHeight);
					}
					if (myW2<0.9999999F){
						final_color += weights.y * (1.0F-myW2) * getColorValue(secI2,mapWidth,mapHeight);
					}
					if (myW3<0.9999999F){
						final_color += weights.z * (1.0F-myW3) * getColorValue(secI3,mapWidth,mapHeight);
					}
                return final_color;
            }
            ENDCG
		}
	}
}

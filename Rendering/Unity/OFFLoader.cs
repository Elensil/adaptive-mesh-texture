/*
Code taken from AARO4130, 2015
Adapted by Matthieu Armando to load OFF meshes with mesh texture format.
(See https://gitlab.inria.fr/marmando/adaptive-mesh-texture)
*/


using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class OFFLoader
{
	public static bool splitByMaterial = false;
	public static string[] searchPaths = new string[] { "", "%FileName%_Textures" + Path.DirectorySeparatorChar };
	//structures
	struct IntVector3
	{
		public Int32 x;
		public Int32 y;
		public Int32 z;
	}
		
	//functions
	#if UNITY_EDITOR
	[MenuItem("GameObject/Import From colored OFF")]
	static void OffLoadMenu()
	{
		string pth = UnityEditor.EditorUtility.OpenFilePanel("Import colored OFF", "", "off");
		if (!string.IsNullOrEmpty(pth))
		{
			System.Diagnostics.Stopwatch s = new System.Diagnostics.Stopwatch();
			s.Start();
			LoadOFFFile(pth);
			Debug.Log("OFF load took " + s.ElapsedMilliseconds + "ms");
			s.Stop();
		}
	}
	#endif

	public static Vector3 ParseVertexFromCMPS(string[] cmps)
	{
		float x = float.Parse(cmps[0]);
		float y = float.Parse(cmps[1]);
		float z = float.Parse(cmps[2]);
		return new Vector3(x, y, z);
	}

	public static string OFFGetFilePath(string path, string basePath, string fileName)
	{
		foreach (string sp in searchPaths)
		{
			string s = sp.Replace("%FileName%", fileName);
			if (File.Exists(basePath + s + path))
			{
				return basePath + s + path;
			}
			else if (File.Exists(path))
			{
				return path;
			}
		}

		return null;
	}

	public static Texture2D LoadPNG(string filePath) {

		Texture2D tex = null;
		byte[] fileData;

		if (File.Exists(filePath))     {
			fileData = File.ReadAllBytes(filePath);

			tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
			tex.LoadImage(fileData); //..this will auto-resize the texture dimensions.

		}
		return tex;
	}

	public static GameObject LoadOFFFile(string fn)
	{

		string meshName = Path.GetFileNameWithoutExtension(fn);
		FileInfo moffFileInfo = new FileInfo(fn);
		string moffFileDirectory = moffFileInfo.Directory.FullName; // + Path.DirectorySeparatorChar;
		//OBJ LISTS
		List<Vector3> vertices = new List<Vector3>();
		List<Int32> vertColorIndices = new List<Int32> ();
		List<IntVector3> faceList = new List<IntVector3> ();
		List<uint> faceRes = new List<uint> ();
		List<IntVector3> edgeColorIndices = new List<IntVector3> ();
		List<uint> faceColorIndices = new List<uint> ();
		Texture2D myColorMap = new Texture2D(1,1);
		List<string> objectNames = new List<string>();

		int nVert = 0;
		int nFace = 0;		//number of vertices and faces
		int currentVert = 0;
		bool isMoff = false;

		foreach (string ln in File.ReadAllLines(fn))
		{
			if (ln.Length > 0 && ln [0] != '#') {
				string l = ln.Trim ().Replace ("  ", " ");
				string[] cmps = l.Split (' ');					//Array of words
				
				if (!isMoff) {					//make sure we get a "MOFF" at the beginning. (Dirty solution using a bool, to fit the code structure).
					if (cmps [0] == "MOFF") {	// (Only MOFF files supported yet)
						isMoff = true;
					} else {
						return null;
					}
					continue;
				}

				//TODO: load colormap here
				if (cmps.Length == 1) {
					string data = l.Remove(0, l.IndexOf(' ') + 1);
					//TEXTURE
					Debug.Log(data + ", " + moffFileDirectory + ", " + meshName);
					string fpth = OFFGetFilePath(data, moffFileDirectory,meshName);
					Debug.Log (fpth);
					if (fpth != null) {
						
						AssetDatabase.ImportAsset(fpth, ImportAssetOptions.ForceUncompressedImport);
						
						Debug.Log ("texture dimensions: " + myColorMap.height);

						myColorMap = LoadPNG (fpth);

						//TextureImporter importer = (TextureImporter)TextureImporter.GetAtPath("Assets/Inria/files/000145.png");
						//importer.npotScale = TextureImporterNPOTScale.None;
						//importer.mipmapEnabled = false;
						//importer.wrapMode = TextureWrapMode.Repeat;
						//importer.filterMode = FilterMode.Point;
						//importer.textureCompression = TextureImporterCompression.Uncompressed;
						//importer.alphaSource = TextureImporterAlphaSource.None;



						Debug.Log ("texture dimensions: " + myColorMap.height);

					}
					continue;
				}
				//When we reach this part, the line should theoretically be numerical values

				if (nVert == 0) {		//first numerical line: get number of vertices and faces;
					int.TryParse (cmps [0], out nVert);
					Int32.TryParse (cmps [1], out nFace);
					if (nVert == 0 || nFace == 0) {
						return null;
					}
					continue;
				}
				if (currentVert < nVert) {							//fill vertices and vertices color indices
					vertices.Add (ParseVertexFromCMPS (cmps));
					int curInd;
					Int32.TryParse (cmps [3], out curInd);
					vertColorIndices.Add (curInd);
					currentVert++;
					continue;
				}

				//If this point is reached, all vertices have been written, and we've reached face lines.
				//There should be 9 (?) elements per line: # of vertices, v1, v2, v3, face resolution, e1 color index, e2Ci, e3Ci, face color index
				if (cmps [0] == "3") {
					//face
					IntVector3 newFace = new IntVector3 ();
					newFace.x = Int32.Parse(cmps [1]);
					newFace.y = Int32.Parse(cmps [2]);
					newFace.z = Int32.Parse(cmps [3]);
					faceList.Add (newFace);
					//face res
					faceRes.Add (uint.Parse(cmps [4]));
					//edge color indices
					IntVector3 eCi = new IntVector3 ();
					eCi.x = Int32.Parse(cmps [5]);
					eCi.y = Int32.Parse(cmps [6]);
					eCi.z = Int32.Parse(cmps [7]);
					edgeColorIndices.Add (eCi);
					//face color index
					faceColorIndices.Add (uint.Parse(cmps [8]));
				}
			}
		}
		if (objectNames.Count == 0)
			objectNames.Add("submesh_1");

		int faceNum = faceList.Count;
		int submeshNum = 2;
		while (faceNum > 21844) {			
			objectNames.Add ("submesh_" + submeshNum.ToString());
			faceNum -= 21844;
			submeshNum++;
		}

		//build objects
		GameObject parentObject = new GameObject(meshName);


		Material[] processedMaterials = new Material[1];
		//processedMaterials [0] = (Material)Resources.Load("meshColorMaterial", typeof(Material));
		processedMaterials [0] = new Material(Shader.Find("Unlit/meshColor"));
		//processedMaterials[0] = (Material)AssetDatabase.LoadAssetAtPath("Assets/Resources/meshColorMaterial.mat", typeof(Material));
		AssetDatabase.CreateAsset (processedMaterials [0], "Assets/Resources/testMat.mat");
		AssetDatabase.SaveAssets ();
		AssetDatabase.Refresh ();
		processedMaterials [0] = (Material)AssetDatabase.LoadAssetAtPath ("Assets/Resources/testMat.mat", typeof(Material));

		processedMaterials [0].SetTexture ("_MainTex", myColorMap);
		//mr.materials = processedMaterials;

		long myHeight = myColorMap.height;
		long myWidth = myColorMap.width;
		int i = 0;
		foreach (string obj in objectNames)
		{
			GameObject subObject = new GameObject(obj);
			subObject.transform.parent = parentObject.transform;
			subObject.transform.localScale = new Vector3(-1, 1, 1);
			//Create mesh
			Mesh m = new Mesh();
			m.name = obj;
			//LISTS FOR REORDERING
			List<Vector3> processedVertices = new List<Vector3>();
			List<Vector4> processedTangents = new List<Vector4>();
			List<Vector3> processedNormals = new List<Vector3>();
			List<Vector2> processedUVs = new List<Vector2>();
			List<Vector2> processedUV2s = new List<Vector2>();
			List<Vector2> processedUV3s = new List<Vector2>();
			List<Vector2> processedUV4s = new List<Vector2>();
			//List<Vector4> processedColors = new List<Vector4>();
			
			List<int> processedIndexes = new List<int>();
			//POPULATE MESH
			for (int j = 0; j < 21844; j++) {


				long vCh1, vCh2, vCh3, vCw1, vCw2, vCw3;
				long eCh1, eCh2, eCh3, eCw1, eCw2, eCw3;
				long fCh, fCw;

				//should make a function / replace vector
				vCh1 = (long)((float)(vertColorIndices [faceList [i].x])/myWidth);
				vCw1 = vertColorIndices [faceList [i].x] - myWidth * vCh1;
				vCh2 = (long)((float)(vertColorIndices [faceList [i].y])/myWidth);
				vCw2 = vertColorIndices [faceList [i].y] - myWidth * vCh2;
				vCh3 = (long)((float)(vertColorIndices [faceList [i].z])/myWidth);
				vCw3 = vertColorIndices [faceList [i].z] - myWidth * vCh3;

				eCh1 = (long)((float)(edgeColorIndices [i].x)/myWidth);
				eCw1 = edgeColorIndices [i].x - myWidth * eCh1;
				eCh2 = (long)((float)(edgeColorIndices [i].y)/myWidth);
				eCw2 = edgeColorIndices [i].y - myWidth * eCh2;
				eCh3 = (long)((float)(edgeColorIndices [i].z)/myWidth);
				eCw3 = edgeColorIndices [i].z - myWidth * eCh3;

				fCh = (long)((float)(faceColorIndices [i])/myWidth);
				fCw = faceColorIndices [i] - myWidth * fCh;

				//POSITION
				processedVertices.Add (vertices [faceList [i].x]);
				processedVertices.Add (vertices [faceList [i].y]);
				processedVertices.Add (vertices [faceList [i].z]);

				//UV: vertex number + face color index
				processedUVs.Add (new Vector2 (0 * myHeight + fCh, faceRes [i]));
				processedUVs.Add (new Vector2 (1 * myHeight + fCh, faceRes [i]));
				processedUVs.Add (new Vector2 (2 * myHeight + fCh, faceRes [i]));

				//normals: vertex color indices + face ind (height)
				processedNormals.Add (new Vector4 ((float)(vCh1), (float)(vCh2), (float)(vCh3)));
				processedNormals.Add (new Vector4 ((float)(vCh1), (float)(vCh2), (float)(vCh3)));
				processedNormals.Add (new Vector4 ((float)(vCh1), (float)(vCh2), (float)(vCh3)));

				//tangents: vertex color indices + face ind (width)
				processedTangents.Add (new Vector4 ((float)(vCw1), (float)(vCw2), (float)(vCw3), (float)(fCw)));
				processedTangents.Add (new Vector4 ((float)(vCw1), (float)(vCw2), (float)(vCw3), (float)(fCw)));
				processedTangents.Add (new Vector4 ((float)(vCw1), (float)(vCw2), (float)(vCw3), (float)(fCw)));

				//UV2: Edge color indices (1st two edges). (width)
				processedUV2s.Add (new Vector2 ((float)(eCw1), (float)(eCw2)));
				processedUV2s.Add (new Vector2 ((float)(eCw1), (float)(eCw2)));
				processedUV2s.Add (new Vector2 ((float)(eCw1), (float)(eCw2)));

				//UV4: Edge color indices (1st two edges). (height)
				processedUV4s.Add (new Vector2 ((float)(eCh1), (float)(eCh2)));
				processedUV4s.Add (new Vector2 ((float)(eCh1), (float)(eCh2)));
				processedUV4s.Add (new Vector2 ((float)(eCh1), (float)(eCh2)));				

				//UV3: edge color indices (3)
				processedUV3s.Add (new Vector2 ((float)(eCw3), (float)(eCh3)));
				processedUV3s.Add (new Vector2 ((float)(eCw3), (float)(eCh3)));
				processedUV3s.Add (new Vector2 ((float)(eCw3), (float)(eCh3)));

				//faces
				processedIndexes.Add (3*j);
				processedIndexes.Add (3 * j+1);
				processedIndexes.Add (3 * j+2);
				i++;
				if (i == faceList.Count) {
					break;
				}
			}


			m.vertices = processedVertices.ToArray();
			m.uv = processedUVs.ToArray();
			m.uv2 = processedUV2s.ToArray();
			m.uv3 = processedUV3s.ToArray();
			m.uv4 = processedUV4s.ToArray();
			m.normals = processedNormals.ToArray();
			m.tangents = processedTangents.ToArray();
			m.triangles = processedIndexes.ToArray();

			MeshFilter mf = subObject.AddComponent<MeshFilter>();
			MeshRenderer mr = subObject.AddComponent<MeshRenderer>();

			mr.material = processedMaterials [0];
			mf.mesh = m;

		}

		return parentObject;
	}
}

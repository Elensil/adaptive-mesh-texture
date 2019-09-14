
# Adaptive Mesh Texture for Multi-View Appearance Modeling

## Introduction
This repository provides all necessary code and information to reproduce the work presented in the following paper:
*Adaptive Mesh Texture for Multi-View Appearance Modeling*, 3DV 2019.
[Matthieu Armando](http://morpheo.inrialpes.fr/people/armando/), [Jean-Sébastien Franco](http://morpheo.inrialpes.fr/~franco/), [Edmond Boyer](http://morpheo.inrialpes.fr/people/Boyer/),
[[HAL]](https://hal.inria.fr/hal-02284101)

Please, bear in mind this is a research project, and we are currently working on making its installation and usage as user-friendly as possible. If you have issues, questions, or suggestions, do not hesitate to [contact us](https://gitlab.inria.fr/marmando/adaptive-mesh-texture#contact).
 
### Abstract

> In this paper we report on the representation of appearance information in the context of 3D multi-view shape modeling. Most applications in image based 3D  modeling resort to texture maps, a 2D mapping of shape color information into image files. Despite their unquestionable merits, in particular the ability to apply standard image tools, including compression, image textures  still suffer from limitations that result from the 2D mapping of information that originally belongs to a 3D structure. This is especially true with 2D texture atlases, a generic 2D mapping for 3D mesh models that introduces discontinuities in the texture space and plagues many 3D appearance algorithms. Moreover,  the per-triangle texel density of 2D image textures cannot be individually adjusted to the corresponding pixel observation density without a global change in the atlas mapping function. To address these issues, we propose a new appearance representation for image-based 3D shape modeling, which stores appearance information directly on 3D meshes, rather than a texture atlas. We show this representation to allow for input-adaptive sampling and compression support. Our experiments demonstrate that it outperforms traditional image textures, in multi-view reconstruction contexts, with better visual quality and memory footprint, which makes it a suitable tool when dealing with large amounts of data as with dynamic scene 3D models.


## Structure

- The [MeshTexture](https://gitlab.inria.fr/marmando/adaptive-mesh-texture/tree/master/MeshTexture) folder contains the main C++ project, used to generate, manipulate, compress/uncompress textured meshes.
- The [Eval](https://gitlab.inria.fr/marmando/adaptive-mesh-texture/tree/master/Eval) folder contains python code to compute similarity scores between textured models and input images.
- The [Rendering](https://gitlab.inria.fr/marmando/adaptive-mesh-texture/tree/master/Rendering) folder contains ressources for displaying textured models.


## Citation
If you use this code, please cite the following:
```
@INPROCEEDINGS{armando19,  
  title     = {Adaptive Mesh Texture for Multi-View Appearance Modeling},  
  author    = {Armando, Matthieu and Franco, Jean-S\'{e}bastien and Boyer, Edmond},  
  booktitle = {3DV},  
  year      = {2019}  
}
```

## License
Please check the [license terms](https://gitlab.inria.fr/marmando/adaptive-mesh-texture/blob/master/LICENSE.md) before downloading and/or using the code.


## Contact
[Matthieu Armando](http://morpheo.inrialpes.fr/people/armando/)
 - INRIA Grenoble Rhône-Alpes
 - 655, avenue de l’Europe, Montbonnot
 - 38334 Saint Ismier, France
 - Email: [matthieu.armando@inria.fr](mailto:matthieu.armando@inria.fr)

## Acknowledgements

The texturing code was originally built on top of the [4DCVT](http://deep4dcvtr.gforge.inria.fr/) project. As such, some bits were written by [Vincent Leroy](http://morpheo.inrialpes.fr/people/vleroy/).
The Unity loader was built upon [...]

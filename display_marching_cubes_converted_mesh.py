import sys
import numpy as np
import skimage.measure
from vispy import scene
from vispy.scene.cameras import TurntableCamera
import vispy.io
import vispy.geometry
from vispy.visuals.transforms import STTransform


##
# verts, faces, normals, nothin = vispy.io.read_mesh("mesh.obj")
# mesh = scene.visuals.Mesh(vertices=verts, shading='smooth', faces=faces)

surf = np.load("vessel_tree_noise.npy")
verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(surf, 0)
meshdata = vispy.geometry.MeshData(vertices=verts, faces=faces)

# sanity check
# assert np.allclose(normals, meshdata.get_vertex_normals(), atol=1e-5)

canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)

# Set up a viewbox to display the cube with interactive arcball
view = canvas.central_widget.add_view()
view.bgcolor = '#efefef'
# extent of the axis should fit twice as much as volume's x, y axis
scale_factor = np.sqrt((surf.shape[0] ** 2 + surf.shape[1] ** 2)) * 2.0
view.camera = TurntableCamera(fov=60, name='turntable', scale_factor=scale_factor)

mesh = scene.visuals.Mesh(meshdata=meshdata, shading='smooth', color='w', parent=view.scene)
mesh.set_gl_state('translucent', depth_test=True, cull_face=True)
view.add(mesh)

# Add a 3D axis to keep us oriented
axis = scene.visuals.XYZAxis(parent=view.scene)
s = STTransform(translate=(-100, -100), scale=(50, 50, 50, 1))
affine = s.as_matrix()
axis.transform = affine


if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run()

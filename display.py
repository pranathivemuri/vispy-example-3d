import sys
import numpy as np
import skimage.measure
from vispy import scene
from vispy.color import Color
from vispy.scene.cameras import TurntableCamera
import vispy.io
import vispy.geometry
##
# verts, faces, normals, nothin = vispy.io.read_mesh("mesh.obj")
# mesh = scene.visuals.Mesh(vertices=verts, shading='smooth', faces=faces)

surf = np.load("/Users/pranathivemuri/Downloads/npys/vessel_tree_noise.npy")
verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(surf, 0)
meshdata = vispy.geometry.MeshData(vertices=verts, faces=faces)

# sanity check
# assert np.allclose(normals, meshdata.get_vertex_normals(), atol=1e-5)


canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)

# Set up a viewbox to display the cube with interactive arcball
view = canvas.central_widget.add_view()
view.bgcolor = '#efefef'
# view.camera = 'turntable'
view.camera = TurntableCamera(fov=45)
# view.padding = 100
# view.camera.depth_value = 10
# print view.camera.depth_value

color = Color("#3f51b5")

mesh = scene.visuals.Mesh(meshdata=meshdata, shading='smooth', color='w')
mesh.set_gl_state('translucent', depth_test=True, cull_face=True)
view.add(mesh)

# Add a 3D axis to keep us oriented
axis = scene.visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run()

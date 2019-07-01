"""
Example volume rendering

python3 display_numpy_volume.py --vol1_path=vessel_tree_noise.npy
python3 display_numpy_volume.py --vol1_path=stent.npz --vol2_path=mri.npz
npz arrays are expected to have numpy arrays at "data" key

Example npz files are grayscale numpy stacks/volumes
Need to test stacks of color images

Controls:

* 1  - toggle camera between first person (fly), regular 3D (turntable) and
       arcball
* 2  - toggle between volume rendering methods
* 3  - toggle between 2 different volumes if present
* 4  - toggle between colormaps
* 0  - reset cameras
* [] - decrease/increase isosurface threshold

With fly camera:

* WASD or arrow keys - move around
* SPACE - brake
* FC - move up-down
* IJKL or mouse - look around
"""
import argparse

from itertools import cycle

import numpy as np
import skimage.measure

import vispy
from vispy import app, scene
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """


# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransFire(), TransGrays()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)


def load_numpy_array(path):
    if path.endswith(".npy"):
        loaded_array = np.load(path)
    elif path.endswith(".npz"):
        with np.load(path)as array:
            loaded_array = array["data"]
    else:
        print("Not a valid path")
        raise AssertionError
    if loaded_array.dtype == np.bool:
        loaded_array = loaded_array.astype(np.uint8) * 255
    return loaded_array


def display_as_mesh(path, view):
    array = load_numpy_array(path)
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        array, 0)

    meshdata = vispy.geometry.MeshData(vertices=verts, faces=faces)

    # extent of the axis should fit twice as much as volume's x, y axis
    if view.camera.name == "Turntable":
        view.camera.scale_factor = np.sqrt(
            (array.shape[0] ** 2 + array.shape[1] ** 2)) * 2.0

    mesh = scene.visuals.Mesh(
        meshdata=meshdata, shading='smooth', color='w', parent=view.scene)
    mesh.set_gl_state('translucent', depth_test=True, cull_face=True)
    view.add(mesh)
    return view


class VispyDisplayVolumeTest():

    def __init__(self, vol1_path, vol2_path):

        vol1_path = vol1_path
        vol2_path = vol2_path

        vol1 = load_numpy_array(vol1_path)
        if vol2_path is not None:
            vol2 = load_numpy_array(vol2_path)

        canvas = scene.SceneCanvas(
            keys='interactive', size=(800, 600), show=True)

        # Set up a viewbox to display the image with interactive pan/zoom
        view = canvas.central_widget.add_view()

        # Set whether we are emulating a 3D texture
        emulate_texture = False

        # Create the volume visuals, only one is visible
        volume1 = scene.visuals.Volume(
            vol1, parent=view.scene, threshold=0.225,
            emulate_texture=emulate_texture)
        volume1.transform = scene.STTransform(translate=(64, 64, 0))
        if vol2_path is not None:
            volume2 = scene.visuals.Volume(
                vol2, parent=view.scene, threshold=0.2,
                emulate_texture=emulate_texture)
            volume2.visible = False

        # Create three cameras (Fly, Turntable and Arcball)
        fov = 60.
        cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov, name='Fly')
        cam2 = scene.cameras.TurntableCamera(
            parent=view.scene, fov=fov,
            name='Turntable')
        cam3 = scene.cameras.ArcballCamera(
            parent=view.scene, fov=fov,
            name='Arcball')
        view.camera = cam2  # Select turntable at first

        # Create an XYZaxis visual
        axis = scene.visuals.XYZAxis(parent=view)
        s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
        affine = s.as_matrix()
        axis.transform = affine

        # Implement axis connection with cam2
        @canvas.events.mouse_move.connect
        def on_mouse_move(event):
            if event.button == 1 and event.is_dragging:
                axis.transform.reset()

                axis.transform.rotate(cam2.roll, (0, 0, 1))
                axis.transform.rotate(cam2.elevation, (1, 0, 0))
                axis.transform.rotate(cam2.azimuth, (0, 1, 0))

                axis.transform.scale((50, 50, 0.001))
                axis.transform.translate((50., 50.))
                axis.update()

        # Implement key presses
        @canvas.events.key_press.connect
        def on_key_press(event):
            global opaque_cmap, translucent_cmap
            if event.text == '1':
                cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
                view.camera = cam_toggle.get(view.camera, cam2)
                print(view.camera.name + ' camera')
                if view.camera is cam2:
                    axis.visible = True
                else:
                    axis.visible = False
            elif event.text == '2':
                methods = ['mip', 'translucent', 'iso', 'additive']
                method = methods[(methods.index(volume1.method) + 1) % 4]
                print("Volume render method: %s" % method)
                cmap = \
                    opaque_cmap if method in ['mip', 'iso'] else \
                    translucent_cmap
                volume1.method = method
                volume1.cmap = cmap
                if vol2_path is not None:
                    volume2.method = method
                    volume2.cmap = cmap
            elif event.text == '3':
                volume1.visible = not volume1.visible
                volume2.visible = not volume1.visible
            elif event.text == '4':
                if volume1.method in ['mip', 'iso']:
                    cmap = opaque_cmap = next(opaque_cmaps)
                else:
                    cmap = translucent_cmap = next(translucent_cmaps)
                volume1.cmap = cmap
                if vol2_path is not None:
                    volume2.cmap = cmap
            elif event.text == '0':
                cam1.set_range()
                cam3.set_range()
            elif event.text != '' and event.text in '[]':
                s = -0.025 if event.text == '[' else 0.025
                volume1.threshold += s
                if vol2_path is not None:
                    volume2.threshold += s
                if volume1.visible:
                    th = volume1.threshold
                if vol2_path is not None:
                    th = volume2.threshold
                print("Isosurface threshold: %0.3f" % th)

    def run(self):
        app.run()


def main():
    parser = argparse.ArgumentParser(
        description="Display 3D volume npy or npz arrays using vispy")
    parser.add_argument(
        "--vol1_path",
        help="path to 1st npy or npz file",
        type=str)
    parser.add_argument(
        "--vol2_path",
        help="path to 2nd npy or npz file",
        type=str,
        default=None)
    args = parser.parse_args()

    vol1_path = args.vol1_path
    vol2_path = args.vol2_path

    vispyVolumeDisplay = VispyDisplayVolumeTest(vol1_path, vol2_path)
    vispyVolumeDisplay.run()


if __name__ == '__main__':
    main()

import trimesh
from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import Meshes, VariableTopologyMeshes
import glob
import argparse

def vis_dynamic(args):
    vertices = []
    faces = []
    vertex_normals = []
    deformed_mesh_paths = sorted(glob.glob(f'{args.path}/*_deformed.ply'))
    for deformed_mesh_path in deformed_mesh_paths:
        mesh = trimesh.load(deformed_mesh_path, process=False)
        # center the human
        mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        vertices.append(mesh.vertices)
        faces.append(mesh.faces)
        vertex_normals.append(mesh.vertex_normals)

    meshes = VariableTopologyMeshes(vertices,
                                    faces,
                                    vertex_normals,
                                    preload=True 
                                    )

    meshes.norm_coloring = True
    meshes.flat_shading = True
    viewer = Viewer()
    viewer.scene.add(meshes)
    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = True
    viewer.run()
def vis_static(args):
    mesh = trimesh.load(args.path, process=False)
    mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='mesh', flat_shading=True)
    mesh.norm_coloring = True
    viewer = Viewer()
    viewer.scene.add(mesh)
    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = True
    viewer.run()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Visualization')
    # static canonical mesh or dynamic sequence
    parser.add_argument('--mode', type=str, help='mode: static or dynamic')
    # mesh/meshes source
    parser.add_argument('--path', type=str, help='path to the file')
    args = parser.parse_args()
    if args.mode == 'static':
        vis_static(args)
    elif args.mode == 'dynamic':
        vis_dynamic(args)
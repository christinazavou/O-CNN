import os
import numpy as np


def read_obj(obj_fn):
    """
    Read obj
    :param obj_fn: str
    :return:
    vertices: N x 3, numpy.ndarray(float)
    faces: M x 3, numpy.ndarray(int)
    components: M x 1, numpy.ndarray(int) (pointing to parenting component of each face i.e. components[face1]=ComponentId)
    """

    assert (os.path.isfile(obj_fn))

    # Return variables
    vertices, faces, components = [], [], []

    with open(obj_fn, 'r') as f_obj:
        component = -1
        # Read obj geometry
        for line in f_obj:
            line = line.strip().split(' ')
            if line[0] == 'v':
                # Vertex row
                assert (len(line) == 4)
                vertex = [float(line[1]), float(line[2]), float(line[3])]
                vertices.append(vertex)
            if line[0] == 'o':
                # object row
                assert (len(line) == 2)
                component = int(line[1])
            if line[0] == 'f':
                # Face row
                face = [float(line[1].split('/')[0]), float(line[2].split('/')[0]), float(line[3].split('/')[0])]
                faces.append(face)
                components.append(component)

    vertices = np.vstack(vertices)
    faces = np.vstack(faces)
    components = np.vstack(components)

    return vertices, faces.astype(np.int32), components.astype(np.int32)


def read_ply(ply_fn):
    """
		Read ply file
	:param ply_fn: str
	:return:
		vertices: N x 3, numpy.ndarray(float)
    faces: M x 3, numpy.ndarray(int)
	"""

    vertices, faces, n_vertices, n_faces = [], [], 0, 0
    header_end = False

    with open(ply_fn, 'r') as fin_ply:
        # Read header
        line = fin_ply.readline().strip()
        assert (line == "ply")
        for line in fin_ply:
            line = line.strip().split(' ')
            if line[0] == "end_header":
                # Header end
                header_end = True
                break
            if (line[0] == "element") and (line[1] == "vertex"):
                n_vertices = int(line[2])
            if (line[0] == "element") and (line[1] == "face"):
                n_faces = int(line[2])
        assert (header_end)

        # Read vertices
        for _ in range(n_vertices):
            line = fin_ply.readline().strip().split(' ')
            assert (len(line) >= 3)
            vertex = [float(line[0]), float(line[1]), float(line[2])]
            vertices.append(vertex)

        # Read faces
        for _ in range(n_faces):
            line = fin_ply.readline().strip().split(' ')
            assert (len(line) >= 4)
            n_face = int(line[0])
            face = []
            for line_idx in range(1, n_face):
                face.append(float(line[line_idx]))
            faces.append(face)

    if len(vertices):
        vertices = np.vstack(vertices)
    if len(faces):
        faces = np.vstack(faces)

    return vertices, faces


def write_ply(ply_fn, vertices, faces, face_color):
    """
		Write shape in .ply with face color information
	:param ply_fn: str
	:param vertices: N x 3, numpy.ndarray(float)
	:param faces: M x 3, numpy.ndarray(int)
	:param face_color: M x 1, numpy.ndarray(float)
	:return:
		None
	"""

    # Create header
    header = 'ply\n' \
             'format ascii 1.0\n' \
             'element vertex ' + str(len(vertices)) + '\n' \
                                                      'property float x\n' \
                                                      'property float y\n' \
                                                      'property float z\n' \
                                                      'element face ' + str(len(faces)) + '\n' \
                                                                                          'property list uchar int vertex_indices\n' \
                                                                                          'property float red\n' \
                                                                                          'property float green\n' \
                                                                                          'property float blue\n' \
                                                                                          'end_header\n'

    if np.min(faces) == 1:
        faces -= 1

    with open(ply_fn, 'w') as f_ply:
        # Write header
        f_ply.write(header)

        # Write vertices
        for vertex in vertices:
            row = ' '.join([str(vertex[0]), str(vertex[1]), str(vertex[2])]) + '\n'
            f_ply.write(row)
        # Write faces + face_color
        for face_ind, face in enumerate(faces):
            color = face_color[face_ind]
            row = ' '.join([str(len(face)), str(face[0]), str(face[1]), str(face[2]),
                            str(color[0]), str(color[1]), str(color[2])]) + '\n'
            f_ply.write(row)


def calculate_face_area(vertices, faces):
    """
    Calculate face area of a triangular mesh
    :param vertices: N x 3, numpy.ndarray(float)
    :param faces: M x 3, numpy.ndarray(int)
    :return:
    face_area: M x 1, numpy.ndarray(float)
    """

    # Get vertices of faces
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    C = vertices[faces[:, 2]]

    # Create face edges
    e1 = B - A
    e2 = C - A

    # Calculate cross product and find length
    cross_prod = np.cross(e1, e2)
    cross_prod_len = np.sqrt(np.sum(cross_prod ** 2, axis=1))

    # Get face area
    face_area = cross_prod_len / 2.0

    return face_area[:, np.newaxis]

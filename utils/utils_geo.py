import numpy as np


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def find_new_position_1vertex(vertices, point, new_vertex):
    """
    Find the new position of a point inside a triangle if one of the vertices of the triangle is moved to a new position.
    """
    # Find the index of the vertex that was moved
    vertex_index = vertices.index(new_vertex)

    # Find the angle between the original and new positions of the moved vertex
    angle = np.atan2(new_vertex[1] - vertices[vertex_index][1], new_vertex[0] - vertices[vertex_index][0])

    # Rotate the triangle and point so that the moved vertex is at the origin
    rotated_vertices = [(vertex[0] - new_vertex[0], vertex[1] - new_vertex[1]) for vertex in vertices]
    rotated_point = (point[0] - new_vertex[0], point[1] - new_vertex[1])

    # Rotate the triangle and point back to their original positions
    new_rotated_vertices = [rotate((0, 0), vertex, -angle) for vertex in rotated_vertices]
    new_rotated_point = rotate((0, 0), rotated_point, -angle)

    # Translate the triangle and point back to their original positions
    new_vertices = [(vertex[0] + new_vertex[0], vertex[1] + new_vertex[1]) for vertex in new_rotated_vertices]
    new_point = (new_rotated_point[0] + new_vertex[0], new_rotated_point[1] + new_vertex[1])

    return new_point


def find_new_position_3points(old_vertices, new_vertices, point):
    # Compute the barycentric coordinates of the point with respect to the old triangle
    v0, v1, v2 = old_vertices

    u = ((v1[1] - v2[1]) * (point[0] - v2[0]) + (v2[0] - v1[0]) * (point[1] - v2[1])) / (
            (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
    v = ((v2[1] - v0[1]) * (point[0] - v2[0]) + (v0[0] - v2[0]) * (point[1] - v2[1])) / (
            (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
    w = 1 - u - v

    # Compute the new position of the point with respect to the new triangle
    new_v0, new_v1, new_v2 = new_vertices
    new_point = u * new_v0 + v * new_v1 + w * new_v2
    if np.min(new_vertices, axis=0)[0] < new_point[0] < np.max(new_vertices, axis=0)[0] and \
            np.min(new_vertices, axis=0)[1] < new_point[1] < np.max(new_vertices, axis=0)[1]:
        pass
    else:
        pass#print(f"{old_vertices} {new_vertices} {point} {new_point}")
    return new_point


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    old_p = np.array([[-38.1, 114.3],
                      [-63.5, 114.3],
                      [-12.7, 114.3]], dtype=float)
    new_p = np.array([[-0.6486, 1.74915],
                      [-0.84525, 1.75605],
                      [-0.47265, 1.7112]], dtype=float)
    p = [-46.8, 138.0]
    q = find_new_position_3points(old_p, new_p, p)

    plt.scatter(old_p[:, 0], old_p[:, 1], alpha=0.5, marker='x', c='r')
    plt.scatter(new_p[:, 0], new_p[:, 1], alpha=0.5, marker='.', c='b')
    plt.scatter(p[0], p[1], alpha=0.5, marker='x', c='orange')
    plt.scatter(q[0], q[1], alpha=0.5, marker='.', c='purple')
    plt.show()

    plt.subplot(121)
    plt.scatter(old_p[:, 0], old_p[:, 1], alpha=0.5, marker='x', c='r')
    plt.scatter(p[0], p[1], alpha=0.5, marker='x', c='orange')
    plt.subplot(122)
    plt.scatter(new_p[:, 0], new_p[:, 1], alpha=0.5, marker='.', c='b')
    plt.scatter(q[0], q[1], alpha=0.5, marker='.', c='purple')
    plt.show()

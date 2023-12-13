from arguments import parser
from data_prepare import DataPrepare

args = parser.parse_args()

prepare = DataPrepare(args, ['1A0G_A_B'])
prepare()

# def collapse_short_edges(mesh, length_threshold):
#     """
#     Collapses edges shorter than the given threshold in the mesh.

#     Args:
#         mesh (trimesh.Trimesh): The input mesh.
#         length_threshold (float): The maximum length of edges to collapse.

#     Returns:
#         trimesh.Trimesh: The mesh after edge collapsing.
#     """

#     # Ensure the mesh is an instance of trimesh.Trimesh
#     if not isinstance(mesh, trimesh.Trimesh):
#         raise TypeError("mesh must be a trimesh.Trimesh object")

#     # Get a copy of the mesh to modify
#     modified_mesh = mesh.copy()

#     # Keep collapsing edges until no short edges are left
#     edges_collapsed = True
#     while edges_collapsed:
#         edges_collapsed = False

#         # Get edge lengths
#         edge_lengths = modified_mesh.edges_unique_length

#         # Find all short edges
#         short_edges = modified_mesh.edges_unique[edge_lengths < length_threshold]

#         for edge in short_edges:
#             # Get the vertices of the edge
#             v1, v2 = edge

#             # Collapse the edge by merging the two vertices
#             # Replace all occurrences of v2 with v1
#             modified_mesh.faces[np.where(modified_mesh.faces == v2)] = v1

#             # Update the mesh
#             modified_mesh.update_faces(modified_mesh.faces)
#             modified_mesh.remove_unreferenced_vertices()
            
#             edges_collapsed = True
#             break  # Break to recompute the edge lengths

#     return modified_mesh
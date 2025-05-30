--[[ @doc
# 2D Orthogonal Grid

Here, we will use the in-house orthogonal mesh generator for a simple Cartesian grid.

## List of nodes
We first create a lua table for the list of nodes. The nodes will be spread from -L/2 to +L/2.
Be mindful that lua indexing starts at 1.
--]]
-- Setup the mesh
nodes = {}
n_cells = 10
length = 2.
xmin = -length / 2.
dx = length / n_cells
for i = 1, (n_cells + 1) do
  k = i - 1
  nodes[i] = xmin + k * dx
end

--[[ @doc
## Mesh and KBA partition

We use the `OrthogonalMeshGenerator` and pass the list of nodes per dimension. Here, we pass 2 times the same
list of nodes to create a 2D geometry with square cells. Thus, we create a square domain, of side length L,
centered on the origin (0,0).

We also introduce partitioning. The following schemes are available
- KBA partitioning for regular grids that can be cut into right parallelpipeds
- Parmetis partitioning and Scotch partitioning, applicable with any grid type.


Here, we first partition the 2D mesh into 2x2 subdomains using `KBAGraphPartitioner`. We place one cut along the x-axis
at x=0 by filling the xcuts array. Likewise for ycuts (y=0). The cell assignment to a partition is done based
on where the cell center is located with respect to the various xcuts, ycuts, and zcuts (a fuzzy logic is applied
to avoid issues).
--]]
meshgen = mesh.OrthogonalMeshGenerator.Create({
  node_sets = { nodes, nodes },
  partitioner = mesh.KBAGraphPartitioner.Create({
    nx = 2,
    ny = 2,
    xcuts = { 0. },
    ycuts = { 0. },
  }),
})
grid = meshgen:Execute()

--[[ @doc
## Material IDs
When using the in-house `OrthogonalMeshGenerator`, no material IDs are assigned. The user needs to
assign material IDs to all cells. Here, we have a homogeneous domain, so we assign a material ID
with value 0 for each cell in the spatial domain.
--]]
-- Set block IDs
grid:SetUniformBlockID(0)

--[[ @doc
## Export the mesh
We export to vtu format. The resulting mesh partition is shown below
![below](images/ortho_2D_KBA.png)
--]]
-- Exporting the mesh
mesh.ExportToPVTU(grid, "ortho_2D_KBA")

--[[ @doc
## Mesh (again) and Parmetis partition
Now, we use the Parmetis partitioner.
--]]
meshgen = mesh.OrthogonalMeshGenerator.Create({
  node_sets = { nodes, nodes },
  partitioner = mesh.PETScGraphPartitioner.Create({ type = "parmetis" }),
})
meshgen:Execute()

--[[ @doc
## Export the mesh
On such a simple regular mesh, both partitioners are giving the same result. The Parmetis partition is shown below
![below](images/ortho_2D_Parmetis.png)
--]]
-- Exporting the mesh
mesh.ExportToPVTU(grid, "ortho_2D_Parmetis")

meshgen = FromFileMeshGenerator(
    filename="../../../../../assets/mesh/GMSH_AllHexes.vtu",
)
grid = meshgen.Execute()

math_SDM_Test02_Discontinuous(grid, "PWLD", False)

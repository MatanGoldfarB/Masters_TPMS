Merge "./simulation_tests/sphere_changekPw5.obj";

// Wrap the mesh into a volume
Surface Loop(1) = {1};
Volume(1) = {1};

// Mesh size control (adjust for your model)
Mesh.CharacteristicLengthMin = 0.5;
Mesh.CharacteristicLengthMax = 2.0;

// Generate tetrahedral mesh
Mesh 3;

// Save output
Save "./tetrahedral/sphere_changekPw5.msh";
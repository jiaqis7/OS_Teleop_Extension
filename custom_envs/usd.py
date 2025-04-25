usd_content = """#usda 1.0
(
    defaultPrim = "Cube"
)

def Xform "Cube"
{
    matrix4d xformOp:transform = (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0.5),  # Cube center at Z=0.5
        (0, 0, 0, 1)
    )
    uniform token[] xformOpOrder = ["xformOp:transform"]

    def Cube "Geom"
    {
        float3 size = (0.05, 0.05, 0.05)  # 5 cm cube
        rel material:binding = </Looks/Gray>
        prepend apiSchemas = ["RigidBodyAPI", "CollisionAPI"]
        physics:rigidBodyEnabled = true
        physics:mass = 0.1
        physics:collisionEnabled = "convexHull"
        physics:approximation = "box"
    }
}

def Scope "Looks"
{
    def Material "Gray"
    {
        token outputs:surface.connect = </Looks/Gray/PreviewSurface.outputs:surface>

        def Shader "PreviewSurface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.3, 0.3, 0.3)
            float inputs:roughness = 0.5
            token outputs:surface
        }
    }
}
"""

with open("/home/stanford/OS_Teleop_Extension/minimal_cube(2).usda", "w") as f:
    f.write(usd_content)

print("✅ Written minimal_cube.usda successfully.")

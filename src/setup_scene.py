import bpy
import sys
import os

# Usage:
# blender.exe --python setup_scene.py -- "<terrain.obj>" "<template.blend>" [out_blend] [out_jpg]
#find and open template, assign snow grey look, few more helpers
def args_after_double_dash():
    if "--" not in sys.argv:
        raise RuntimeError("Expected args after --")
    return sys.argv[sys.argv.index("--") + 1:]

def open_template(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise RuntimeError(f"Template not found: {path}")
    bpy.ops.wm.open_mainfile(filepath=path)

def _import_override():
    wm = bpy.context.window_manager
    if not wm.windows:
        return None 

    win = wm.windows[0]
    screen = win.screen
    area = next((a for a in screen.areas if a.type == "VIEW_3D"), None)
    if area is None and screen.areas:
        area = screen.areas[0]

    region = None
    if area:
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region is None and area.regions:
            region = area.regions[0]

    ov = {"window": win, "screen": screen}
    if area: ov["area"] = area
    if region: ov["region"] = region
    return ov


def import_obj(obj_path):
    obj_path = os.path.abspath(obj_path)
    if not os.path.exists(obj_path):
        raise RuntimeError(f"OBJ not found: {obj_path}")

    before = set(o.name for o in bpy.context.scene.objects if o.type == "MESH")

    ov = _import_override()
    if ov is None:
        raise RuntimeError("No UI window found. Run Blender without -b (GUI mode).")

    try:
        with bpy.context.temp_override(**ov):
            bpy.ops.wm.obj_import(filepath=obj_path)
    except Exception as e:
        raise RuntimeError(f"OBJ import failed (context): {e}")

    imported = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name not in before]
    if not imported:
        imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not imported:
        raise RuntimeError("OBJ import ran but no new mesh objects detected.")
    return imported

def make_terrain_material():
    mat = bpy.data.materials.new("TerrainGrey") #give grey snow mountain look
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.55, 0.56, 0.58, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

#assign grey snow terrain
def assign_material(objs, mat):
    for o in objs:
        if o.type != "MESH": 
            continue
        if not o.data.materials:
            o.data.materials.append(mat)
        else:
            o.data.materials[0] = mat

def smooth_shade(objs):
    for o in objs:
        if o.type != "MESH":
            continue
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        try:
            bpy.ops.object.shade_smooth()
        except:
            pass
        o.select_set(False)

def render_jpg(path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    scene = bpy.context.scene
    scene.render.filepath = path
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 95
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100

    bpy.ops.render.render(write_still=True)
    print("Saved JPG:", path)

def main():
    args = args_after_double_dash()
    if len(args) < 2:
        raise RuntimeError('Need: "<terrain.obj>" "<template.blend>" [out_blend] [out_jpg]')

    obj_path = args[0]
    template = args[1]
    out_blend = args[2] if len(args) >= 3 else None
    out_jpg   = args[3] if len(args) >= 4 else None

    open_template(template)

    terrain = import_obj(obj_path)

    # force a clean grey material
    mat = make_terrain_material()
    assign_material(terrain, mat)

    smooth_shade(terrain)

    if out_blend:
        out_blend = os.path.abspath(out_blend)
        os.makedirs(os.path.dirname(out_blend), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        print("Saved blend:", out_blend)

    if out_jpg:
        render_jpg(out_jpg)

    print("Done.")

if __name__ == "__main__":
    main()

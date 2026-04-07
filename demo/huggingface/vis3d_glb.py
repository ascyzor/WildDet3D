"""Generate GLB scenes with colored point clouds / textured meshes and 3D boxes.

Uses pygltflib for point cloud + wireframe boxes (GL_POINTS/GL_LINES),
and trimesh + utils3d for textured mesh generation.

Usage:
    from vis3d_glb import depth_to_pointcloud, create_scene_glb
    from vis3d_glb import create_mesh_scene_glb

    # Point cloud mode
    points, colors = depth_to_pointcloud(depth_map, image, intrinsics)
    create_scene_glb(points, colors, boxes3d_list, output_path)

    # Textured mesh mode (like MoGe2)
    create_mesh_scene_glb(depth_map, image, intrinsics, boxes3d_list, output_path)
"""

import numpy as np
import pygltflib


def depth_to_pointcloud(
    depth_map: np.ndarray,
    image: np.ndarray,
    intrinsics: np.ndarray,
    max_depth: float = 20.0,
    subsample: int = 4,
    padding: tuple[int, int, int, int] | None = None,
    remove_edge: bool = True,
    edge_rtol: float = 0.04,
    confidence_map: np.ndarray | None = None,
    confidence_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth map + RGB image to colored point cloud.

    Args:
        depth_map: (H, W) or (1, H, W) depth in meters.
        image: (H, W, 3) RGB image, uint8 [0-255].
        intrinsics: (3, 3) camera intrinsics matrix.
        max_depth: Discard points beyond this depth.
        subsample: Take every Nth pixel to reduce point count.
        padding: (left, right, top, bottom) CenterPad offsets to exclude.
        remove_edge: Remove points at depth discontinuity edges
            (like MoGe2). Uses utils3d.np.depth_map_edge.
        edge_rtol: Relative tolerance for edge detection. Larger
            values remove more aggressive edges.
        confidence_map: (H, W) or (1, H, W) per-pixel confidence in
            [0, 1]. Points below confidence_threshold are discarded.
        confidence_threshold: Minimum confidence to keep a point.

    Returns:
        points: (N, 3) float32 xyz in camera frame.
        colors: (N, 4) uint8 RGBA.
    """
    # Handle various depth_map shapes
    while depth_map.ndim > 2:
        depth_map = depth_map.squeeze(0)  # (1, 1, H, W) -> (H, W)

    H, W = depth_map.shape

    # Handle confidence_map shape
    if confidence_map is not None:
        while confidence_map.ndim > 2:
            confidence_map = confidence_map.squeeze(0)

    # Handle various image shapes: (1, H, W, 3), (1, 1, H, W) etc
    while image.ndim > 3:
        image = image.squeeze(0)
    # If image is (3, H, W), transpose to (H, W, 3)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    # If grayscale (H, W), repeat to (H, W, 3)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Match image size to depth map
    if image.shape[0] != H or image.shape[1] != W:
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(image)
        img_pil = img_pil.resize((W, H), PILImage.BILINEAR)
        image = np.array(img_pil)

    # Build full-resolution valid mask before subsampling
    full_valid = (depth_map > 0.01) & (depth_map < max_depth) & np.isfinite(depth_map)

    # Exclude padding regions (full resolution)
    if padding is not None:
        pad_left, pad_right, pad_top, pad_bottom = padding
        pad_mask = np.ones((H, W), dtype=bool)
        pad_mask[:, :pad_left] = False
        pad_mask[:pad_top, :] = False
        if pad_right > 0:
            pad_mask[:, W - pad_right:] = False
        if pad_bottom > 0:
            pad_mask[H - pad_bottom:, :] = False
        full_valid &= pad_mask

    # Remove depth discontinuity edges (MoGe2 style)
    if remove_edge:
        import utils3d
        edge_mask = utils3d.np.depth_map_edge(depth_map, rtol=edge_rtol)
        full_valid &= ~edge_mask

    # Filter by confidence
    if confidence_map is not None and confidence_threshold > 0:
        full_valid &= (confidence_map >= confidence_threshold)

    # Subsample grid
    ys = np.arange(0, H, subsample)
    xs = np.arange(0, W, subsample)
    xx, yy = np.meshgrid(xs, ys)

    depth_sub = depth_map[yy, xx]
    rgb_sub = image[yy, xx]  # (h, w, 3)
    valid = full_valid[yy, xx]

    # Unproject to 3D
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x3d = (xx[valid] - cx) * depth_sub[valid] / fx
    y3d = (yy[valid] - cy) * depth_sub[valid] / fy
    z3d = depth_sub[valid]

    # OpenCV (x-right, y-down, z-away) to glTF (x, -y, -z)
    points = np.stack([x3d, -y3d, -z3d], axis=-1).astype(np.float32)

    # Colors
    rgb = rgb_sub[valid]  # (N, 3) uint8
    alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
    colors = np.concatenate([rgb, alpha], axis=-1)  # (N, 4)

    return points, colors


def _quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)


def boxes3d_to_corners(boxes3d: np.ndarray) -> list[np.ndarray]:
    """Convert 3D box params to 8 corner points in GLB coords.

    Args:
        boxes3d: (N, 10) boxes in OpenCV camera frame.
            Format: [cx, cy, cz, w, h, l, qw, qx, qy, qz]

    Returns:
        List of (8, 3) corner arrays in GLB/Three.js coords (y-up, z-backward).
    """
    corners_list = []
    # Same transform as point cloud:
    # OpenCV (x,y,z) -> glTF (x, -y, -z)
    T = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    for box in boxes3d:
        cx, cy, cz = box[0], box[1], box[2]
        # Omni3D format: [width, length, height] not [w, h, l]
        # width = x-extent, length = z-extent, height = y-extent
        bw, bl, bh = box[3], box[4], box[5]
        qw, qx, qy, qz = box[6], box[7], box[8], box[9]

        hw, hl, hh = bw / 2, bl / 2, bh / 2

        # 8 local corners: x=length, y=height, z=width
        local_corners = np.array([
            [-hl, -hh, -hw],
            [ hl, -hh, -hw],
            [ hl,  hh, -hw],
            [-hl,  hh, -hw],
            [-hl, -hh,  hw],
            [ hl, -hh,  hw],
            [ hl,  hh,  hw],
            [-hl,  hh,  hw],
        ], dtype=np.float32)

        # Rotate by quaternion and translate (in OpenCV coords)
        R_cv = _quaternion_to_rotation_matrix(qw, qx, qy, qz)
        corners_cv = (R_cv @ local_corners.T).T + np.array([cx, cy, cz])

        # Convert OpenCV -> glTF: (-z, -y, x)
        corners = (T @ corners_cv.T).T

        corners_list.append(corners.astype(np.float32))

    return corners_list


def _generate_box_colors(n_boxes: int) -> list[list[int]]:
    """Generate distinct colors for boxes."""
    base_colors = [
        [255, 0, 0, 255],    # red
        [0, 255, 0, 255],    # green
        [0, 100, 255, 255],  # blue
        [255, 255, 0, 255],  # yellow
        [255, 0, 255, 255],  # magenta
        [0, 255, 255, 255],  # cyan
        [255, 128, 0, 255],  # orange
        [128, 0, 255, 255],  # purple
    ]
    colors = []
    for i in range(n_boxes):
        colors.append(base_colors[i % len(base_colors)])
    return colors


def _pad_to_4(data: bytes) -> bytes:
    """Pad binary data to 4-byte alignment (glTF requirement)."""
    remainder = len(data) % 4
    if remainder:
        data += b"\x00" * (4 - remainder)
    return data


def create_scene_glb(
    points: np.ndarray,
    point_colors: np.ndarray,
    boxes3d_list: list[np.ndarray],
    output_path: str,
    max_points: int = 500000,
) -> str:
    """Create a GLB file with colored point cloud + wireframe 3D boxes.

    Args:
        points: (N, 3) float32 point cloud xyz.
        point_colors: (N, 4) uint8 RGBA colors.
        boxes3d_list: List of (M, 10) box arrays (one per image).
        output_path: Where to save the .glb file.
        max_points: Max number of points to include.

    Returns:
        output_path.
    """
    # Subsample points if too many
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        point_colors = point_colors[idx]

    points = np.ascontiguousarray(points, dtype=np.float32)
    point_colors = np.ascontiguousarray(point_colors, dtype=np.uint8)
    n_points = len(points)

    # Build box geometry
    all_corners_list = []
    for boxes3d in boxes3d_list:
        if len(boxes3d) > 0:
            corners = boxes3d_to_corners(boxes3d)
            all_corners_list.extend(corners)

    n_boxes = len(all_corners_list)
    box_colors_rgba = _generate_box_colors(n_boxes)

    # Box vertices and indices
    all_box_verts = []
    all_box_colors = []
    all_box_indices = []
    vertex_offset = 0

    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    for i, corners in enumerate(all_corners_list):
        all_box_verts.append(corners)
        color = box_colors_rgba[i]
        all_box_colors.append(
            np.tile(np.array(color, dtype=np.uint8), (8, 1))
        )
        indices = np.array(
            [(a + vertex_offset, b + vertex_offset) for a, b in edge_pairs],
            dtype=np.uint16,
        )
        all_box_indices.append(indices)
        vertex_offset += 8

    has_boxes = n_boxes > 0

    if has_boxes:
        box_verts = np.concatenate(all_box_verts, axis=0).astype(np.float32)
        box_vert_colors = np.concatenate(all_box_colors, axis=0).astype(np.uint8)
        box_indices = np.concatenate(all_box_indices, axis=0).flatten().astype(np.uint16)
    else:
        box_verts = np.zeros((0, 3), dtype=np.float32)
        box_vert_colors = np.zeros((0, 4), dtype=np.uint8)
        box_indices = np.zeros(0, dtype=np.uint16)

    # Build binary blob
    points_bin = _pad_to_4(points.tobytes())
    colors_bin = _pad_to_4(point_colors.tobytes())
    box_verts_bin = _pad_to_4(box_verts.tobytes())
    box_colors_bin = _pad_to_4(box_vert_colors.tobytes())
    box_indices_bin = _pad_to_4(box_indices.tobytes())

    blob = points_bin + colors_bin + box_verts_bin + box_colors_bin + box_indices_bin

    # Build glTF structure
    buffer_views = []
    accessors = []
    offset = 0

    # BV0: point positions
    buffer_views.append(pygltflib.BufferView(
        buffer=0, byteOffset=offset, byteLength=len(points_bin),
        target=pygltflib.ARRAY_BUFFER,
    ))
    accessors.append(pygltflib.Accessor(
        bufferView=0, componentType=pygltflib.FLOAT,
        count=n_points, type=pygltflib.VEC3,
        max=points.max(axis=0).tolist() if n_points > 0 else [0, 0, 0],
        min=points.min(axis=0).tolist() if n_points > 0 else [0, 0, 0],
    ))
    offset += len(points_bin)

    # BV1: point colors
    buffer_views.append(pygltflib.BufferView(
        buffer=0, byteOffset=offset, byteLength=len(colors_bin),
        target=pygltflib.ARRAY_BUFFER,
    ))
    accessors.append(pygltflib.Accessor(
        bufferView=1, componentType=pygltflib.UNSIGNED_BYTE,
        count=n_points, type=pygltflib.VEC4,
        normalized=True,
    ))
    offset += len(colors_bin)

    nodes = []
    meshes = []

    # Point cloud mesh (GL_POINTS = mode 0)
    meshes.append(pygltflib.Mesh(
        primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=0, COLOR_0=1),
            mode=0,
        )]
    ))
    nodes.append(pygltflib.Node(mesh=0))

    if has_boxes:
        # BV2: box vertices
        buffer_views.append(pygltflib.BufferView(
            buffer=0, byteOffset=offset, byteLength=len(box_verts_bin),
            target=pygltflib.ARRAY_BUFFER,
        ))
        accessors.append(pygltflib.Accessor(
            bufferView=2, componentType=pygltflib.FLOAT,
            count=len(box_verts), type=pygltflib.VEC3,
            max=box_verts.max(axis=0).tolist(),
            min=box_verts.min(axis=0).tolist(),
        ))
        offset += len(box_verts_bin)

        # BV3: box colors
        buffer_views.append(pygltflib.BufferView(
            buffer=0, byteOffset=offset, byteLength=len(box_colors_bin),
            target=pygltflib.ARRAY_BUFFER,
        ))
        accessors.append(pygltflib.Accessor(
            bufferView=3, componentType=pygltflib.UNSIGNED_BYTE,
            count=len(box_vert_colors), type=pygltflib.VEC4,
            normalized=True,
        ))
        offset += len(box_colors_bin)

        # BV4: box indices
        buffer_views.append(pygltflib.BufferView(
            buffer=0, byteOffset=offset, byteLength=len(box_indices_bin),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        ))
        accessors.append(pygltflib.Accessor(
            bufferView=4, componentType=pygltflib.UNSIGNED_SHORT,
            count=len(box_indices), type=pygltflib.SCALAR,
            max=[int(box_indices.max())],
            min=[int(box_indices.min())],
        ))
        offset += len(box_indices_bin)

        # Box wireframe mesh (GL_LINES = mode 1)
        meshes.append(pygltflib.Mesh(
            primitives=[pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=2, COLOR_0=3),
                indices=4,
                mode=1,
            )]
        ))
        nodes.append(pygltflib.Node(mesh=1))

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=list(range(len(nodes))))],
        nodes=nodes,
        meshes=meshes,
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[pygltflib.Buffer(byteLength=len(blob))],
    )
    gltf.set_binary_blob(blob)
    gltf.save(output_path)

    return output_path


def _create_edge_cylinder(p1, p2, radius=0.01, sections=6):
    """Create a thin cylinder mesh between two 3D points.

    Args:
        p1, p2: (3,) endpoints.
        radius: cylinder radius.
        sections: number of radial segments.

    Returns:
        trimesh.Trimesh or None if edge is degenerate.
    """
    import trimesh

    segment = p2 - p1
    length = float(np.linalg.norm(segment))
    if length < 1e-6:
        return None

    cyl = trimesh.creation.cylinder(
        radius=radius, height=length, sections=sections
    )
    direction = segment / length

    # Align cylinder Z-axis to segment direction
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    cross = np.cross(z_axis, direction)
    dot = np.dot(z_axis, direction)
    cross_len = np.linalg.norm(cross)

    if cross_len < 1e-6:
        R = np.eye(3) if dot > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        cross_n = cross / cross_len
        angle = np.arccos(np.clip(dot, -1, 1))
        K = np.array([
            [0, -cross_n[2], cross_n[1]],
            [cross_n[2], 0, -cross_n[0]],
            [-cross_n[1], cross_n[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (p1 + p2) / 2.0
    cyl.apply_transform(T)
    return cyl


def _create_wireframe_box_trimesh(corners, color_rgba, radius=0.015):
    """Create wireframe box as thin cylinders.

    Args:
        corners: (8, 3) corner positions in glTF coords.
        color_rgba: [R, G, B, A] uint8 color.
        radius: cylinder radius in meters.

    Returns:
        trimesh.Trimesh or None.
    """
    import trimesh

    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    parts = []
    for a, b in edge_pairs:
        cyl = _create_edge_cylinder(
            corners[a].astype(np.float64),
            corners[b].astype(np.float64),
            radius=radius,
            sections=6,
        )
        if cyl is not None:
            cyl.visual.face_colors = color_rgba
            parts.append(cyl)

    if parts:
        return trimesh.util.concatenate(parts)
    return None


def create_mesh_scene_glb(
    depth_map: np.ndarray,
    image: np.ndarray,
    intrinsics: np.ndarray,
    boxes3d_list: list[np.ndarray],
    output_path: str,
    max_depth: float = 20.0,
    padding: tuple[int, int, int, int] | None = None,
    remove_edge: bool = True,
    edge_rtol: float = 0.04,
) -> str:
    """Create GLB with textured mesh (MoGe2 style) + wireframe 3D boxes.

    Args:
        depth_map: (H, W) or (1, H, W) depth in meters.
        image: (H, W, 3) RGB uint8 [0-255].
        intrinsics: (3, 3) camera intrinsics.
        boxes3d_list: List of (M, 10) box arrays.
        output_path: Where to save .glb.
        max_depth: Max depth cutoff.
        padding: (left, right, top, bottom) to exclude.
        remove_edge: Remove depth discontinuity edges.
        edge_rtol: Edge detection tolerance.

    Returns:
        output_path.
    """
    import utils3d
    import trimesh
    from PIL import Image as PILImage

    # Prepare depth
    while depth_map.ndim > 2:
        depth_map = depth_map.squeeze(0)
    depth_map = depth_map.astype(np.float32)
    H, W = depth_map.shape

    # Prepare image
    while image.ndim > 3:
        image = image.squeeze(0)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[0] != H or image.shape[1] != W:
        img_pil = PILImage.fromarray(image)
        img_pil = img_pil.resize((W, H), PILImage.BILINEAR)
        image = np.array(img_pil)

    # Build valid mask
    valid = (
        (depth_map > 0.01)
        & (depth_map < max_depth)
        & np.isfinite(depth_map)
    )

    if padding is not None:
        pad_left, pad_right, pad_top, pad_bottom = padding
        if pad_left > 0:
            valid[:, :pad_left] = False
        if pad_right > 0:
            valid[:, W - pad_right:] = False
        if pad_top > 0:
            valid[:pad_top, :] = False
        if pad_bottom > 0:
            valid[H - pad_bottom:, :] = False

    if remove_edge:
        edge = utils3d.np.depth_map_edge(depth_map, rtol=edge_rtol)
        valid &= ~edge

    # Unproject to 3D in OpenCV coords (x-right, y-down, z-forward)
    # Build mesh in OpenCV space first so triangle winding is correct,
    # then transform vertices to glTF coords afterwards.
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x3d = (u - cx) * depth_map / fx
    y3d = (v - cy) * depth_map / fy
    z3d = depth_map
    points_cv = np.stack([x3d, y3d, z3d], axis=-1).astype(np.float32)

    # UV map
    uv = np.stack(
        [u / max(W - 1, 1), v / max(H - 1, 1)], axis=-1
    ).astype(np.float32)

    # Colors normalized [0, 1]
    colors = image.astype(np.float32) / 255.0

    # Build triangulated mesh in OpenCV coords (preserves correct winding)
    faces, vertices, vertex_colors, vertex_uvs = (
        utils3d.np.build_mesh_from_map(
            points_cv, colors, uv, mask=valid, tri=True
        )
    )

    print(
        f"[Mesh] {vertices.shape[0]} vertices, "
        f"{faces.shape[0]} faces, "
        f"valid pixels: {valid.sum()}/{valid.size}"
    )

    if len(vertices) == 0:
        # Fallback to empty file
        scene = trimesh.Scene()
        scene.export(output_path)
        return output_path

    # Transform vertices: OpenCV (x, y, z) -> glTF (x, -y, -z)
    # This is a 180-degree rotation around x-axis (det=+1),
    # so it preserves triangle winding order.
    vertices = vertices * np.array([1.0, -1.0, -1.0], dtype=np.float32)

    # Trimesh flips UV v when exporting to GLB (OpenGL v=0 at bottom
    # vs glTF v=0 at top). Our UVs are already in image convention
    # (v=0 at top), so pre-flip to compensate for trimesh's flip.
    vertex_uvs = vertex_uvs.copy()
    vertex_uvs[:, 1] = 1.0 - vertex_uvs[:, 1]

    # Create textured mesh (process=False to avoid trimesh modifying geometry)
    texture_img = PILImage.fromarray(image)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_img,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    visuals = trimesh.visual.TextureVisuals(
        uv=vertex_uvs, material=material
    )
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, visual=visuals,
        process=False,
    )

    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="scene_mesh")

    # Add wireframe 3D boxes as thin cylinder geometry
    all_corners = []
    for boxes3d in boxes3d_list:
        if len(boxes3d) > 0:
            corners = boxes3d_to_corners(boxes3d)
            all_corners.extend(corners)

    box_colors = _generate_box_colors(len(all_corners))
    for i, corners in enumerate(all_corners):
        box_mesh = _create_wireframe_box_trimesh(
            corners, box_colors[i], radius=0.015
        )
        if box_mesh is not None:
            scene.add_geometry(
                box_mesh, node_name=f"box_{i}"
            )

    scene.export(output_path)
    return output_path

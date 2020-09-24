#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import pycuda.driver as cuda
import pycuda.autoinit, time
from pycuda.compiler import SourceModule
from numpy import array, arange, zeros, where, dtype, float32, int32, uint32
from numpy import count_nonzero, resize, uint8

helpers_mod = SourceModule(open("cuda_helpers.cu", "r").read(),
    include_dirs=["/home/ckw/ResearchDev/VCDev/peridynamics/src/viewer",
        "/Users/ckw/ResearchDev/__PerRsrch__/peridynamics/src/viewer",
        "/home/cwatcha/RDev/viewer"])

def find_boundary(d_tetras, d_bounders, n_tetras, d_boundary_flags,
    n_threads=512):

    d_n_tetras = int32(n_tetras)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("find_boundary")
    cuda_func(d_tetras, d_bounders, d_n_tetras, d_boundary_flags,
        grid=grid, block=block)

# Compute cut weights between cut_function and tetrahedral edges.
def compute_cut_weights(d_positions, d_sim_positions,
    d_tetras, n_tetras, d_cut_weights, d_broken_edges, cut_plane,
    n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_tetras = int32(n_tetras)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("compute_cut_weights")
    cuda_func(d_positions, d_sim_positions,
        d_tetras, d_n_tetras, d_cut_weights, d_broken_edges,
        cuda.In(cut_plane), grid=grid, block=block)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("compute_cut_weights: CUDA clock timing: %.4f secs" % secs)
    return

# Use geometric distance to determine edge breaking in the tetrahedron.
def breaking_edge_check(d_positions, d_sim_positions,
    d_tetras, n_tetras, d_breaking_edges, d_broken_edges,
    broken_distance, n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_tetras = int32(n_tetras)
    d_broken_distance = float32(broken_distance)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("breaking_edge_check")
    cuda_func(d_positions, d_sim_positions, d_tetras, d_n_tetras,
        d_breaking_edges, d_broken_edges, d_broken_distance,
        grid=grid, block=block)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("breaking_edge_check: CUDA clock timing: %.4f secs" % secs)

# Calculate the number of triangles based on the broken edges.
def tri_counts(d_boundary_flags, n_tetras, d_broken_edges,
    tri_counts, accum_tri_counts, n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_tetras = int32(n_tetras)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("tri_counts")
    cuda_func(d_boundary_flags, d_n_tetras, d_broken_edges,
        cuda.Out(tri_counts), grid=grid, block=block)
    accum_tri_counts[:] = 0
    for i in range(1, n_tetras):
        accum_tri_counts[i] = accum_tri_counts[i-1] + tri_counts[i-1]

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("tri_counts: CUDA clock timing: %.4f secs" % secs)
    return accum_tri_counts[n_tetras-1] + tri_counts[n_tetras-1]

# Use geometric distance to determine edge breaks in the tetrahedron.
def broken_edge_check_tri_counts(d_positions, d_sim_positions,
    n_active_particles, d_labels, d_boundary_flags,
    d_tetras, n_tetras, d_broken_edges, broken_distance,
    tri_counts, accum_tri_counts, n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_active_particles = int32(n_active_particles)
    d_n_tetras = int32(n_tetras)
    d_broken_distance = float32(broken_distance)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("broken_edge_check_tri_counts")
    cuda_func(d_positions, d_sim_positions, d_n_active_particles, d_labels,
        d_boundary_flags, d_tetras, d_n_tetras, d_broken_edges,
        d_broken_distance, cuda.Out(tri_counts), grid=grid, block=block)
    accum_tri_counts[:] = 0
    for i in range(1, n_tetras):
        accum_tri_counts[i] = accum_tri_counts[i-1] + tri_counts[i-1]

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("broken_..._tri_counts: CUDA clock timing: %.4f secs" % secs)
    return accum_tri_counts[n_tetras-1] + tri_counts[n_tetras-1]

# Use bond information to determine edge breaks in the tetrahedron.
def bond_broken_edge_tri_counts(d_positions, d_sim_positions,
    n_active_particles, bondlist, n_bonds, maxbonds, d_labels,
    d_boundary_flags, d_tetras, n_tetras, d_broken_edges,
    tri_counts, accum_tri_counts, n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_active_particles = int32(n_active_particles)
    d_n_tetras = int32(n_tetras)
    d_maxbonds = int32(maxbonds)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("bond_broken_edge_tri_counts")
    cuda_func(d_positions, d_sim_positions, d_n_active_particles,
        cuda.In(bondlist), cuda.In(n_bonds), d_maxbonds,
        d_labels, d_boundary_flags, d_tetras, d_n_tetras, d_broken_edges,
        cuda.Out(tri_counts), grid=grid, block=block)
    accum_tri_counts[:] = 0
    for i in range(1, n_tetras):
        accum_tri_counts[i] = accum_tri_counts[i-1] + tri_counts[i-1]

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("broken_..._tri_counts: CUDA clock timing: %.4f secs" % secs)
    return accum_tri_counts[n_tetras-1] + tri_counts[n_tetras-1]

# extracting surface with cut weights and per particle transformation
def march_tetra_with_cut_weights(d_positions, d_sim_positions, d_labels,
    d_T0, d_T1, d_R, d_A, d_boundary_flags, d_tetras, n_tetras,
    n_bonds, d_cut_weights, d_broken_edges, accum_tri_counts, n_triangles,
    label_component, d_tri_vertices, d_tri_normals, d_tri_uvs, d_tri_labels,
    tri_vertices, tri_normals, tri_uvs, tri_labels,
    n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_tetras = int32(n_tetras)
    d_label_component = int32(label_component)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("march_tetra_with_cut_weights")
    cuda_func(d_positions, d_sim_positions,
        d_labels, d_T0, d_T1, d_R, d_A,
        d_boundary_flags, d_tetras, d_n_tetras, cuda.In(n_bonds),
        d_cut_weights, d_broken_edges, cuda.In(accum_tri_counts),
        d_label_component,
        d_tri_vertices, d_tri_normals, d_tri_uvs, d_tri_labels,
        grid=grid, block=block)

    cuda.memcpy_dtoh(tri_vertices, d_tri_vertices)
    cuda.memcpy_dtoh(tri_normals, d_tri_normals)
    cuda.memcpy_dtoh(tri_uvs, d_tri_uvs)
    cuda.memcpy_dtoh(tri_labels, d_tri_labels)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("surface: CUDA clock timing: %.4f secs" % secs)

    return tri_vertices, tri_normals, tri_uvs, tri_labels, n_triangles

# extracting surface with cut weights and per particle transformation
# also generating triangle split side of a cutting plane
def march_tetra_with_cut_weights_and_split(
    d_positions, d_sim_positions, d_labels,
    d_T0, d_T1, d_R, d_A, d_boundary_flags, d_tetras, n_tetras,
    n_bonds, d_cut_weights, d_broken_edges, accum_tri_counts, n_triangles,
    label_component, d_tri_vertices, d_tri_normals, d_tri_uvs, d_tri_labels,
    tri_vertices, tri_normals, tri_uvs, tri_labels,
    cut_plane, tri_cut_sides,
    n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_tetras = int32(n_tetras)
    d_label_component = int32(label_component)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function( \
        "march_tetra_with_cut_weights_and_split")
    cuda_func(d_positions, d_sim_positions,
        d_labels, d_T0, d_T1, d_R, d_A,
        d_boundary_flags, d_tetras, d_n_tetras, cuda.In(n_bonds),
        d_cut_weights, d_broken_edges, cuda.In(accum_tri_counts),
        d_label_component,
        d_tri_vertices, d_tri_normals, d_tri_uvs, d_tri_labels,
        cuda.In(cut_plane), cuda.Out(tri_cut_sides), grid=grid, block=block)

    cuda.memcpy_dtoh(tri_vertices, d_tri_vertices)
    cuda.memcpy_dtoh(tri_normals, d_tri_normals)
    cuda.memcpy_dtoh(tri_uvs, d_tri_uvs)
    cuda.memcpy_dtoh(tri_labels, d_tri_labels)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("surface: CUDA clock timing: %.4f secs" % secs)

    return tri_vertices, tri_normals, tri_uvs, tri_labels, n_triangles

# extracting surface with function cuts
def march_tetra_cuts(d_positions, d_sim_positions, d_labels,
    d_T0, d_T1, d_R, d_A, d_tetras, d_bounders, n_tetras,
    n_bonds, d_cut_weights, d_broken_edges, broken_distance,
    label_component, n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    tris_per_tet = 24+24
    tri_counts = zeros((n_tetras), dtype=int32)
    d_n_tetras = int32(n_tetras)
    d_tris_per_tet = int32(tris_per_tet)
    d_broken_distance = float32(broken_distance)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    d_out_vertices = cuda.mem_alloc(dtype('float32').itemsize * \
        n_tetras*tris_per_tet*3*3)
    d_tri_labels = cuda.mem_alloc(dtype('int32').itemsize * \
        n_tetras*tris_per_tet*3*3)
    m_surf_func = helpers_mod.get_function("march_tetra_cuts")
    m_surf_func(d_positions, d_sim_positions,
        d_labels, d_T0, d_T1, d_R, d_A,
        d_tetras, d_bounders, d_n_tetras, cuda.In(n_bonds),
        d_cut_weights, d_broken_edges, d_broken_distance,
        d_label_component,
        d_out_vertices, cuda.Out(tri_counts), d_tri_labels,
        d_tris_per_tet, grid=grid, block=block)
    n_triangles = int(sum(tri_counts))

    if n_triangles != 0:
        packed_vertices = zeros((n_triangles*3*3), dtype=float32)
        packed_normals  = zeros((n_triangles*3*3), dtype=float32)
        packed_labels   = zeros((n_triangles*3*3), dtype=float32)

        pack_vertices_func = helpers_mod.get_function("pack_vertices_labels")
        pack_vertices_func(d_out_vertices, cuda.In(tri_counts),
            d_tri_labels, cuda.Out(packed_vertices),
            cuda.Out(packed_normals), cuda.Out(packed_labels),
            d_tris_per_tet, d_n_tetras, grid=grid, block=block)
    else:
        packed_vertices = None
        packed_normals  = None
        packed_labels   = None

    d_out_vertices.free()
    d_tri_labels.free()
    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("surface: CUDA clock timing: %.4f secs" % secs)

    return packed_vertices, packed_normals, packed_labels, n_triangles

# extracting surface with per particle transformation
def march_tetra(d_positions, d_sim_positions, n_active_particles, d_labels,
    d_T0, d_T1, d_R, d_A, d_boundary_flags, d_tetras, n_tetras,
    n_bonds, d_broken_edges, accum_tri_counts, n_triangles,
    label_component, d_tri_vertices, d_tri_normals, d_tri_uvs, d_tri_labels,
    tri_vertices, tri_normals, tri_uvs, tri_labels,
    n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_active_particles = int32(n_active_particles)
    d_n_tetras = int32(n_tetras)
    d_label_component = int32(label_component)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("march_tetra")
    cuda_func(d_positions, d_sim_positions, d_n_active_particles,
        d_labels, d_T0, d_T1, d_R, d_A,
        d_boundary_flags, d_tetras, d_n_tetras, cuda.In(n_bonds),
        d_broken_edges, cuda.In(accum_tri_counts),
        d_label_component,
        d_tri_vertices, d_tri_normals, d_tri_uvs, d_tri_labels,
        grid=grid, block=block)

    cuda.memcpy_dtoh(tri_vertices, d_tri_vertices)
    cuda.memcpy_dtoh(tri_normals, d_tri_normals)
    cuda.memcpy_dtoh(tri_uvs, d_tri_uvs)
    cuda.memcpy_dtoh(tri_labels, d_tri_labels)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("surface: CUDA clock timing: %.4f secs" % secs)

    return tri_vertices, tri_normals, tri_uvs, tri_labels, n_triangles

# extracting surface with per component transformation
def march_tetra_per_component_xform(d_positions, d_sim_positions, n_particles,
    d_labels, d_T0, d_T1, d_R, d_A, d_boundary_flags, d_tetras, n_tetras,
    n_bonds, d_broken_edges, accum_tri_counts, n_triangles,
    label_component, d_tri_vertices, d_tri_normals, d_tri_uvs, d_tri_labels,
    tri_vertices, tri_normals, tri_uvs, tri_labels,
    n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_tetras = int32(n_tetras)
    d_label_component = int32(label_component)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("march_tetra_per_component_xform")
    cuda_func(d_positions, d_sim_positions,
        d_labels, d_T0, d_T1, d_R, d_A,
        d_boundary_flags, d_tetras, d_n_tetras, cuda.In(n_bonds),
        d_broken_edges, cuda.In(accum_tri_counts),
        d_label_component,
        d_tri_vertices, d_tri_normals, d_tri_uvs, d_tri_labels,
        grid=grid, block=block)

    cuda.memcpy_dtoh(tri_vertices, d_tri_vertices)
    cuda.memcpy_dtoh(tri_normals, d_tri_normals)
    cuda.memcpy_dtoh(tri_uvs, d_tri_uvs)
    cuda.memcpy_dtoh(tri_labels, d_tri_labels)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("surface: CUDA clock timing: %.4f secs" % secs)

    return tri_vertices, tri_normals, tri_uvs, tri_labels, n_triangles

def march_tetra_pack_vertices(d_positions, d_sim_positions, d_labels,
    d_T0, d_T1, d_R, d_A, d_tetras, d_bounders, n_tetras,
    n_bonds, d_broken_edges, broken_distance,
    label_component, n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    tris_per_tet = 24+24
    tri_counts = zeros((n_tetras), dtype=int32)
    d_n_tetras = int32(n_tetras)
    d_tris_per_tet = int32(tris_per_tet)
    d_broken_distance = float32(broken_distance)
    d_label_component = int32(label_component)

    grid = (n_tetras//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    d_out_vertices = cuda.mem_alloc(dtype('float32').itemsize * \
        n_tetras*tris_per_tet*3*3)
    d_tri_labels = cuda.mem_alloc(dtype('int32').itemsize * \
        n_tetras*tris_per_tet*3*3)
    m_surf_func = helpers_mod.get_function("march_tetra_per_particle_xform")
    m_surf_func(d_positions, d_sim_positions,
        d_labels, d_T0, d_T1, d_R, d_A,
        d_tetras, d_bounders, d_n_tetras, cuda.In(n_bonds),
        d_broken_edges, d_broken_distance,
        d_label_component,
        d_out_vertices, cuda.Out(tri_counts), d_tri_labels,
        d_tris_per_tet, grid=grid, block=block)
    n_triangles = int(sum(tri_counts))

    if n_triangles != 0:
        packed_vertices = zeros((n_triangles*3*3), dtype=float32)
        packed_normals  = zeros((n_triangles*3*3), dtype=float32)
        packed_labels   = zeros((n_triangles*3*3), dtype=float32)

        pack_vertices_func = helpers_mod.get_function("pack_vertices_labels")
        pack_vertices_func(d_out_vertices, cuda.In(tri_counts),
            d_tri_labels, cuda.Out(packed_vertices),
            cuda.Out(packed_normals), cuda.Out(packed_labels),
            d_tris_per_tet, d_n_tetras, grid=grid, block=block)
    else:
        packed_vertices = None
        packed_normals  = None
        packed_labels   = None

    d_out_vertices.free()
    d_tri_labels.free()
    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("surface: CUDA clock timing: %.4f secs" % secs)

    return packed_vertices, packed_normals, packed_labels, n_triangles

def label_particles(h_bondlist, h_n_bonds, n_particles, maxbonds, h_labels,
    label_to_arr_idx=None, pack_labels=True, n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_bondlist = cuda.mem_alloc(h_bondlist.size * h_bondlist.dtype.itemsize)
    d_n_bonds = cuda.mem_alloc(h_n_bonds.size * h_n_bonds.dtype.itemsize)
    d_labels = cuda.mem_alloc(h_labels.size * h_labels.dtype.itemsize)
    d_tmp_labels = cuda.mem_alloc(h_labels.size * h_labels.dtype.itemsize)
    cuda.memcpy_htod(d_bondlist, h_bondlist)
    cuda.memcpy_htod(d_n_bonds, h_n_bonds)
    d_n_particles = int32(n_particles)
    d_maxbonds = int32(maxbonds)
    h_n_updates = zeros(1, dtype=int32)
    d_n_updates = cuda.mem_alloc(h_n_updates.size * h_n_updates.dtype.itemsize)
    h_marked_labels = zeros(n_particles, dtype=uint8)
    d_marked_labels = cuda.mem_alloc(n_particles)

    grid = (n_particles//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    h_labels[:] = arange(n_particles);
    cuda.memcpy_htod(d_tmp_labels, h_labels)

    label_particles_func = helpers_mod.get_function("label_particles")
    mark_labels_func = helpers_mod.get_function("mark_labels")
    renum_labels_func = helpers_mod.get_function("renum_labels")

    while True:
        h_n_updates[0] = 0
        cuda.memcpy_htod(d_n_updates, h_n_updates)
        label_particles_func(d_bondlist, d_n_bonds, d_n_particles, d_maxbonds,
            d_tmp_labels, d_n_updates, grid=grid, block=block)
        cuda.memcpy_dtoh(h_n_updates, d_n_updates)
        if h_n_updates[0] == 0:
            break

    if pack_labels:
        mark_labels_func(d_n_particles, d_tmp_labels, d_marked_labels,
            grid=grid, block=block)
        renum_labels_func(d_n_particles, d_tmp_labels, d_marked_labels, d_labels,
            grid=grid, block=block)

        cuda.memcpy_dtoh(h_labels, d_labels)
        cuda.memcpy_dtoh(h_marked_labels, d_marked_labels)
        n_labels = count_nonzero(h_marked_labels)
    else:
        cuda.memcpy_dtoh(h_labels, d_tmp_labels)
        label_to_arr_idx[:] = -1
        mapped_index = where(h_labels==arange(n_particles))
        label_to_arr_idx[mapped_index] = arange(len(mapped_index[0]))
        n_labels = len(mapped_index[0])

    d_bondlist.free()
    d_n_bonds.free()
    d_labels.free()
    d_tmp_labels.free()
    d_n_updates.free()
    d_marked_labels.free()

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("label: CUDA clock timing: %.4f secs" % secs)

    return n_labels

def compute_per_particle_procrustes(d_positions, d_sim_positions, n_particles,
    bondlist, n_bonds, maxbonds, d_T0, d_T1, d_R, d_A, random_scale,
    n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_particles = int32(n_particles)
    d_maxbonds = int32(maxbonds)

    grid = (n_particles//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    procrustes_func = helpers_mod.get_function(
        "compute_per_particle_procrustes")
    procrustes_func(d_positions, d_sim_positions, d_n_particles,
        cuda.In(bondlist), cuda.In(n_bonds), d_maxbonds,
        d_T0, d_T1, d_R, d_A, cuda.In(random_scale), grid=grid, block=block)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("compute_procrustes: CUDA clock timing: %.4f secs" % secs)

def compute_per_particle_procrustes_cem(d_positions, d_sim_positions,
    n_particles, bondlist, n_bonds, maxbonds, d_T0, d_T1, d_R, d_A,
    random_scale, n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_particles = int32(n_particles)
    d_maxbonds = int32(maxbonds)

    grid = (n_particles//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    procrustes_func = helpers_mod.get_function(
        "compute_per_particle_procrustes_cem")
    procrustes_func(d_positions, d_sim_positions, d_n_particles,
        cuda.In(bondlist), cuda.In(n_bonds), d_maxbonds,
        d_T0, d_T1, d_R, d_A, cuda.In(random_scale), grid=grid, block=block)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("compute_procrustes: CUDA clock timing: %.4f secs" % secs)

def compute_per_component_procrustes(d_positions, d_sim_positions, n_particles,
    d_labels, max_labels, bondlist, n_bonds, maxbonds,
    d_T0, d_T1, d_R, d_A, random_scale,
    n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_particles = int32(n_particles)
    d_max_labels = int32(max_labels)
    d_maxbonds = int32(maxbonds)

    grid = (max_labels//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    procrustes_func = helpers_mod.get_function(
        "compute_per_component_procrustes")
    procrustes_func(d_positions, d_sim_positions, d_n_particles,
        d_labels, d_max_labels,
        cuda.In(bondlist), cuda.In(n_bonds), d_maxbonds,
        d_T0, d_T1, d_R, d_A, cuda.In(random_scale), grid=grid, block=block)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("compute_procrustes: CUDA clock timing: %.4f secs" % secs)

# Compute FTLEs.
def compute_FTLE(d_positions, next_positions, n_particles, bondlist,
    n_bonds, accum_n_bonds, maxbonds, inv_ftle_tau, locations, ftles,
    n_threads=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    d_n_particles = int32(n_particles)
    d_maxbonds = int32(maxbonds)
    d_inv_ftle_tau = float32(inv_ftle_tau)

    grid = (n_particles//n_threads + 1, 1, 1)
    if grid[0] > 65535:
        grid = (32768, grid[0]//32768, 1)
    block = (n_threads, 1, 1)

    cuda_func = helpers_mod.get_function("compute_FTLE")
    cuda_func(d_positions, cuda.In(next_positions), d_n_particles,
        cuda.In(bondlist), cuda.In(n_bonds), cuda.In(accum_n_bonds),
        d_maxbonds, d_inv_ftle_tau, cuda.Out(locations), cuda.Out(ftles),
        grid=grid, block=block)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("compute_FTLE: CUDA clock timing: %.4f secs" % secs)

def remove_duplicates(vertices, n_vertices, n_slices=512):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    grid = (n_vertices//n_slices+1, n_slices, 1)
    block = (n_slices, 1, 1)

    out_vertices = zeros(n_vertices*3, dtype=float32)
    faces = arange(n_vertices, dtype=int32)
    out_faces = arange(n_vertices, dtype=int32)
    h_actual_n_vertices = zeros(1, dtype=int32)
    d_actual_n_vertices = cuda.mem_alloc(h_actual_n_vertices.size * h_actual_n_vertices.dtype.itemsize)
    cuda.memcpy_htod(d_actual_n_vertices, h_actual_n_vertices)
    d_n_vertices = int32(n_vertices)
    d_n_slices = int32(n_slices)

    remove_duplicates_func = helpers_mod.get_function("remove_duplicates")
    remove_duplicates_func(cuda.In(vertices), cuda.In(faces),
        d_n_slices, cuda.InOut(out_faces),
        d_n_vertices, d_actual_n_vertices,
        grid=grid, block=block)

    cuda.memcpy_dtoh(h_actual_n_vertices, d_actual_n_vertices)
    resize(out_vertices, (h_actual_n_vertices[0]*3, 1))

    d_actual_n_vertices.free()

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print("remove dups: CUDA clock timing: %.4f secs" % secs)
    print(min(out_faces), max(out_faces))

    return out_vertices, out_faces, h_actual_n_vertices[0], n_vertices//3

def print_threads(grid=(2,3,1), block=(4,1,1)):

    print_threads_func = helpers_mod.get_function("print_threads")
    print_threads_func(grid=grid, block=block)

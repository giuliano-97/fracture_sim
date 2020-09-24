#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from numpy import array, zeros, arange, float32, int32, uint32, uint8, uint16
from numpy import all, where, fromstring, linalg, cross, reshape, nan, dot
from numpy import ravel, dtype, mean, argmax, count_nonzero, errstate, divide
from numpy import identity, zeros_like, append, amin, cumsum
from numpy.linalg import svd, inv, solve, norm
from scipy.linalg import eigh
from numpy import random, append as np_append
from math import sqrt, log
from sys import stdout
import re, os, sys, numpy.random, struct
from time import time, sleep, clock
import cuda_helpers
import pycuda.driver as cuda

vertices, triangles, marched_labels, marched_n_triangles = None,False,None,0
start_frame, frame_no, mesh_file, label_components = None, 1, None, False
cpu_procrustes, per_component_procrustes, no_procrustes = False, False, False
write_start_frame, write_stl_config, write_rest_stl = -1, None, False
bond_broken_edge, end_frame = False, -1

class SimFile:
    def __init__(self, filename, counter=0):
        try:
            self.f_handler = open(filename, "rb")
        except:
            print("Couldn't open \"{0}\" for read!".format(filename))
            raise
        self.sim_filename = filename
        self.project_name = None
        while True:
            header = self.f_handler.readline().decode('utf-8')
            match = re.match('.*object_size.*?=[ ](.*?)[, ]', header, re.I)
            if match:
                break
        self.time_step = 1
        int_list = ("object_size", "n_particles", "n_glass_particles", \
            "maxbonds", )
        special_list = ("steps_per_render", "lambda", "info", )
        try:
            for w in int_list + special_list:
                match = re.match(".*%s[ ]=[ ](.*?)[,\n]" % w, header, re.I)
                if match:
                    if w in int_list:
                        exec("self.%s = int(%s)" % (w, match.group(1)))
                    elif w == "steps_per_render":
                        self.time_step = float(match.group(1))
                    elif w == "lambda":
                        self.p_lambda = float(match.group(1))
                    else:
                        exec("self.%s = \"%s\"" % (w, match.group(1)))
        except:
            print("Couldn't open \"{0}\" for read!".format(filename))
            raise
        self.n_active_particles = self.n_particles
        self.has_all_info = self.has_color_info = self.has_bond_info = \
            self.has_velocity_info = self.has_force_info = False
        if self.object_size == 16:
            self.has_bond_info = True
        if self.object_size >= 24 and self.object_size <= 40:
            if self.object_size == 24 and 'V' not in self.info:
                self.has_velocity_info = True
            if 'B' in self.info:
                self.has_bond_info = True
            if 'V' in self.info:
                self.has_velocity_info = True
            if 'F' in self.info:
                self.has_force_info = True
        elif self.object_size > 40:
            self.has_all_info = self.has_bond_info = self.has_velocity_info = \
                self.has_force_info = True
        self.start_frame = self.f_handler.tell()
        self.frame_elements = 3 * self.n_particles  # pos element
        self.frame_bytes = self.object_size*self.n_particles
        self.last_time, self.frame_no = 0, 0
        self.read_bytes = 3 * dtype('float32').itemsize * self.n_particles
        self.vertices = zeros(self.frame_elements, dtype=float32)
        self.next_vertices = zeros(self.frame_elements, dtype=float32)
        self.data = zeros(self.frame_elements, dtype=float32)
        self.n_bonds = zeros(self.n_particles, dtype=int32)
        self.start_n_bonds = zeros(self.n_particles, dtype=int32)
        self.bond_ratios = zeros(self.n_particles, dtype=float32)
        self.velocities = zeros(self.frame_elements, dtype=float32)
        if self.has_force_info:
            self.forces = zeros(self.frame_elements, dtype=float32)
            self.x_forces3 = zeros((self.n_particles, 3), dtype=float32)

        self.start_n_bonds_loaded = False
        if self.has_all_info:
            self.bondlist = zeros(self.n_particles*self.maxbonds,
                dtype=int32)
            self.labels = zeros(self.n_particles, dtype=int32)
            self.n_labels = 0.
            self.org_bondlist = zeros(self.n_particles*self.maxbonds,
                dtype=int32)
            self.org_n_bonds = zeros(self.n_particles, dtype=int32)
        if counter == 0:
            self.org_vertices = zeros(self.frame_elements, dtype=float32)
            if self.has_all_info:
                self.d_org_vertices = cuda.mem_alloc( \
                        self.org_vertices.size * \
                        self.org_vertices.dtype.itemsize)
                self.d_vertices = cuda.mem_alloc( \
                        self.vertices.size * self.vertices.dtype.itemsize)
                self.d_labels = cuda.mem_alloc( \
                    self.labels.size * self.labels.dtype.itemsize)
                self.d_bond_ratios = cuda.mem_alloc( \
                    self.bond_ratios.size * self.bond_ratios.dtype.itemsize)
                self.d_org_bondlist = cuda.mem_alloc( \
                    self.org_bondlist.size * self.org_bondlist.dtype.itemsize)
                self.d_org_n_bonds = cuda.mem_alloc( \
                    self.org_n_bonds.size * self.org_n_bonds.dtype.itemsize)
                self.n_triangles = 0
        else:
            self.org_vertices = None

    def read_data(self):
        int_size = dtype('int32').itemsize
        buf = self.f_handler.read(self.read_bytes)
        if len(buf) != self.read_bytes:
            return 0
        self.vertices[:] = fromstring(buf, dtype=float32)
        if self.has_bond_info or self.has_all_info:
            buf = self.f_handler.read(int_size*self.n_particles)
            if len(buf) != int_size*self.n_particles:
                return 0
            self.n_bonds[:] = fromstring(buf, dtype=int32)
            self.n_bonds[:] += 1
        if self.has_velocity_info or self.has_all_info:
            buf = self.f_handler.read(self.read_bytes)
            if len(buf) != self.read_bytes:
                return 0
            self.velocities[:] = fromstring(buf, dtype=float32)
        if self.has_force_info or self.has_all_info:
            buf = self.f_handler.read(self.read_bytes)
            if len(buf) != self.read_bytes:
                return 0
            self.forces[:] = fromstring(buf, dtype=float32)
        if self.has_all_info:
            buf = self.f_handler.read(int_size*self.n_particles*self.maxbonds)
            if len(buf) != int_size*self.n_particles*self.maxbonds:
                return 0
            self.bondlist[:] = fromstring(buf, dtype=int32)
        return 1

    def play(self, cur_time=None):
        if cur_time is not None:
            cnt = int(cur_time / self.time_step)
            if cnt < 0:
                return 0
            self.f_handler.seek(self.start_frame+cnt*self.frame_bytes, 0)

        if self.read_data() == 0:
            return 0

        if cur_time == 0:
            if self.org_vertices is not None:
                self.org_vertices[:] = self.vertices[:]
                if self.has_all_info:
                    self.org_bondlist[:] = self.bondlist[:]
                    self.org_n_bonds[:] = self.n_bonds[:]
                    cuda.memcpy_htod(self.d_org_vertices, self.org_vertices)
                    cuda.memcpy_htod(self.d_org_bondlist, self.org_bondlist)
                    cuda.memcpy_htod(self.d_org_n_bonds, self.org_n_bonds)
            if not self.start_n_bonds_loaded and \
                (self.has_all_info or self.has_bond_info):
                self.start_n_bonds[:] = self.n_bonds[:]
                self.start_n_bonds_loaded = True
        self.frame_no += 1
        self.last_time += self.time_step
        return 1

    def prev_frame(self):
        if self.f_handler.tell() < self.start_frame:
            return 0
        self.f_handler.seek(-self.frame_bytes, 1)
        if self.read_data() == 0:
            return 0
        self.frame_no -= 1
        self.last_time -= self.time_step
        return 1

    def rewind(self):
        self.f_handler.seek(self.start_frame, 0)
        self.read_data()
        self.frame_no = 1
        self.last_time = self.time_step

def load_sim_files(filenames):
    global last_time, frame_cnt, fps, current_t, time_step
    global sim_files

    current_t, last_time, frame_cnt, fps = 0, time(), 0, 0
    sim_files = []
    for i,fname in enumerate(filenames):
        if i > 0 and fname == sim_files[i-1].sim_filename:
            sim_files.append(sim_files[i-1])
        else:
            sim_files.append(SimFile(fname,i))

def fprint_stl(*v, file=stdout):
    print("facet normal 0 0 0\nouter loop\nvertex %f %f %f\nvertex %f %f %f\nvertex %f %f %f\nendloop\nendfacet" % (v[0][0],v[0][1],v[0][2],v[1][0],v[1][1],v[1][2],v[2][0],v[2][1],v[2][2]), file=file)

def compute_per_particle_procrustes(org_vertices3, sim_vertices3,
    bondlist, n_bonds, id, T0, T1, R, E, I, scale):
    p_idx = np_append(bondlist[id][0:n_bonds[id]], id)
    P  = org_vertices3[p_idx]
    Q  = sim_vertices3[p_idx]
    if len(p_idx) == 1:
        T0[id] = org_vertices3[id]
        T1[id] = sim_vertices3[id]
        R[id]  = I
        R[id][0][0] = scale[id][0]
        R[id][1][1] = scale[id][1]
        R[id][2][2] = scale[id][2]
        return None, None, None, None, None, None, None, None, None
    Cp = mean(P, axis=0)
    Cq = mean(Q, axis=0)
    if len(p_idx) < 8:
        T0[id] = Cp
        T1[id] = Cq
        R[id]  = I
        return None, None, None, None, None, None, None, None, None
    else:
        Po     = P - Cp
        Qo     = Q - Cq
        M      = Po.T.dot(Qo)
        U,S,V  = svd(M)
        T0[id] = Cp
        T1[id] = Cq
        UV     = U.dot(V)
        R[id]  = UV
    return M, U, S, V, P, Q, Po, Qo, p_idx

def write_to_stl(frame_no=-1,
    filename_format="output_surfaces/extracted_surface%s.stl"):
    n_triangles = sim_files[0].n_triangles
    marched_vertices3 = reshape(marched_vertices, (n_triangles*3, 3))
    if frame_no == -1:
        filename = filename_format % ""
    else:
        filename = filename_format % ("_%03d" % frame_no)

    if write_stl_config is not None:
        if len(write_stl_config) == 2:
            n_objs, n_tets_per_obj = write_stl_config
        else:
            n_objs, n_tets_per_obj = \
                write_stl_config[0], write_stl_config[1:]
            total = 0
            for i in range(len(n_tets_per_obj)):
                n_tets_per_obj[i] += total
                total = n_tets_per_obj[i]
    else:
        n_objs, n_tets_per_obj = 1, 0
    filename_format = filename_format % ("_%03d_%%d" % frame_no)
    acc_cnt = sim_files[0].accum_tri_counts
    for k in range(n_objs):
        f_tri = open(filename_format % k, "w+")
        print("solid triangles", file=f_tri)
        if type(n_tets_per_obj) is not int:
            if k == 0:
                bt, et = 0, acc_cnt[n_tets_per_obj[k]]
            elif k == len(n_tets_per_obj)-1:
                bt, et = acc_cnt[n_tets_per_obj[k-1]], \
                    marched_n_triangles
            else:
                bt, et = acc_cnt[n_tets_per_obj[k-1]], \
                    acc_cnt[n_tets_per_obj[k]]
        elif k < n_objs-1:
            bt, et = acc_cnt[k*n_tets_per_obj], \
                acc_cnt[(k+1)*n_tets_per_obj]
        else:
            bt, et = acc_cnt[k*n_tets_per_obj], marched_n_triangles
        for i in range(bt, et):
            v0 = marched_vertices3[i*3]
            v1 = marched_vertices3[i*3+1]
            v2 = marched_vertices3[i*3+2]
            fprint_stl(v0, v1, v2, file=f_tri)
        print("endsolid triangles", file=f_tri)
        f_tri.close()

def write_to_binary_stl(frame_no=-1,
    filename_format="output_surfaces/extracted_surface%s.stl", write_rest=False):
    if write_rest:
        filename_format="output_surfaces/rest_extracted_surface%s.stl"
    n_triangles = sim_files[0].n_triangles
    marched_vertices3 = reshape(marched_vertices, (n_triangles*3, 3))
    marched_normals3 = reshape(marched_normals, (n_triangles*3, 3))
    if frame_no == -1:
        filename = filename_format % ""
    else:
        filename = filename_format % ("_%03d" % frame_no)

    if write_stl_config is not None:
        if len(write_stl_config) == 2:
            n_objs, n_tets_per_obj = write_stl_config
        else:
            n_objs, n_tets_per_obj = \
                write_stl_config[0], write_stl_config[1:]
            total = 0
            for i in range(len(n_tets_per_obj)):
                n_tets_per_obj[i] += total
                total = n_tets_per_obj[i]
    else:
        n_objs, n_tets_per_obj = 1, 0
    filename_format = filename_format % ("_%03d_%%d" % frame_no)
    acc_cnt = sim_files[0].accum_tri_counts

    for k in range(n_objs):
        f_tri = open(filename_format % k, "wb+")
        if type(n_tets_per_obj) is not int:
            if k == 0:
                bt, et = 0, acc_cnt[n_tets_per_obj[k]]
            elif k == len(n_tets_per_obj)-1:
                bt, et = acc_cnt[n_tets_per_obj[k-1]], \
                    marched_n_triangles
            else:
                bt, et = acc_cnt[n_tets_per_obj[k-1]], \
                    acc_cnt[n_tets_per_obj[k]]
        elif k < n_objs-1:
            bt, et = acc_cnt[k*n_tets_per_obj], \
                acc_cnt[(k+1)*n_tets_per_obj]
        else:
            bt, et = acc_cnt[k*n_tets_per_obj], marched_n_triangles
        n_tris = uint32(et-bt)
        f_tri.write(struct.pack("80sI", b'Python Binary STL Writer', n_tris))
        for i in range(bt, et):
            n  = marched_normals3[i*3]
            v0 = marched_vertices3[i*3]
            v1 = marched_vertices3[i*3+1]
            v2 = marched_vertices3[i*3+2]
            data = [
                n[0], n[1], n[2],
                v0[0], v0[1], v0[2],
                v1[0], v1[1], v1[2],
                v2[0], v2[1], v2[2],
                0
            ]
            f_tri.write(struct.pack("12fH", *data))
        f_tri.close()

def extract_surface():
    global marched_vertices, marched_normals, marched_uvs
    global marched_labels, marched_n_triangles

    s = sim_files[0]
    print("Clock:", int(current_t))
    if label_components or per_component_procrustes:
        n_labels = cuda_helpers.label_particles(s.bondlist,
                s.n_bonds, s.n_particles, s.maxbonds, s.labels,
                s.lab_to_arr_idx, pack_labels=False)
    else:
        n_labels = 1
    print("result: %d labels" % (n_labels))

    n_particles    = s.n_particles
    bond_ratios    = s.bond_ratios
    with errstate(divide='ignore', invalid='ignore'):
        bond_ratios[:] = divide(s.n_bonds, s.start_n_bonds)
        bond_ratios[s.start_n_bonds == 0.] = 0.
    org_vertices3  = reshape(s.org_vertices, (n_particles,3))
    sim_vertices3  = reshape(s.vertices, (n_particles,3))
    bondlist       = reshape(s.bondlist, (s.n_particles, s.maxbonds))
    E = array((0, 0, 0), dtype=float32)
    I = array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float32)

    cuda.memcpy_htod(s.d_vertices, s.vertices)
    cuda.memcpy_htod(s.d_labels, s.labels)
    if not cpu_procrustes:
        if no_procrustes:
            pass
        elif per_component_procrustes:
            cuda_helpers.compute_per_component_procrustes(s.d_org_vertices,
                s.d_vertices, s.n_particles,
                s.d_labels, s.n_particles,   # n_particles -> max_labels
                s.bondlist, s.n_bonds,
                s.maxbonds, s.d_T0, s.d_T1, s.d_R, s.d_A, s.random_scale)
        else:
            cuda_helpers.compute_per_particle_procrustes(s.d_org_vertices,
                s.d_vertices, s.n_particles, s.bondlist, s.n_bonds,
                s.maxbonds, s.d_T0, s.d_T1, s.d_R, s.d_A, s.random_scale)
    else: # CPU procrustes
        random_scale3 = reshape(s.random_scale, (s.n_particles,3))
        print("Computing procrustes...", end="\r", flush=True)
        start_time = clock()
        for l in range(s.n_particles):
            compute_per_particle_procrustes(org_vertices3, sim_vertices3,
                bondlist, s.n_bonds, l, s.T0, s.T1, s.R, E, I,
                random_scale3)
        end_time = clock()
        print("Computing procrustes done in %.4f secs." % \
            (end_time-start_time))
        cuda.memcpy_htod(s.d_T0, s.T0)
        cuda.memcpy_htod(s.d_T1, s.T1)
        cuda.memcpy_htod(s.d_R, s.R)

    start_time = clock()
    if "vase" in mesh_file:
        broken_distance =  0.001   # value for Vase model
    else:
        broken_distance =  0.01*s.p_lambda
    if bond_broken_edge:
        n_triangles = cuda_helpers.bond_broken_edge_tri_counts(
            s.d_org_vertices, s.d_vertices, s.n_active_particles,
            s.bondlist, s.n_bonds, s.maxbonds,
            s.d_labels, s.d_boundary_flags, d_tetras, n_tetras,
            s.d_broken_edges, s.tri_counts, s.accum_tri_counts)
    else:
        n_triangles = cuda_helpers.broken_edge_check_tri_counts(
            s.d_org_vertices, s.d_vertices, s.n_active_particles, s.d_labels,
            s.d_boundary_flags, d_tetras, n_tetras,
            s.d_broken_edges, broken_distance,
            s.tri_counts, s.accum_tri_counts)
    if n_triangles > s.n_triangles:
        if s.n_triangles > 0:
            s.d_tri_vertices.free()
            s.d_tri_normals.free()
            s.d_tri_uvs.free()
            s.d_tri_labels.free()
            del s.tri_vertices
            del s.tri_normals
            del s.tri_labels
        alloc_n_triangles = int(n_triangles*1.1)
        s.tri_vertices = zeros((alloc_n_triangles*3*3), dtype=float32)
        s.tri_normals = zeros((alloc_n_triangles*3*3), dtype=float32)
        s.tri_uvs = zeros((alloc_n_triangles*3*3), dtype=float32)
        s.tri_labels = zeros((alloc_n_triangles*3*3), dtype=float32)
        s.d_tri_vertices = cuda.mem_alloc(s.tri_vertices.size * \
            s.tri_vertices.dtype.itemsize)
        s.d_tri_normals = cuda.mem_alloc(s.tri_normals.size * \
            s.tri_normals.dtype.itemsize)
        s.d_tri_uvs = cuda.mem_alloc(s.tri_uvs.size * \
            s.tri_uvs.dtype.itemsize)
        s.d_tri_labels = cuda.mem_alloc(s.tri_labels.size * \
            s.tri_labels.dtype.itemsize)
        s.n_triangles = alloc_n_triangles

    if write_rest_stl:
        marched_vertices, marched_normals, marched_uvs, marched_labels, \
            marched_n_triangles = \
            cuda_helpers.march_tetra(
                s.d_org_vertices, s.d_vertices, s.n_active_particles,
                s.d_labels, s.d_identity_T0,
                s.d_identity_T1, s.d_identity_R, s.d_A, s.d_boundary_flags,
                d_tetras, n_tetras,
                s.n_bonds, s.d_broken_edges, s.accum_tri_counts,
                n_triangles, int(label_components),
                s.d_tri_vertices, s.d_tri_normals, s.d_tri_uvs, s.d_tri_labels,
                s.tri_vertices, s.tri_normals, s.tri_uvs, s.tri_labels)
        if marched_n_triangles > 0:
            if write_start_frame is not None:
                target = int(current_t / s.time_step)
                if type(write_start_frame) in (list, tuple) and \
                    target in write_start_frame or \
                    type(write_start_frame) is int and \
                    target == write_start_frame or \
                    type(write_start_frame) is int and write_start_frame == -1:
                    write_to_binary_stl(target, write_rest=True)

    if per_component_procrustes:
        marched_vertices, marched_normals, marched_uvs, marched_labels, \
            marched_n_triangles = \
            cuda_helpers.march_tetra_per_component_xform(
                s.d_org_vertices, s.d_vertices, s.n_particles,
                s.d_labels, s.d_T0, s.d_T1, s.d_R, s.d_A,
                s.d_boundary_flags, d_tetras, n_tetras,
                s.n_bonds, s.d_broken_edges, s.accum_tri_counts,
                n_triangles, int(label_components),
                s.d_tri_vertices, s.d_tri_normals, s.d_tri_uvs, s.d_tri_labels,
                s.tri_vertices, s.tri_normals, s.tri_uvs, s.tri_labels)
    else:
        marched_vertices, marched_normals, marched_uvs, marched_labels, \
            marched_n_triangles = \
            cuda_helpers.march_tetra(
                s.d_org_vertices, s.d_vertices, s.n_active_particles,
                s.d_labels, s.d_T0,
                s.d_T1, s.d_R, s.d_A, s.d_boundary_flags,
                d_tetras, n_tetras,
                s.n_bonds, s.d_broken_edges, s.accum_tri_counts,
                n_triangles, int(label_components),
                s.d_tri_vertices, s.d_tri_normals, s.d_tri_uvs, s.d_tri_labels,
                s.tri_vertices, s.tri_normals, s.tri_uvs, s.tri_labels)

    end_time = clock()
    print("surface total: %d triangles in %.4f secs." % \
        (marched_n_triangles, end_time-start_time))

    if marched_n_triangles > 0:
        if write_start_frame is not None:
            target = int(current_t / s.time_step)
            if type(write_start_frame) in (list, tuple) and \
                target in write_start_frame or \
                type(write_start_frame) is int and \
                target == write_start_frame or \
                type(write_start_frame) is int and write_start_frame == -1:
                write_to_binary_stl(target)

def load_mesh_file(input_file):
    global n_vertices, n_tetras, tetras, bounders
    global d_tetras, d_bounders

    try:
        f = open(input_file, "r")
    except:
        print("%s not found!" % input_file)
        exit(0)
    header = f.readline().split()
    data = f.readlines()
    f.close()
    n_vertices = int(header[0])
    n_tetras = int(header[1])
    tetras = zeros((n_tetras, 4), dtype=int32)
    bounders = zeros((n_tetras, 4), dtype=int32)
    for i,v in enumerate(data[n_vertices:]):
        elements = [int(j) for j in v.split()]
        tetras[i][:] = elements[:4]
        bounders[i][:] = elements[4:]
    d_tetras = cuda.mem_alloc(tetras.size * tetras.dtype.itemsize)
    cuda.memcpy_htod(d_tetras, tetras)
    d_bounders = cuda.mem_alloc(bounders.size * bounders.dtype.itemsize)
    cuda.memcpy_htod(d_bounders, bounders)

def prepare_bondlist_info():
    s = sim_files[0]

    int_size = dtype('int32').itemsize
    cur_pos = s.f_handler.tell()
    s.f_handler.seek(0, 2)
    s.eof_pos = s.f_handler.tell()
    frame_no = int(s.eof_pos/s.frame_bytes)-1
    s.f_handler.seek(s.start_frame+frame_no*s.frame_bytes+s.read_bytes, 0)
    buf = s.f_handler.read(int_size*s.n_particles)
    if len(buf) != int_size*s.n_particles:
        print("Cannot seek to the last frame!")
        return
    s.last_n_bonds = zeros(s.n_particles, dtype=int32)
    s.last_n_bonds[:] = fromstring(buf, dtype=int32)
    s.last_n_bonds[:] += 1

    s.f_handler.seek(s.read_bytes+s.read_bytes, 1)  # skip vel and force?
    buf = s.f_handler.read(int_size*s.n_particles*s.maxbonds)
    if len(buf) != int_size*s.n_particles*s.maxbonds:
        print("Cannot seek to the last frame!")
        return
    s.last_bondlist = zeros(s.n_particles*s.maxbonds, dtype=int32)
    s.last_bondlist[:] = fromstring(buf, dtype=int32)
    s.f_handler.seek(cur_pos, 0)

    s.last_labels = zeros(s.n_particles, dtype=int32)
    s.last_lab_to_arr_idx = zeros(s.n_particles, dtype=int32)

    if label_components:
        n_labels = cuda_helpers.label_particles(s.last_bondlist,
            s.last_n_bonds, s.n_particles, s.maxbonds, s.last_labels,
            s.last_lab_to_arr_idx, pack_labels=False)
    else:
        n_labels = 1

    s.T0 = zeros((s.n_particles, 3), dtype=float32)
    s.T1 = zeros((s.n_particles, 3), dtype=float32)
    s.R  = zeros((s.n_particles, 3, 3), dtype=float32)
    s.A  = zeros((s.n_particles, 3, 3), dtype=float32)
    s.random_scale = zeros((s.n_particles*3), dtype=float32)
    s.random_scale[:] = 0.1+0.3*random.rand( \
        s.n_particles*3)
    s.d_A  = cuda.mem_alloc(s.A.size * s.A.dtype.itemsize)
    s.d_T0 = cuda.mem_alloc(s.T0.size * s.T0.dtype.itemsize)
    s.d_T1 = cuda.mem_alloc(s.T1.size * s.T1.dtype.itemsize)
    s.d_R  = cuda.mem_alloc(s.R.size * s.R.dtype.itemsize)
    s.lab_to_arr_idx = zeros(s.n_particles, dtype=int32)
    s.R[:] = array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float32)
    cuda.memcpy_htod(s.d_T0, s.T0)
    cuda.memcpy_htod(s.d_T1, s.T1)
    cuda.memcpy_htod(s.d_R, s.R)
    if mesh_file:
        s.broken_edges = zeros((n_tetras, 6), dtype=uint8)
        s.d_broken_edges = cuda.mem_alloc(s.broken_edges.size * \
            s.broken_edges.dtype.itemsize)
        cuda.memcpy_htod(s.d_broken_edges, s.broken_edges)

    if write_rest_stl:
        s.identity_T0 = zeros((s.n_particles, 3), dtype=float32)
        s.identity_T1 = zeros((s.n_particles, 3), dtype=float32)
        s.identity_R  = zeros((s.n_particles, 3, 3), dtype=float32)
        s.d_identity_T0 = cuda.mem_alloc(s.identity_T0.size * \
            s.T0.dtype.itemsize)
        s.d_identity_T1 = cuda.mem_alloc(s.identity_T1.size * \
            s.T1.dtype.itemsize)
        s.d_identity_R  = cuda.mem_alloc(s.identity_R.size * \
            s.R.dtype.itemsize)
        s.identity_R[:] = array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), \
            dtype=float32)
        cuda.memcpy_htod(s.d_identity_T0, s.identity_T0)
        cuda.memcpy_htod(s.d_identity_T1, s.identity_T1)
        cuda.memcpy_htod(s.d_identity_R, s.identity_R)

    if mesh_file:
        s.tri_counts = zeros((n_tetras), dtype=int32)
        s.accum_tri_counts = zeros((n_tetras), dtype=int32)
        s.d_boundary_flags = cuda.mem_alloc(4 * n_tetras)
        cuda_helpers.find_boundary(d_tetras, d_bounders, n_tetras,
            s.d_boundary_flags)

def initialize(filenames):
    global current_t, frame_no, steps_per_render

    load_sim_files(filenames)

    s = sim_files[0]
    if not s.has_all_info and not no_procrustes:
        print("Bond information NOT found in the .rec file!")
        exit(0)
    elif not mesh_file:
        print("Mesh file NOT found!")
        exit(0)

    steps_per_render = min([ s.time_step for s in sim_files ])
    for s in sim_files:
        s.play(current_t)

    if mesh_file:
        load_mesh_file(mesh_file)
    if sim_files[0].has_all_info:
        prepare_bondlist_info()

    if start_frame is not None and start_frame >= 0:
        current_t = start_frame * steps_per_render
        frame_no  = start_frame
        if start_frame > 0:
            for s in sim_files:
                s.play(current_t)

def help(argv):
    print("Usage: %s [Options] sim_file..." % argv[0])
    print("Options:")
    print("  -mesh_file filename\t")

def main():
    global start_frame, write_start_frame, write_stl_config
    global cpu_procrustes, per_component_procrustes, no_procrustes
    global current_t, bond_broken_edge, write_rest_stl

    try:
        sysname = os.uname().sysname
    except:
        sysname = os.name

    sim_fnames  = []
    int_list    = ("start_frame", "end_frame", )
    str_list    = ("mesh_file", )
    vector_list = ("write_start_frame", "write_stl_config", "n_plates", )
    flag_list   = ("label_components", "cpu_procrustes",
        "per_component_procrustes", "no_procrustes", "bond_broken_edge",
        "write_rest_stl", )
    argv_iter   = iter(sys.argv[1:])
    while True:
        argv = next(argv_iter, None)
        if not argv: break
        elif argv[0] == '-': argv = argv[1:]

        if argv in int_list:
            globals()[argv] = int(next(argv_iter, None))
        elif argv in str_list:
            globals()[argv] = next(argv_iter, None)
        elif argv in vector_list:
            globals()[argv] = eval(next(argv_iter, None))
        elif argv in flag_list:
            globals()[argv] = True
        else:
            sim_fnames.append(argv)

    if len(sim_fnames) == 0:
        print("No simulation file found!")
        help(sys.argv)
        exit(0)

    initialize(sim_fnames)
    while True:
        extract_surface()
        current_t += steps_per_render
        if current_t < 0.0: current_t = 0.0
        for i,s in enumerate(sim_files):
            if i > 0 and s.sim_filename == sim_files[i-1].sim_filename:
                continue
        if int(current_t / sim_files[0].time_step) == end_frame:
            break

if __name__ == "__main__":
    main()

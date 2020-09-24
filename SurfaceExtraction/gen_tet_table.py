from itertools import permutations, product
from numpy import array, float32
from math import sqrt
from sys import stdout

midpoint_codes = {
    (0,): 0,       (1,): 1,       (2,): 2,     (3,): 3,
    (0,1): 4,      (0,2): 5,      (0,3): 6,    (1,0): 7,  (1,2): 8,  (1,3): 9,
    (2,0): 10,     (2,1): 11,     (2,3): 12,   (3,0): 13, (3,1): 14, (3,2): 15,
    (0,1,2): 16,   (0,1,3): 17,   (0,2,1): 18, (0,2,3): 19, (0,3,1): 20,
    (0,3,2): 21,   (1,0,2): 22,   (1,0,3): 23, (1,2,0): 24, (1,2,3): 25,
    (1,3,0): 26,   (1,3,2): 27,   (2,0,1): 28, (2,0,3): 29, (2,1,0): 30,
    (2,1,3): 31,   (2,3,0): 32,   (2,3,1): 33, (3,0,1): 34, (3,0,2): 35,
    (3,1,0): 36,   (3,1,2): 37,   (3,2,0): 38, (3,2,1): 39,
    (0,1,2,3): 40, (1,0,2,3): 41, (2,0,1,3): 42, (3,0,1,2): 43,

    # another set of different weight vertices
    ((0,1),0): 44, ((0,2),0): 45, ((0,3),0): 46, ((1,0),0): 47, ((1,2),0): 48,
    ((1,3),0): 49, ((2,0),0): 50, ((2,1),0): 51, ((2,3),0): 52, ((3,0),0): 53,
    ((3,1),0): 54, ((3,2),0): 55,
    ((0,1,2),0): 56,  ((0,1,3),0): 57,  ((0,2,1),0): 58,  ((0,2,3),0): 59,
    ((0,3,1),0): 60,  ((0,3,2),0): 61,  ((1,0,2),0): 62,  ((1,0,3),0): 63,
    ((1,2,0),0): 64,  ((1,2,3),0): 65,  ((1,3,0),0): 66,  ((1,3,2),0): 67,
    ((2,0,1),0): 68,  ((2,0,3),0): 69,  ((2,1,0),0): 70,  ((2,1,3),0): 71,
    ((2,3,0),0): 72,  ((2,3,1),0): 73,  ((3,0,1),0): 74,  ((3,0,2),0): 75,
    ((3,1,0),0): 76,  ((3,1,2),0): 77,  ((3,2,0),0): 78,  ((3,2,1),0): 79,
    ((0,1,2,3),0): 80, ((1,0,2,3),0): 81, ((2,0,1,3),0): 82,
    ((3,0,1,2),0): 83,

    # for switching dominant weights
    ((0,1,2),1): 84,  ((0,1,3),1): 85,  ((0,2,1),1): 86,  ((0,2,3),1): 87,
    ((0,3,1),1): 88,  ((0,3,2),1): 89,  ((1,0,2),1): 90,  ((1,0,3),1): 91,
    ((1,2,0),1): 92,  ((1,2,3),1): 93,  ((1,3,0),1): 94,  ((1,3,2),1): 95,
    ((2,0,1),1): 96,  ((2,0,3),1): 97,  ((2,1,0),1): 98,  ((2,1,3),1): 99,
    ((2,3,0),1): 100, ((2,3,1),1): 101, ((3,0,1),1): 102, ((3,0,2),1): 103,
    ((3,1,0),1): 104, ((3,1,2),1): 105, ((3,2,0),1): 106, ((3,2,1),1): 107,
}

# main() will automatically insert a value
surf_midpoints = [0 for i in range(len(midpoint_codes))]

# w2_2_a = 4.1/9.
# w2_2_b = 2.45/9.
# w2_2_c = 1.-w2_2_a-w2_2_b
w2_3_a = 3.4/9.
w2_3_b = 2.8/9.
w2_3_c = 1.-w2_3_a-w2_3_b

weight_1 = (1.,)
weight_2 = (5./9.,4./9.)
weight_3 = (3.4/9.,2.8/9.,2.8/9.)
weight_4 = (0.31,0.23,0.23,0.23)

# another set of weights
weight2_1 = (1.,)
weight2_2 = (4./9.,5./9.)
weight2_3 = (1.4/9.,3.8/9.,3.8/9.)   # used in one broken edge cases
weight2_4 = (0.25,0.25,0.25,0.25)

# used for switching the dominant vertex
weight3_3 = (w2_3_b, w2_3_a, w2_3_c)

surf_weights = (
    weight_1, weight_1, weight_1, weight_1,
    weight_2, weight_2, weight_2, weight_2, weight_2, weight_2,
    weight_2, weight_2, weight_2, weight_2, weight_2, weight_2,
    weight_3, weight_3, weight_3, weight_3, weight_3, weight_3,
    weight_3, weight_3, weight_3, weight_3, weight_3, weight_3,
    weight_3, weight_3, weight_3, weight_3, weight_3, weight_3,
    weight_3, weight_3, weight_3, weight_3, weight_3, weight_3,
    weight_4, weight_4, weight_4, weight_4,

    # another set of different weight vertices
    weight2_2, weight2_2, weight2_2, weight2_2, weight2_2, weight2_2,
    weight2_2, weight2_2, weight2_2, weight2_2, weight2_2, weight2_2,
    weight2_3, weight2_3, weight2_3, weight2_3, weight2_3, weight2_3,
    weight2_3, weight2_3, weight2_3, weight2_3, weight2_3, weight2_3,
    weight2_3, weight2_3, weight2_3, weight2_3, weight2_3, weight2_3,
    weight2_3, weight2_3, weight2_3, weight2_3, weight2_3, weight2_3,
    weight2_4, weight2_4, weight2_4, weight2_4,

   # for switching dominant weights
    weight3_3, weight3_3, weight3_3, weight3_3, weight3_3, weight3_3,
    weight3_3, weight3_3, weight3_3, weight3_3, weight3_3, weight3_3,
    weight3_3, weight3_3, weight3_3, weight3_3, weight3_3, weight3_3,
    weight3_3, weight3_3, weight3_3, weight3_3, weight3_3, weight3_3,
)

order_face = ((0,1,2), (0,1,3), (1,2,3), (0,2,3)) # must be in increasing order
face_v_ccw_orient = ((0,2,1), (0,1,3), (1,2,3), (0,3,2))

# ccw vertex orientation of faces
def is_ccw(a, b, c):
    for v in face_v_ccw_orient:
        if (v[0] == a or v[1] == a or v[2] == a) and \
            (v[0] == b or v[1] == b or v[2] == b) and \
            (v[0] == c or v[1] == c or v[2] == c):
            na = v.index(a)
            if (a,b,c) == v[na:] + v[0:na]:
                return 1
    return 0

def correct_face_v_ccw_orient(i,j,k):
    for v in face_v_ccw_orient:
        if (v[0] == i or v[1] == i or v[2] == i) and \
            (v[0] == j or v[1] == j or v[2] == j) and \
            (v[0] == k or v[1] == k or v[2] == k):
            return v[0],v[1],v[2]
    return i,j,k

def check_broken_edges(e, n_broken):
    broken_e = [0, 0, 0, 0, 0, 0]
    e_cnt = 0
    for i in range(len(broken_e)):
        if e[i]:
            broken_e[e_cnt] = i
            e_cnt += 1
    if e_cnt != n_broken:
        print("# of broken edges is not %d!" % n_broken)
        exit(1)
    return broken_e

# -BEF ~ -Broken-Edge Face
# 4 zero-BEFs
def case_0(case_id, e, in_funcs, out_funcs):   # case 0
    broken_e = check_broken_edges(e, 0)
    v3,v0,v1,v2 = 3,0,1,2

    surface = in_funcs[0](v3,v0,v1,v2)
    bnd_surface = [out_funcs[0](v1,v0,v2),
                   out_funcs[0](v3,v0,v1),
                   out_funcs[0](v3,v1,v2),
                   out_funcs[0](v3,v2,v0)]
    bnd_check = [[v1,v0,v2], [v3,v0,v1], [v3,v1,v2], [v3,v2,v0]]
    return case_id, surface, bnd_check, bnd_surface

# Format (e0): (top,left,mid,right)
# 2 zero-BEFs + 2 one-BEFs; the cut will be at (top,mid)
edge1_connected_points = {
    (0,): (0,2,1,3),   # case 32
    (1,): (0,3,2,1),   # case 16
    (2,): (0,1,3,2),   # case 8
    (3,): (1,0,2,3),   # case 4
    (4,): (3,0,1,2),   # case 2
    (5,): (3,1,2,0),   # case 1
}

def case_1(case_id, e, in_funcs, out_funcs):   # case 1, 2, 4, 8, 16, 32
    broken_e = check_broken_edges(e, 1)
    abcd = edge1_connected_points[tuple(broken_e[0:1])]
    v3,v0,v1,v2 = abcd   # top,left,mid,right

    surface = in_funcs[1](v3,v0,v1,v2)
    bnd_surface = [out_funcs[0](v1,v0,v2),
                   out_funcs[1](v3,v0,v1, (0,0,1)),
                   out_funcs[1](v3,v1,v2, (1,0,0)),
                   out_funcs[0](v3,v2,v0)]
    bnd_check = [[v1,v0,v2], [v3,v0,v1], [v3,v1,v2], [v3,v2,v0]]
    return case_id, surface, bnd_check, bnd_surface

# Format (e0,e1,e2,e3): (group no., (top,left,mid,right))
# -BEF ~ -Broken-Edge Face
# Group 0: 4 one-BEFs
#    the cuts will be at (top,mid), (right,left)
# Group 1: 1 zero-BEF + 2 one-BEFs + 1 two-BEF
#    the cuts will be at (top,mid), (top,right)

edge2_connected_points = {
    (0,1): (1, (0,3,2,1)),   # case 48
    (0,2): (1, (0,2,1,3)),   # case 40
    (0,3): (1, (1,3,0,2)),   # case 36
    (0,4): (1, (1,2,3,0)),   # case 34
    (0,5): (0, (3,1,2,0)),   # case 33
    (1,2): (1, (0,1,3,2)),   # case 24
    (1,3): (1, (2,3,1,0)),   # case 20
    (1,4): (0, (0,3,2,1)),   # case 18
    (1,5): (1, (2,1,0,3)),   # case 17
    (2,3): (0, (1,0,2,3)),   # case 12
    (2,4): (1, (3,2,0,1)),   # case 10
    (2,5): (1, (3,1,2,0)),   # case 9
    (3,4): (1, (1,0,2,3)),   # case 6
    (3,5): (1, (2,0,3,1)),   # case 5
    (4,5): (1, (3,0,1,2)),   # case 3
}

# case 3, 5, 6, 9, 10, 12, 17, 18, 20, 24, 33, 34, 36, 40, 48
def case_2(case_id, e, in_funcs, out_funcs):
    broken_e = check_broken_edges(e, 2)
    grp,abcd = edge2_connected_points[tuple(broken_e[0:2])]
    v3,v0,v1,v2 = abcd   # top,left,mid,right

    surface = in_funcs[2](grp, v3,v0,v1,v2)
    if grp == 0:   # case 12, 18, 33
        bnd_surface = [out_funcs[1](v1,v0,v2, (0,1,0)),
                       out_funcs[1](v3,v0,v1, (0,0,1)),
                       out_funcs[1](v3,v1,v2, (1,0,0)),
                       out_funcs[1](v3,v2,v0, (0,1,0))]
    else:   # case 3, 5, 6, 9, 10, 17, 20, 24, 34, 36, 40, 48
        bnd_surface = [out_funcs[0](v1,v0,v2),
                       out_funcs[1](v3,v0,v1, (0,0,1)),
                       out_funcs[2](v3,v1,v2, (1,0,1)),
                       out_funcs[1](v3,v2,v0, (1,0,0))]
    bnd_check = [[v1,v0,v2], [v3,v0,v1], [v3,v1,v2], [v3,v2,v0]]
    return case_id, surface, bnd_check, bnd_surface

# Format (e0,e1,e2,e3): (group no., (top,left,mid,right))
# Group 0: case A of multi-material cases
#    the cuts will be at (top,left), (top,mid), (top,right)
# Group 1: 1 three-BEF + 3 one-BEFs
#    the cuts will be at (top,mid), (top,right), (mid,right)
# Group 2: 2 two-BEFs + 2 one-BEFs
#    the cuts will be at (top,mid), (top,right), (left,mid)
# Group 3: 2 two-BEFs + 2 one-BEFs
#    the cuts will be at (top,mid), (top,right), (right,left)
edge3_connected_points = {
    (0,1,2): (0, (0,1,3,2)),     # case 56
    (0,1,3): (1, (0,3,2,1)),     # case 52
    (0,1,4): (3, (0,3,2,1)),     # case 50
    (0,1,5): (2, (0,3,2,1)),     # case 49
    (0,2,3): (2, (0,2,1,3)),     # case 44
    (0,2,4): (1, (0,2,1,3)),     # case 42
    (0,2,5): (3, (0,2,1,3)),     # case 41
    (0,3,4): (0, (1,0,2,3)),     # case 38
    (0,3,5): (3, (1,3,0,2)),     # case 37
    (0,4,5): (2, (1,2,3,0)),     # case 35
    (1,2,3): (3, (0,1,3,2)),     # case 28
    (1,2,4): (2, (0,1,3,2)),     # case 26
    (1,2,5): (1, (0,1,3,2)),     # case 25
    (1,3,4): (2, (2,3,1,0)),     # case 22
    (1,3,5): (0, (2,0,3,1)),     # case 21
    (1,4,5): (3, (3,0,1,2)),     # case 19
    (2,3,4): (3, (3,2,0,1)),     # case 14
    (2,3,5): (2, (3,1,2,0)),     # case 13
    (2,4,5): (0, (3,0,1,2)),     # case 11
    (3,4,5): (1, (1,0,2,3)),     # case 7
}

def case_3(case_id, e, in_funcs, out_funcs):
    broken_e = check_broken_edges(e, 3)
    grp,abcd = edge3_connected_points[tuple(broken_e[0:3])]
    v3,v0,v1,v2 = abcd   # top,left,mid,right

    surface = in_funcs[3](grp, v3,v0,v1,v2)
    if grp == 0:   # case 11, 21, 38, 56
        # falls into the case_A of multi-material.
        bnd_surface = [out_funcs[0](v1,v0,v2),
                       out_funcs[2](v3,v0,v1, (1,0,1)),
                       out_funcs[2](v3,v1,v2, (1,0,1)),
                       out_funcs[2](v3,v2,v0, (1,0,1))]
    elif grp == 1:   # case 7, 25, 42, 52
        bnd_surface = [out_funcs[1](v1,v0,v2, (0,0,1)),
                       out_funcs[1](v3,v0,v1, (0,0,1)),
                       out_funcs[3](v3,v1,v2),
                       out_funcs[1](v3,v2,v0, (1,0,0))]
    elif grp == 2:   # case 13, 22, 26, 35, 44, 49
        bnd_surface = [out_funcs[1](v1,v0,v2, (1,0,0)),
                       out_funcs[2](v3,v0,v1, (0,1,1)),
                       out_funcs[2](v3,v1,v2, (1,0,1)),
                       out_funcs[1](v3,v2,v0, (1,0,0))]
    elif grp == 3:   # case 14, 19, 28, 37, 41, 50
        bnd_surface = [out_funcs[1](v1,v0,v2, (0,1,0)),
                       out_funcs[1](v3,v0,v1, (0,0,1)),
                       out_funcs[2](v3,v1,v2, (1,0,1)),
                       out_funcs[2](v3,v2,v0, (1,1,0))]
    else:  # shouldn't have reached this case!
        print("Something is wrong here!")
        exit(1)
    bnd_check = [[v1,v0,v2], [v3,v0,v1], [v3,v1,v2], [v3,v2,v0]]
    return case_id, surface, bnd_check, bnd_surface

# Format (e0,e1,e2,e3): (group no., (top,left,mid,right))
# Group 0: case B of multi-material cases
#    the cuts will be at (top,left), (left,mid), (mid,right), (top,right)
# Group 1: case 1 of multi-material cases + one broken edge case (case_1)
#    the cuts will be at (top,left), (top,mid), (top,right), (left,mid)
edge4_connected_points = {
    (0,1,2,3):(1, (0,2,1,3)),   # case 60
    (0,1,2,4):(1, (0,1,3,2)),   # case 58
    (0,1,2,5):(1, (0,3,2,1)),   # case 57
    (0,1,3,4):(1, (1,0,2,3)),   # case 54
    (0,1,3,5):(1, (2,1,0,3)),   # case 53
    (0,1,4,5):(0, (0,1,3,2)),   # case 51
    (0,2,3,4):(1, (1,3,0,2)),   # case 46
    (0,2,3,5):(0, (0,3,2,1)),   # case 45
    (0,2,4,5):(1, (3,0,1,2)),   # case 43
    (0,3,4,5):(1, (1,2,3,0)),   # case 39
    (1,2,3,4):(0, (3,1,2,0)),   # case 30
    (1,2,3,5):(1, (2,0,3,1)),   # case 29
    (1,2,4,5):(1, (3,2,0,1)),   # case 27
    (1,3,4,5):(1, (2,3,1,0)),   # case 23
    (2,3,4,5):(1, (3,1,2,0)),   # case 15
}

def case_4(case_id, e, in_funcs, out_funcs):
    broken_e = check_broken_edges(e, 4)
    grp,abcd = edge4_connected_points[tuple(broken_e[0:4])]
    v3,v0,v1,v2 = abcd   # top,left,mid,right

    surface = in_funcs[4](grp, v3,v0,v1,v2)
    if grp == 0:   # case 30, 45, 51
        # falls into the case_B of multi-material.
        bnd_surface = [out_funcs[2](v1,v0,v2, (1,0,1)),
                       out_funcs[2](v3,v0,v1, (1,1,0)),
                       out_funcs[2](v3,v1,v2, (0,1,1)),
                       out_funcs[2](v3,v2,v0, (1,0,1))]
    elif grp == 1:
        # case 15, 23, 27, 29, 39, 43, 46, 53, 54, 57, 58, 60
        # case_A of multi-material + one broken edge case (case_1)
        bnd_surface = [out_funcs[1](v1,v0,v2, (1,0,0)),
                       out_funcs[3](v3,v0,v1),
                       out_funcs[2](v3,v1,v2, (1,0,1)),
                       out_funcs[2](v3,v2,v0, (1,0,1))]
    else:  # shouldn't have reached this case!
        print("Something is wrong here!")
        exit(1)
    bnd_check = [[v1,v0,v2], [v3,v0,v1], [v3,v1,v2], [v3,v2,v0]]
    return case_id, surface, bnd_check, bnd_surface

edge5_connected_points = {
    # (broken edge list): (v_top,v_left,v_mid,v_right)
    (0,1,2,3,4):(0,2,1,3),   # case 62
    (0,1,2,3,5):(0,3,2,1),   # case 61
    (0,1,2,4,5):(3,2,0,1),   # case 59
    (0,1,3,4,5):(2,3,1,0),   # case 55
    (0,2,3,4,5):(3,0,1,2),   # case 47
    (1,2,3,4,5):(3,1,2,0)    # case 31
}

def case_5(case_id, e, in_funcs, out_funcs):   # case 31, 47, 55, 59, 61, 62
    # case_C of muti-material cases
    broken_e = check_broken_edges(e, 5)

    # v* order is (top,left,middle,right)
    # faces (v0,v1,v2) and (v0,v2,v3) have 3 broken edges
    #    faces (v0,v1,v3) and (v2,v1,v3) have 2 broken edges
    v3,v0,v1,v2 = edge5_connected_points[tuple(broken_e[0:5])]

    surface = in_funcs[5](v3,v0,v1,v2)
    bnd_surface = [out_funcs[2](v1,v0,v2, (1,0,1)),
        out_funcs[3](v3,v0,v1),
        out_funcs[3](v3,v1,v2),
        out_funcs[2](v3,v2,v0, (1,0,1))]
    bnd_check = [[v1,v0,v2], [v3,v0,v1], [v3,v1,v2], [v3,v2,v0]]
    return case_id, surface, bnd_check, bnd_surface

def case_6(case_id, e, in_funcs, out_funcs):   # case 64
    # case_D of multi-material cases
    broken_e = check_broken_edges(e, 6)
    v0,v1,v2,v3 = 0,1,2,3

    surface = in_funcs[6](v3,v0,v1,v2)
    bnd_surface = [out_funcs[3](v0,v2,v1),
                   out_funcs[3](v0,v1,v3),
                   out_funcs[3](v1,v2,v3),
                   out_funcs[3](v0,v3,v2)]
    bnd_check = [[v1,v0,v2], [v3,v0,v1], [v3,v1,v2], [v3,v2,v0]]
    return case_id, surface, bnd_check, bnd_surface

def case_0_inner_surface(l,i,j,k):   # (top,left,mid,right)
    return []

def case_3_inner_surface_2_1(grp, l,i,j,k):  # (group type, top,left,mid,right)
    m = midpoint_codes
    if grp == 1:   # case 7, 25, 42, 52
        mjk = m[(j,k)]
        mjl = m[(j,l)]
        mkj = m[(k,j)]
        mkl = m[(k,l)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mijk = m[((i,j,k),0)]
        mijl = m[((i,j,l),0)]
        mikl = m[((i,k,l),0)]
        mjkl = m[(j,k,l)]
        mkjl = m[(k,j,l)]
        mljk = m[(l,j,k)]
        return [i,mijl,mjkl,  i,mjkl,mijk,  i,mijk,mkjl,  i,mkjl,mikl,
                i,mikl,mljk,  i,mljk,mijl,
                mijl,mjl,mjkl,  mjkl,mjk,mijk,  mijk,mkj,mkjl,  mkjl,mkl,mikl,
                mikl,mlk,mljk,  mljk,mlj,mijl]
    else: # case 11, 13, 14, 19, 21, 22, 26, 28, 35, 37, 38, 41, 44, 49, 50, 56
        return case_3_inner_surface_2_0(grp, l,i,j,k)

def case_4_inner_surface_2_1(grp, l,i,j,k):  # (group type, top,left,mid,right)
    m = midpoint_codes
    if grp == 1:
        # case 15, 23, 27, 29, 39, 43, 46, 53, 54, 57, 58, 60
        # case_A of multi-material + one broken edge case (case_1)
        mij = m[(i,j)]
        mil = m[(i,l)]
        mji = m[(j,i)]
        mjl = m[(j,l)]
        mkl = m[(k,l)]
        mli = m[(l,i)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mijl = m[(i,j,l)]
        mjil = m[(j,i,l)]
        mkij = m[((k,i,j),0)]
        mlij = m[(l,i,j)]
        return [mlij,mli,mlk,  mlij,mlk,mlj,  mkl,mil,mijl,
                mkl,mijl,mkij,  mkij,mjil,mkl,
                mkl,mjil,mjl,  mijl,mij,mkij,  mjil,mkij,mji]
    else:   # case 30, 45, 51
        return case_4_inner_surface_2_0(grp, l,i,j,k)

def case_1_inner_surface_2_0(l,i,j,k):   # (top,left,mid,right)
    m = midpoint_codes
    mjl = m[(j,l)]
    mlj = m[(l,j)]
    mijl = m[((i,j,l),0)]   # put weight towards i
    mkjl = m[((k,j,l),0)]   # put weight towards k
    return [mlj,mijl,mkjl,  mijl,mjl,mkjl]

def case_2_inner_surface_2_0(grp, l,i,j,k):  # (group type, top,left,mid,right)
    m = midpoint_codes
    if grp == 0:   # case 12, 18, 33
        mik = m[(i,k)]
        mjl = m[(j,l)]
        mki = m[(k,i)]
        mlj = m[(l,j)]
        mijl = m[((i,j,l),0)]
        mjik = m[((j,i,k),0)]
        mkjl = m[((k,j,l),0)]
        mlik = m[((l,i,k),0)]
        return [mlj,mijl,mkjl,  mkjl,mijl,mjl,
                mki,mjik,mlik,  mlik,mjik,mik]
    else:   # case 3, 5, 6, 9, 10, 17, 20, 24, 34, 36, 40, 48
        mjl = m[(j,l)]
        mkl = m[(k,l)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mijl = m[((i,j,l),0)]
        mikl = m[((i,k,l),0)]
        return [mlk,mlj,mikl,  mlj,mijl,mikl,
                mikl,mijl,mjl,  mikl,mjl,mkl]

def case_3_inner_surface_2_0(grp, l,i,j,k):  # (group type, top,left,mid,right)
    m = midpoint_codes
    if grp == 0:   # case 11, 21, 38, 56
        # falls into the case_A of multi-material.
        mil = m[(i,l)]
        mjl = m[(j,l)]
        mkl = m[(k,l)]
        mli = m[(l,i)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        return [mli,mlk,mlj,  mil,mjl,mkl]
    elif grp == 1:   # case 7, 25, 42, 52
        mjk = m[(j,k)]
        mjl = m[(j,l)]
        mkj = m[(k,j)]
        mkl = m[(k,l)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mijk = m[((i,j,k),0)]
        mijl = m[((i,j,l),0)]
        mikl = m[((i,k,l),0)]
        mjkl = m[(j,k,l)]
        mkjl = m[(k,j,l)]
        mljk = m[(l,j,k)]
        return [mljk,mlj,mijl,  mjl,mjkl,mijl,  mjkl,mljk,mijl,
                mjkl,mjk,mijk,  mkj,mkjl,mijk,  mkjl,mjkl,mijk,
                mkjl,mkl,mikl,  mlk,mljk,mikl,  mljk,mkjl,mikl,
                mjkl,mkjl,mljk]
    elif grp == 2:   # case 13, 22, 26, 35, 44, 49
        mij = m[(i,j)]
        mji = m[(j,i)]
        mjl = m[(j,l)]
        mkl = m[(k,l)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mikl = m[((i,k,l),0)]
        mkij = m[((k,i,j),0)]
        return [mlj,mij,mkij,  mji,mjl,mkij,
                mlk,mlj,mkij,  mjl,mkl,mkij,
                mikl,mlk,mkij,  mkl,mikl,mkij]
    else:   # case 14, 19, 28, 37, 41, 50
        mik = m[(i,k)]
        mjl = m[(j,l)]
        mki = m[(k,i)]
        mkl = m[(k,l)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mijl = m[((i,j,l),0)]
        mjik = m[((j,i,k),0)]
        return [mlj,mijl,mjik,  mijl,mjl,mjik,
                mlk,mlj,mjik,  mjl,mkl,mjik,
                mik,mlk,mjik,  mkl,mki,mjik]

def case_4_inner_surface_2_0(grp, l,i,j,k):  # (group type, top,left,mid,right)
    m = midpoint_codes
    if grp == 0:   # case 30, 45, 51
        # falls into the case_B of multi-material.
        mij = m[(i,j)]
        mil = m[(i,l)]
        mji = m[(j,i)]
        mjk = m[(j,k)]
        mkj = m[(k,j)]
        mkl = m[(k,l)]
        mli = m[(l,i)]
        mlk = m[(l,k)]
        return [mil,mkj,mkl,  mil,mij,mkj,  mli,mlk,mjk,  mli,mjk,mji]
    else:
        # case 15, 23, 27, 29, 39, 43, 46, 53, 54, 57, 58, 60
        # case_A of multi-material + one broken edge case (case_1)
        mij = m[(i,j)]
        mil = m[(i,l)]
        mji = m[(j,i)]
        mjl = m[(j,l)]
        mkl = m[(k,l)]
        mli = m[(l,i)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mijl = m[(i,j,l)]
        mjil = m[(j,i,l)]
        mkij = m[((k,i,j),0)]
        mlij = m[(l,i,j)]
        return [mlij,mli,mlk,  mlij,mlk,mlj,
                mkl,mil,mijl,  mkl,mijl,mjil,  mkl,mjil,mjl,
                mijl,mij,mkij,  mijl,mkij,mjil,  mjil,mkij,mji]

def case_5_inner_surface_2_0(l,i,j,k):   # (top,left,mid,right)
    m = midpoint_codes
    mij = m[(i,j)]
    mil = m[(i,l)]
    mji = m[(j,i)]
    mjk = m[(j,k)]
    mjl = m[(j,l)]
    mkj = m[(k,j)]
    mkl = m[(k,l)]
    mli = m[(l,i)]
    mlj = m[(l,j)]
    mlk = m[(l,k)]
    mijl = m[(i,j,l)]
    mjil = m[(j,i,l)]
    mjkl = m[(j,k,l)]
    mkjl = m[(k,j,l)]
    mlij = m[(l,i,j)]
    mljk = m[(l,j,k)]
    return [mlj,mlij,mljk,  mlij,mli,mljk,  mljk,mli,mlk,
            mil,mijl,mkjl,  mil,mkjl,mkl,  mijl,mij,mkjl,  mkjl,mij,mkj,
            mjil,mjl,mjkl,  mji,mjil,mjkl,  mji,mjkl,mjk]

def case_6_inner_surface_2_0(l,i,j,k):   # (top,left,mid,right)
    m = midpoint_codes
    mij = m[(i,j)]
    mik = m[(i,k)]
    mil = m[(i,l)]
    mji = m[(j,i)]
    mjk = m[(j,k)]
    mjl = m[(j,l)]
    mki = m[(k,i)]
    mkj = m[(k,j)]
    mkl = m[(k,l)]
    mli = m[(l,i)]
    mlj = m[(l,j)]
    mlk = m[(l,k)]
    mijk = m[(i,j,k)]
    mijl = m[(i,j,l)]
    mikl = m[(i,k,l)]
    mjik = m[(j,i,k)]
    mjil = m[(j,i,l)]
    mjkl = m[(j,k,l)]
    mkij = m[(k,i,j)]
    mkil = m[(k,i,l)]
    mkjl = m[(k,j,l)]
    mlij = m[(l,i,j)]
    mlik = m[(l,i,k)]
    mljk = m[(l,j,k)]
    mijkl = m[(i,j,k,l)]
    mjikl = m[(j,i,k,l)]
    mkijl = m[(k,i,j,l)]
    mlijk = m[(l,i,j,k)]
    surface  = [mij,mijk,mijkl,  mij,mijkl,mijl,  mik,mijkl,mijk,
                mik,mikl,mijkl,  mil,mijl,mijkl,  mil,mijkl,mikl]
    surface += [mji,mjikl,mjik,  mji,mjil,mjikl,  mjk,mjik,mjikl,
                mjk,mjikl,mjkl,  mjl,mjkl,mjikl,  mjl,mjikl,mjil]
    surface += [mkj,mkijl,mkij,  mkj,mkjl,mkijl,  mki,mkij,mkijl,
                mki,mkijl,mkil,  mkl,mkil,mkijl,  mkl,mkijl,mkjl]
    surface += [mlj,mlijk,mljk,  mlj,mlij,mlijk,  mli,mlijk,mlij,
                mli,mlik,mlijk,  mlk,mlijk,mlik,  mlk,mljk,mlijk]
    return surface

# Version 1.5 of inner surface generators
def case_1_inner_surface_1_5(l,i,j,k):   # (top,left,mid,right)
    m = midpoint_codes
    mjl = m[(j,l)]
    mlj = m[(l,j)]
    mijl = m[((i,j,l),0)]   # put weight towards i
    mkjl = m[((k,j,l),0)]   # put weight towards k
    return [mlj,mijl,mkjl,  mijl,mjl,mkjl]

def case_2_inner_surface_1_5(grp, l,i,j,k):  # (group type, top,left,mid,right)
    m = midpoint_codes
    if grp == 0:   # case 12, 18, 33
        mik = m[(i,k)]
        mjl = m[(j,l)]
        mki = m[(k,i)]
        mlj = m[(l,j)]
        mijl = m[((i,j,l),0)]
        mjik = m[((j,i,k),0)]
        mkjl = m[((k,j,l),0)]
        mlik = m[((l,i,k),0)]
        return [mlj,mijl,mkjl,  mkjl,mijl,mjl,
                mki,mjik,mlik,  mlik,mjik,mik]
    else:   # case 3, 5, 6, 9, 10, 17, 20, 24, 34, 36, 40, 48
        mjl = m[(j,l)]
        mkl = m[(k,l)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mijl = m[((i,j,l),0)]
        mikl = m[((i,k,l),0)]
        return [mlk,mlj,mikl,  mlj,mijl,mikl,
                mikl,mijl,mjl,  mikl,mjl,mkl]

def case_3_inner_surface_1_5(grp, l,i,j,k):  # (group type, top,left,mid,right)
    m = midpoint_codes
    if grp == 0:   # case 11, 21, 38, 56
        # falls into the case_A of multi-material.
        mil = m[(i,l)]
        mjl = m[(j,l)]
        mkl = m[(k,l)]
        mli = m[(l,i)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        return [mli,mlk,mlj,  mil,mjl,mkl]
    elif grp == 1:   # case 7, 25, 42, 52
        mjk = m[(j,k)]
        mjl = m[(j,l)]
        mkj = m[(k,j)]
        mkl = m[(k,l)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mijk = m[((i,j,k),0)]
        mijl = m[((i,j,l),0)]
        mikl = m[((i,k,l),0)]
        return [mijl,mijk,mikl,
                mijl,mjl,mjk,  mijl,mjk,mijk,
                mijk,mkj,mkl,  mijk,mkl,mikl,
                mikl,mlk,mlj,  mikl,mlj,mijl]
    elif grp == 2:   # case 13, 22, 26, 35, 44, 49
        mij = m[(i,j)]
        mji = m[(j,i)]
        mjl = m[(j,l)]
        mkl = m[(k,l)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mikl = m[((i,k,l),0)]
        mkij = m[((k,i,j),0)]
        return [mlj,mij,mkij,  mji,mjl,mkij,
                mlk,mlj,mkij,  mjl,mkl,mkij,
                mikl,mlk,mkij,  mkl,mikl,mkij]
    else:   # case 14, 19, 28, 37, 41, 50
        mik = m[(i,k)]
        mjl = m[(j,l)]
        mki = m[(k,i)]
        mkl = m[(k,l)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mijl = m[((i,j,l),0)]
        mjik = m[((j,i,k),0)]
        return [mlj,mijl,mjik,  mijl,mjl,mjik,
                mlk,mlj,mjik,  mjl,mkl,mjik,
                mik,mlk,mjik,  mkl,mki,mjik]

def case_4_inner_surface_1_5(grp, l,i,j,k):  # (group type, top,left,mid,right)
    m = midpoint_codes
    if grp == 0:   # case 30, 45, 51
        # falls into the case_B of multi-material.
        mij = m[(i,j)]
        mil = m[(i,l)]
        mji = m[(j,i)]
        mjk = m[(j,k)]
        mkj = m[(k,j)]
        mkl = m[(k,l)]
        mli = m[(l,i)]
        mlk = m[(l,k)]
        return [mil,mkj,mkl,  mil,mij,mkj,  mli,mlk,mjk,  mli,mjk,mji]
    else:
        # case 15, 23, 27, 29, 39, 43, 46, 53, 54, 57, 58, 60
        # case_A of multi-material + one broken edge case (case_1)
        mij = m[(i,j)]
        mil = m[(i,l)]
        mji = m[(j,i)]
        mjl = m[(j,l)]
        mkl = m[(k,l)]
        mli = m[(l,i)]
        mlj = m[(l,j)]
        mlk = m[(l,k)]
        mkij = m[((k,i,j),0)]
        return [mli,mlk,mlj,
                mkij,mil,mij,  mkij,mkl,mil,  mkij,mjl,mkl,  mkij,mji,mjl]

def case_5_inner_surface_1_5(l,i,j,k):   # (top,left,mid,right)
    m = midpoint_codes
    mij = m[(i,j)]
    mil = m[(i,l)]
    mji = m[(j,i)]
    mjk = m[(j,k)]
    mjl = m[(j,l)]
    mkj = m[(k,j)]
    mkl = m[(k,l)]
    mli = m[(l,i)]
    mlj = m[(l,j)]
    mlk = m[(l,k)]
    return [mli,mlk,mlj,  mjk,mji,mjl,
            mil,mij,mkj,  mil,mkj,mkl]

def case_6_inner_surface_1_5(l,i,j,k):   # (top,left,mid,right)
    m = midpoint_codes
    mij = m[(i,j)]
    mik = m[(i,k)]
    mil = m[(i,l)]
    mji = m[(j,i)]
    mjk = m[(j,k)]
    mjl = m[(j,l)]
    mki = m[(k,i)]
    mkj = m[(k,j)]
    mkl = m[(k,l)]
    mli = m[(l,i)]
    mlj = m[(l,j)]
    mlk = m[(l,k)]
    return [mli,mlk,mlj,  mij,mik,mil,  mji,mjl,mjk,  mki,mkj,mkl]

def null_func(*params):
    return []

def zero_broken_edge_face(i,j,k):
    return [i,j,k]

# (i,j,k) ~ (top,left,right) is a triangular face
# e is an array of broken edge binary
def one_broken_edge_face(i,j,k,e):
    m = midpoint_codes
    if e[0]:
        mij = m[(i,j)]
        mji = m[(j,i)]
        mijk = m[((k,i,j),0)]
        return [k,i,mijk,  mijk,i,mij,  mijk,mji,j,  mijk,j,k]
    elif e[1]:
        mjk = m[(j,k)]
        mkj = m[(k,j)]
        mijk = m[((i,j,k),0)]
        return [i,j,mijk,  mijk,j,mjk,  mijk,mkj,k,  mijk,k,i]
    else:
        mik = m[(i,k)]
        mki = m[(k,i)]
        mijk = m[((j,i,k),0)]
        return [i,j,mijk,  mijk,j,k,  mijk,k,mki,  i,mijk,mik]

def two_broken_edge_face(i,j,k,e):   # e is an array of broken edge binary
    m = midpoint_codes
    if e[0] and e[2]:
        mij = m[(i,j)]
        mji = m[(j,i)]
        mik = m[(i,k)]
        mki = m[(k,i)]
        return [mik,i,mij,  k,mki,mji,  k,mji,j]
    elif e[0] and e[1]:
        mij = m[(i,j)]
        mji = m[(j,i)]
        mjk = m[(j,k)]
        mkj = m[(k,j)]
        return [mji,j,mjk,  i,mij,mkj,  i,mkj,k]
    else:
        mik = m[(i,k)]
        mjk = m[(j,k)]
        mki = m[(k,i)]
        mkj = m[(k,j)]
        return [mki,mkj,k,  mik,j,mjk,  i,j,mik]

# no face mid-points in edge connections
def three_broken_edge_face_1_5(i,j,k):
    m = midpoint_codes
    mij = m[(i,j)]
    mji = m[(j,i)]
    mik = m[(i,k)]
    mki = m[(k,i)]
    mjk = m[(j,k)]
    mkj = m[(k,j)]
    return [i,mij,mik,  mji,j,mjk,  mki,mkj,k]

def three_broken_edge_face_2_0(i,j,k):
    m = midpoint_codes
    mij = m[(i,j)]
    mji = m[(j,i)]
    mik = m[(i,k)]
    mki = m[(k,i)]
    mjk = m[(j,k)]
    mkj = m[(k,j)]
    mijk = m[(i,j,k)]
    mjik = m[(j,i,k)]
    mkij = m[(k,i,j)]
    return [i,mij,mijk,  i,mijk,mik,
            j,mjk,mjik,  j,mjik,mji,
            k,mki,mkij,  k,mkij,mkj]

def fprint_stl(*v, file=stdout):
    print("facet normal 0 0 0\nouter loop\nvertex %f %f %f\nvertex %f %f %f\nvertex %f %f %f\nendloop\nendfacet" % (v[0][0],v[0][1],v[0][2],v[1][0],v[1][1],v[1][2],v[2][0],v[2][1],v[2][2]), file=file)

edge_points = ((0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
in_surface_funcs = (case_0_inner_surface, case_1_inner_surface_2_0,
    case_2_inner_surface_2_0, case_3_inner_surface_2_1,
    case_4_inner_surface_2_1, case_5_inner_surface_2_0,
    case_6_inner_surface_2_0)
out_surface_funcs = (zero_broken_edge_face, one_broken_edge_face,
    two_broken_edge_face, three_broken_edge_face_2_0)
# in_surface_funcs = (null_func, null_func, null_func, null_func, null_func,
#     null_func, null_func)
# in_surface_funcs = (case_0_inner_surface, case_1_inner_surface_1_5,
#     case_2_inner_surface_1_5, case_3_inner_surface_1_5,
#     case_4_inner_surface_1_5, case_5_inner_surface_1_5,
#     case_6_inner_surface_1_5)
# out_surface_funcs = (zero_broken_edge_face, one_broken_edge_face,
#     two_broken_edge_face, three_broken_edge_face_1_5)

def gen_surface(case_id=0):
    edge_len = 2
    stretch = 0.0
    t_midpoints = array([(1/3*sqrt(3)*edge_len, 0, 0),
        (-1/6*sqrt(3)*edge_len, -1/2*edge_len, 0),
        (0, 0, 1/3*sqrt(6)*edge_len), (-1/6*sqrt(3), 1/2*edge_len, 0)], dtype=float32)
    t_midpoints = array([(-1/6*sqrt(3), 1/2*edge_len, 0),
        (-1/6*sqrt(3)*edge_len, -1/2*edge_len, 0),
        (1/3*sqrt(3)*edge_len, 0, 0), (0, 0, 1/3*sqrt(6)*edge_len)], dtype=float32)
    s_midpoints = [0 for i in range(len(midpoint_codes))]
    e = [0, 1, 2, 3, 4, 5]
    for v in range(len(midpoint_codes)):
        s_midpoints[v] = array((0,0,0), dtype=float32)
        for l in range(1, surf_midpoints[v][0]+1):
            s_midpoints[v] += surf_weights[v][l-1] * \
                            t_midpoints[surf_midpoints[v][l]]
    T = [array((0,0,0), dtype=float32) for i in range(4)]

    e[0] = (case_id >> 5) & 0b1
    e[1] = (case_id >> 4) & 0b1
    e[2] = (case_id >> 3) & 0b1
    e[3] = (case_id >> 2) & 0b1
    e[4] = (case_id >> 1) & 0b1
    e[5] = (case_id >> 0) & 0b1
    e_cnt = e[0]+e[1]+e[2]+e[3]+e[4]+e[5]

    for i in range(6):
        if e[i] == 0: continue
        ij = edge_points[i]
        #  vij ~ a unit vector used to move i away from j
        vij = (t_midpoints[ij[0]]-t_midpoints[ij[1]])/edge_len
        vji = -vij
        T[ij[0]] += stretch * vij
        T[ij[1]] += stretch * vji

    func = eval("case_%d" % e_cnt)
    ret_id, s,u,v = func(case_id, e, in_surface_funcs, out_surface_funcs)

    for i in range(4):  # loop through faces
        for j in range(4):
            check = list(u[j])
            check.sort()
            if i != j and tuple(check) == order_face[i]:
                u[i], u[j] = u[j], u[i]
                v[i], v[j] = v[j], v[i]
                break
    for k in range(len(s)//3):
        p0 = surf_midpoints[s[3*k]][1]
        p1 = surf_midpoints[s[3*k+1]][1]
        p2 = surf_midpoints[s[3*k+2]][1]
    for i,vv in enumerate(v):
        check = list(u[i])
        check.sort()
        if tuple(check) != order_face[i]:
            print("error!!!!!!", case_id, i, u, e)
            continue
        for k in range(len(vv)//3):
            p0 = surf_midpoints[vv[3*k]][1]
            p1 = surf_midpoints[vv[3*k+1]][1]
            p2 = surf_midpoints[vv[3*k+2]][1]

    x = (case_id % 8 - 3.5)*2.5*edge_len
    y = (case_id // 8 - 3.5)*2.5*edge_len
    for mp in s_midpoints:
        mp += (x, y, 0)
    for k in range(len(s)//3):
        p0 = surf_midpoints[s[3*k]][1]
        p1 = surf_midpoints[s[3*k+1]][1]
        p2 = surf_midpoints[s[3*k+2]][1]
    for i,vv in enumerate(v):
        check = list(u[i])
        check.sort()
        if tuple(check) != order_face[i]:
            print("error!!!!!!", case_id, i, u, e)
            continue
        for k in range(len(vv)//3):
            p0 = surf_midpoints[vv[3*k]][1]
            p1 = surf_midpoints[vv[3*k+1]][1]
            p2 = surf_midpoints[vv[3*k+2]][1]

def print_case(case_id, s, u, v, prefix=""):
    print(prefix, case_id)
    print("surface:", len(s)//3, end=",  ")
    for i in range(len(s)):
        print(s[i], end="%s" % ("\n" if i == len(s)-1 else ", "))
    if len(u) == 0:
        return
    for i in range(4):  # loop through faces
        for j in range(4):
            check = list(u[j])
            check.sort()
            if i != j and tuple(check) == order_face[i]:
                u[i], u[j] = u[j], u[i]
                v[i], v[j] = v[j], v[i]
                break
    print("check: ", end="")
    for i in range(len(u)):
        print("{%d,%d,%d}" % (u[i][0],u[i][1],u[i][2]),
            end="%s" % ("\n" if i == len(u)-1 else ", "))
    print("boundary: ", end="")
    for vv in v:
        if len(vv) == 0: continue
        print("      {%d, " % (len(vv)//3), end="")
        for i in range(len(vv)):
            print(vv[i], end="%s" % ("},\n" if i == len(vv)-1 else ","))
    print("v_compute: ", end="")
    flags = [0 for i in range(len(midpoint_codes))]
    for i in range(len(s)):
        flags[s[i]] = 1
    for vv in v:
        if len(vv) == 0: continue
        for i in range(len(vv)):
            flags[vv[i]] = 1
    cnt = flags.count(1)
    print("      {%d, " % cnt, end="")
    for i in range(len(flags)):
        if flags[i]:
            cnt -= 1
            print(i, end="%s" % ("},\n" if cnt==0 else ","))
    s_cnt = len(s)//3
    v_cnt = 0
    for vv in v:
        if len(vv) == 0: continue
        v_cnt += len(vv)//3
    print("n_triangles:       %d: %d %d" % (s_cnt+v_cnt,s_cnt,v_cnt))
    print()

def gen_code_case(case_id, s, u, v, prefix=""):
    # reorder face checking!
    for i in range(4):  # loop through faces
        for j in range(4):
            check = list(u[j])
            check.sort()
            if i != j and tuple(check) == order_face[i]:
                u[i], u[j] = u[j], u[i]
                v[i], v[j] = v[j], v[i]
                break
    e0 = (case_id >> 5) & 0b1
    e1 = (case_id >> 4) & 0b1
    e2 = (case_id >> 3) & 0b1
    e3 = (case_id >> 2) & 0b1
    e4 = (case_id >> 1) & 0b1
    e5 = (case_id >> 0) & 0b1
    print("// case %02d: %d%d%d%d%d%d" % (case_id,e0,e1,e2,e3,e4,e5))
    print("__constant__ unsigned char surface_%02d[] = {" % case_id)
    print("   %d,  " % (len(s)//3), end="")
    for i in range(len(s)):
        print(s[i], end=", ")
    print("};")
    for i,vv in enumerate(v):
        print("__constant__ unsigned char boundary_surface_%02d_%d[] = {" % \
            (case_id, i))
        print("   %d,  " % (len(vv)//3), end="")
        for i in range(len(vv)):
            print(vv[i], end=", ")
        print("};")
    flags = [0 for i in range(len(midpoint_codes))]
    for i in range(len(s)):
        flags[s[i]] = 1
    for vv in v:
        if len(vv) == 0: continue
        for i in range(len(vv)):
            flags[vv[i]] = 1
    print("__constant__ unsigned char v_compute_%02d[] = {" % case_id)
    print("   %d,  " % flags.count(1), end="")
    for i in range(len(flags)):
        if flags[i]:
            print(i, end=", ")
    print("};")

def print_cases():
    e = [0, 1, 2, 3, 4, 5]
    for i in range(64):
        e[0] = (i >> 5) & 0b1
        e[1] = (i >> 4) & 0b1
        e[2] = (i >> 3) & 0b1
        e[3] = (i >> 2) & 0b1
        e[4] = (i >> 1) & 0b1
        e[5] = (i >> 0) & 0b1
        e_cnt = e[0]+e[1]+e[2]+e[3]+e[4]+e[5]
        if e_cnt in (0,1,2,3,4,5,6):
            func = eval("case_%d" % e_cnt)
            gen_code_case(*func(i, e, in_surface_funcs, out_surface_funcs),
                prefix="case_%d:" % e_cnt)
            gen_surface(i)

def print_64_cases():
    for i in range(64):
        e0 = (i >> 5) & 0b1
        e1 = (i >> 4) & 0b1
        e2 = (i >> 3) & 0b1
        e3 = (i >> 2) & 0b1
        e4 = (i >> 1) & 0b1
        e5 = (i >> 0) & 0b1
        e_cnt = e0+e1+e2+e3+e4+e5
        print("// %02d %d %d%d%d%d%d%d" % (i, e_cnt, e0,e1,e2,e3,e4,e5))

if __name__ == "__main__":
    # generate a list of surf_midpoints
    for k,v in midpoint_codes.items():
        if type(k[0]) is not tuple:
            surf_midpoints[v] = (len(k),) + k
        else:
            surf_midpoints[v] = (len(k[0]),) + k[0]

    print_cases()

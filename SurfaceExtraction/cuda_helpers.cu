#include <stdio.h>

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b)
{
   float3 out;
   out.x = a.x + b.x;
   out.y = a.y + b.y;
   out.z = a.z + b.z;
   return out;
}
__device__ __forceinline__ float3 operator-(const float3& a, const float3& b)
{
   float3 out;
   out.x = a.x - b.x;
   out.y = a.y - b.y;
   out.z = a.z - b.z;
   return out;
}
__device__ __forceinline__ float3 operator*(const double b, const float3& a)
{
   float3 out;
   out.x = a.x * b;
   out.y = a.y * b;
   out.z = a.z * b;
   return out;
}
__device__ __forceinline__ float3& operator+=(float3& lhs, const float3& rhs)
{
   lhs.x += rhs.x;
   lhs.y += rhs.y;
   lhs.z += rhs.z;
   return lhs;
}
__device__ __forceinline__ float3& operator-=(float3& lhs, const float3& rhs)
{
   lhs.x -= rhs.x;
   lhs.y -= rhs.y;
   lhs.z -= rhs.z;
   return lhs;
}
__device__ __forceinline__ float3& operator*=(float3& lhs, const double rhs)
{
   lhs.x *= rhs;
   lhs.y *= rhs;
   lhs.z *= rhs;
   return lhs;
}
__device__ __forceinline__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return invLen * v;
}
__device__ __forceinline__ float3 cross(float3 a, float3 b)
{  float3 out;

   out.x = a.y * b.z - b.y * a.z;
   out.y = a.z * b.x - b.z * a.x;
   out.z = a.x * b.y - b.x * a.y;
   return out;
}

#define EPSILON  1e-5
#define DOUBLE_EPSILON  1e-15

// surface_midpoints and weights are used for computing all points that
// may occur on the marched surface inside a tet. (total points = 44)
__constant__ unsigned char surface_midpoints[][5] = {
   /* {cnt, vertex list} */
   {1, 0},     {1, 1},     {1, 2},     {1, 3},
   {2, 0,1},   {2, 0,2},   {2, 0,3},   {2, 1,0},   {2, 1,2},   {2, 1,3},
   {2, 2,0},   {2, 2,1},   {2, 2,3},   {2, 3,0},   {2, 3,1},   {2, 3,2},
   {3, 0,1,2}, {3, 0,1,3}, {3, 0,2,1}, {3, 0,2,3}, {3, 0,3,1},
   {3, 0,3,2}, {3, 1,0,2}, {3, 1,0,3}, {3, 1,2,0}, {3, 1,2,3},
   {3, 1,3,0}, {3, 1,3,2}, {3, 2,0,1}, {3, 2,0,3}, {3, 2,1,0},
   {3, 2,1,3}, {3, 2,3,0}, {3, 2,3,1}, {3, 3,0,1}, {3, 3,0,2},
   {3, 3,1,0}, {3, 3,1,2}, {3, 3,2,0}, {3, 3,2,1},
   {4, 0,1,2,3},{4, 1,0,2,3},{4, 2,0,1,3},{4, 3,0,1,2},

   // another set of different weight vertices
   {2, 0,1},   {2, 0,2},   {2, 0,3},   {2, 1,0},   {2, 1,2},   {2, 1,3},
   {2, 2,0},   {2, 2,1},   {2, 2,3},   {2, 3,0},   {2, 3,1},   {2, 3,2},
   {3, 0,1,2}, {3, 0,1,3}, {3, 0,2,1}, {3, 0,2,3}, {3, 0,3,1},
   {3, 0,3,2}, {3, 1,0,2}, {3, 1,0,3}, {3, 1,2,0}, {3, 1,2,3},
   {3, 1,3,0}, {3, 1,3,2}, {3, 2,0,1}, {3, 2,0,3}, {3, 2,1,0},
   {3, 2,1,3}, {3, 2,3,0}, {3, 2,3,1}, {3, 3,0,1}, {3, 3,0,2},
   {3, 3,1,0}, {3, 3,1,2}, {3, 3,2,0}, {3, 3,2,1},
   {4, 0,1,2,3},{4, 1,0,2,3},{4, 2,0,1,3},{4, 3,0,1,2},

   // for switching dominant weights
   {3, 0,1,2}, {3, 0,1,3}, {3, 0,2,1}, {3, 0,2,3}, {3, 0,3,1},
   {3, 0,3,2}, {3, 1,0,2}, {3, 1,0,3}, {3, 1,2,0}, {3, 1,2,3},
   {3, 1,3,0}, {3, 1,3,2}, {3, 2,0,1}, {3, 2,0,3}, {3, 2,1,0},
   {3, 2,1,3}, {3, 2,3,0}, {3, 2,3,1}, {3, 3,0,1}, {3, 3,0,2},
   {3, 3,1,0}, {3, 3,1,2}, {3, 3,2,0}, {3, 3,2,1},
};
#define  N_MIDPOINTS  (sizeof(surface_midpoints)/sizeof(surface_midpoints[0]))
#define  w2_2_a    4.1/9.
#define  w2_2_b    2.45/9.
#define  w2_2_c    (1.-w2_2_a-w2_2_b)
#define  w2_3_a    3./9.
#define  w2_3_b    3./9.
#define  w2_3_c    (1.-w2_3_a-w2_3_b)

// __constant__ float weight1_1[] = {1.};
// __constant__ float weight1_2[] = {4.7/9.,4.3/9.};
// __constant__ float weight1_3[] = {3.2/9.,2.9/9.,2.9/9.};
// __constant__ float weight1_4[] = {0.28,0.24,0.24,0.24};
// __constant__ float weight2_1[] = {1.};
// __constant__ float weight2_2[] = {4.7/9.,4.3/9.};
// __constant__ float weight2_3[] = {w2_2_a, w2_2_b, w2_2_c};
// __constant__ float weight2_4[] = {0.25,0.25,0.25,0.25};
// __constant__ float weight3_3[] = {w2_2_b, w2_2_a, w2_2_c};

__constant__ float weight1_1[] = {1.};
__constant__ float weight1_2[] = {4.5/9.,4.5/9.};
__constant__ float weight1_3[] = {3./9.,3./9.,3./9.};
__constant__ float weight1_4[] = {0.25,0.25,0.25,0.25};
// another set of weights
__constant__ float weight2_1[] = {1.};
__constant__ float weight2_2[] = {4.5/9.,4.5/9.};
   // weight2_3 used in one broken edge cases
__constant__ float weight2_3[] = {1.4/9.,3.8/9.,3.8/9.};
__constant__ float weight2_4[] = {0.25,0.25,0.25,0.25};
// used for switching the dominant vertex
__constant__ float weight3_3[] = {w2_3_a, w2_3_b, w2_3_c};

__constant__ float *surface_weights[] = {
   weight1_1, weight1_1, weight1_1, weight1_1,
   weight1_2, weight1_2, weight1_2, weight1_2, weight1_2, weight1_2,
   weight1_2, weight1_2, weight1_2, weight1_2, weight1_2, weight1_2,
   weight1_3, weight1_3, weight1_3, weight1_3, weight1_3, weight1_3,
   weight1_3, weight1_3, weight1_3, weight1_3, weight1_3, weight1_3,
   weight1_3, weight1_3, weight1_3, weight1_3, weight1_3, weight1_3,
   weight1_3, weight1_3, weight1_3, weight1_3, weight1_3, weight1_3,
   weight1_4, weight1_4, weight1_4, weight1_4,

   weight2_2, weight2_2, weight2_2, weight2_2, weight2_2, weight2_2,
   weight2_2, weight2_2, weight2_2, weight2_2, weight2_2, weight2_2,
   weight2_3, weight2_3, weight2_3, weight2_3, weight2_3, weight2_3,
   weight2_3, weight2_3, weight2_3, weight2_3, weight2_3, weight2_3,
   weight2_3, weight2_3, weight2_3, weight2_3, weight2_3, weight2_3,
   weight2_3, weight2_3, weight2_3, weight2_3, weight2_3, weight2_3,
   weight2_4, weight2_4, weight2_4, weight2_4,

   weight3_3, weight3_3, weight3_3, weight3_3, weight3_3, weight3_3,
   weight3_3, weight3_3, weight3_3, weight3_3, weight3_3, weight3_3,
   weight3_3, weight3_3, weight3_3, weight3_3, weight3_3, weight3_3,
   weight3_3, weight3_3, weight3_3, weight3_3, weight3_3, weight3_3,
};

// surface_** give a list of inner triangles.
// boundary_surface_**_# contain lists of boundary triangles.
// v_compute_** give us a list of midpoints needed to be computed for
//    composing surface triangles.
// where ** indicates a case number, and # indicates a face number.
// An order of face numbers follows data in face_check.
// A face_check is used for checking a boundary face.
__constant__ unsigned char face_check[4][3] = {
   {0,2,1}, {0,1,3}, {1,2,3}, {0,3,2}};
#include "surface_array.h"

__constant__ unsigned char *inner_surface[64] = {
   surface_00, surface_01, surface_02, surface_03, surface_04, surface_05,
   surface_06, surface_07, surface_08, surface_09, surface_10, surface_11,
   surface_12, surface_13, surface_14, surface_15, surface_16, surface_17,
   surface_18, surface_19, surface_20, surface_21, surface_22, surface_23,
   surface_24, surface_25, surface_26, surface_27, surface_28, surface_29,
   surface_30, surface_31, surface_32, surface_33, surface_34, surface_35,
   surface_36, surface_37, surface_38, surface_39, surface_40, surface_41,
   surface_42, surface_43, surface_44, surface_45, surface_46, surface_47,
   surface_48, surface_49, surface_50, surface_51, surface_52, surface_53,
   surface_54, surface_55, surface_56, surface_57, surface_58, surface_59,
   surface_60, surface_61, surface_62, surface_63, };
__constant__ unsigned char *boundary_surface[64][4] = {
   boundary_surface_00_0, boundary_surface_00_1, boundary_surface_00_2,
   boundary_surface_00_3, boundary_surface_01_0, boundary_surface_01_1,
   boundary_surface_01_2, boundary_surface_01_3, boundary_surface_02_0,
   boundary_surface_02_1, boundary_surface_02_2, boundary_surface_02_3,
   boundary_surface_03_0, boundary_surface_03_1, boundary_surface_03_2,
   boundary_surface_03_3, boundary_surface_04_0, boundary_surface_04_1,
   boundary_surface_04_2, boundary_surface_04_3, boundary_surface_05_0,
   boundary_surface_05_1, boundary_surface_05_2, boundary_surface_05_3,
   boundary_surface_06_0, boundary_surface_06_1, boundary_surface_06_2,
   boundary_surface_06_3, boundary_surface_07_0, boundary_surface_07_1,
   boundary_surface_07_2, boundary_surface_07_3, boundary_surface_08_0,
   boundary_surface_08_1, boundary_surface_08_2, boundary_surface_08_3,
   boundary_surface_09_0, boundary_surface_09_1, boundary_surface_09_2,
   boundary_surface_09_3, boundary_surface_10_0, boundary_surface_10_1,
   boundary_surface_10_2, boundary_surface_10_3, boundary_surface_11_0,
   boundary_surface_11_1, boundary_surface_11_2, boundary_surface_11_3,
   boundary_surface_12_0, boundary_surface_12_1, boundary_surface_12_2,
   boundary_surface_12_3, boundary_surface_13_0, boundary_surface_13_1,
   boundary_surface_13_2, boundary_surface_13_3, boundary_surface_14_0,
   boundary_surface_14_1, boundary_surface_14_2, boundary_surface_14_3,
   boundary_surface_15_0, boundary_surface_15_1, boundary_surface_15_2,
   boundary_surface_15_3, boundary_surface_16_0, boundary_surface_16_1,
   boundary_surface_16_2, boundary_surface_16_3, boundary_surface_17_0,
   boundary_surface_17_1, boundary_surface_17_2, boundary_surface_17_3,
   boundary_surface_18_0, boundary_surface_18_1, boundary_surface_18_2,
   boundary_surface_18_3, boundary_surface_19_0, boundary_surface_19_1,
   boundary_surface_19_2, boundary_surface_19_3, boundary_surface_20_0,
   boundary_surface_20_1, boundary_surface_20_2, boundary_surface_20_3,
   boundary_surface_21_0, boundary_surface_21_1, boundary_surface_21_2,
   boundary_surface_21_3, boundary_surface_22_0, boundary_surface_22_1,
   boundary_surface_22_2, boundary_surface_22_3, boundary_surface_23_0,
   boundary_surface_23_1, boundary_surface_23_2, boundary_surface_23_3,
   boundary_surface_24_0, boundary_surface_24_1, boundary_surface_24_2,
   boundary_surface_24_3, boundary_surface_25_0, boundary_surface_25_1,
   boundary_surface_25_2, boundary_surface_25_3, boundary_surface_26_0,
   boundary_surface_26_1, boundary_surface_26_2, boundary_surface_26_3,
   boundary_surface_27_0, boundary_surface_27_1, boundary_surface_27_2,
   boundary_surface_27_3, boundary_surface_28_0, boundary_surface_28_1,
   boundary_surface_28_2, boundary_surface_28_3, boundary_surface_29_0,
   boundary_surface_29_1, boundary_surface_29_2, boundary_surface_29_3,
   boundary_surface_30_0, boundary_surface_30_1, boundary_surface_30_2,
   boundary_surface_30_3, boundary_surface_31_0, boundary_surface_31_1,
   boundary_surface_31_2, boundary_surface_31_3, boundary_surface_32_0,
   boundary_surface_32_1, boundary_surface_32_2, boundary_surface_32_3,
   boundary_surface_33_0, boundary_surface_33_1, boundary_surface_33_2,
   boundary_surface_33_3, boundary_surface_34_0, boundary_surface_34_1,
   boundary_surface_34_2, boundary_surface_34_3, boundary_surface_35_0,
   boundary_surface_35_1, boundary_surface_35_2, boundary_surface_35_3,
   boundary_surface_36_0, boundary_surface_36_1, boundary_surface_36_2,
   boundary_surface_36_3, boundary_surface_37_0, boundary_surface_37_1,
   boundary_surface_37_2, boundary_surface_37_3, boundary_surface_38_0,
   boundary_surface_38_1, boundary_surface_38_2, boundary_surface_38_3,
   boundary_surface_39_0, boundary_surface_39_1, boundary_surface_39_2,
   boundary_surface_39_3, boundary_surface_40_0, boundary_surface_40_1,
   boundary_surface_40_2, boundary_surface_40_3, boundary_surface_41_0,
   boundary_surface_41_1, boundary_surface_41_2, boundary_surface_41_3,
   boundary_surface_42_0, boundary_surface_42_1, boundary_surface_42_2,
   boundary_surface_42_3, boundary_surface_43_0, boundary_surface_43_1,
   boundary_surface_43_2, boundary_surface_43_3, boundary_surface_44_0,
   boundary_surface_44_1, boundary_surface_44_2, boundary_surface_44_3,
   boundary_surface_45_0, boundary_surface_45_1, boundary_surface_45_2,
   boundary_surface_45_3, boundary_surface_46_0, boundary_surface_46_1,
   boundary_surface_46_2, boundary_surface_46_3, boundary_surface_47_0,
   boundary_surface_47_1, boundary_surface_47_2, boundary_surface_47_3,
   boundary_surface_48_0, boundary_surface_48_1, boundary_surface_48_2,
   boundary_surface_48_3, boundary_surface_49_0, boundary_surface_49_1,
   boundary_surface_49_2, boundary_surface_49_3, boundary_surface_50_0,
   boundary_surface_50_1, boundary_surface_50_2, boundary_surface_50_3,
   boundary_surface_51_0, boundary_surface_51_1, boundary_surface_51_2,
   boundary_surface_51_3, boundary_surface_52_0, boundary_surface_52_1,
   boundary_surface_52_2, boundary_surface_52_3, boundary_surface_53_0,
   boundary_surface_53_1, boundary_surface_53_2, boundary_surface_53_3,
   boundary_surface_54_0, boundary_surface_54_1, boundary_surface_54_2,
   boundary_surface_54_3, boundary_surface_55_0, boundary_surface_55_1,
   boundary_surface_55_2, boundary_surface_55_3, boundary_surface_56_0,
   boundary_surface_56_1, boundary_surface_56_2, boundary_surface_56_3,
   boundary_surface_57_0, boundary_surface_57_1, boundary_surface_57_2,
   boundary_surface_57_3, boundary_surface_58_0, boundary_surface_58_1,
   boundary_surface_58_2, boundary_surface_58_3, boundary_surface_59_0,
   boundary_surface_59_1, boundary_surface_59_2, boundary_surface_59_3,
   boundary_surface_60_0, boundary_surface_60_1, boundary_surface_60_2,
   boundary_surface_60_3, boundary_surface_61_0, boundary_surface_61_1,
   boundary_surface_61_2, boundary_surface_61_3, boundary_surface_62_0,
   boundary_surface_62_1, boundary_surface_62_2, boundary_surface_62_3,
   boundary_surface_63_0, boundary_surface_63_1, boundary_surface_63_2,
   boundary_surface_63_3, };
__constant__ unsigned char *v_compute[64] = {
   v_compute_00, v_compute_01, v_compute_02, v_compute_03, v_compute_04,
   v_compute_05, v_compute_06, v_compute_07, v_compute_08, v_compute_09,
   v_compute_10, v_compute_11, v_compute_12, v_compute_13, v_compute_14,
   v_compute_15, v_compute_16, v_compute_17, v_compute_18, v_compute_19,
   v_compute_20, v_compute_21, v_compute_22, v_compute_23, v_compute_24,
   v_compute_25, v_compute_26, v_compute_27, v_compute_28, v_compute_29,
   v_compute_30, v_compute_31, v_compute_32, v_compute_33, v_compute_34,
   v_compute_35, v_compute_36, v_compute_37, v_compute_38, v_compute_39,
   v_compute_40, v_compute_41, v_compute_42, v_compute_43, v_compute_44,
   v_compute_45, v_compute_46, v_compute_47, v_compute_48, v_compute_49,
   v_compute_50, v_compute_51, v_compute_52, v_compute_53, v_compute_54,
   v_compute_55, v_compute_56, v_compute_57, v_compute_58, v_compute_59,
   v_compute_60, v_compute_61, v_compute_62, v_compute_63, };

__constant__ unsigned char edge_connected_points[12] = {
   0,1,  0,2,  0,3,  1,2,  1,3,  2,3 };

// edge_cut_points contain a list of (x,y) where x indicates the "edge" element
// number in surface_midpoints that has a y end-point vertices.
// y refers to an element in edge_connected_points; y >= 6 indicates the
// reverse order of the end points.
__constant__ unsigned char edge_cut_points[][2] = {
   { 4, 0},   { 5, 1},   { 6, 2},   { 7, 6},   { 8, 3},   { 9, 4},
   {10, 7},   {11, 9},   {12, 5},   {13, 8},   {14,10},   {15,11},
   {44, 0},   {45, 1},   {46, 2},   {47, 6},   {48, 3},   {49, 4},
   {50, 7},   {51, 9},   {52, 5},   {53, 8},   {54,10},   {55,11}};

__device__ __forceinline__ void vertex_xform(float3 *vout,
   float3 vin, float3 T0, float3 T1, float *R)
{  float3 t;

   t = vin;
   t -= T0;
   (*vout).x = R[0]*t.x + R[3]*t.y + R[6]*t.z;
   (*vout).y = R[1]*t.x + R[4]*t.y + R[7]*t.z;
   (*vout).z = R[2]*t.x + R[5]*t.y + R[8]*t.z;
   (*vout) += T1;
}

__device__ __forceinline__ int is_broken(int b1, int b2, float3 p1, float3 p2,
   float3 q1, float3 q2, float broken_distance)
{  float3 diff;
   float d1, d2;

   if (b1 != b2)
      return 1;
   diff = p1 - p2;
   d1 = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
   diff = q1 - q2;
   d2 = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
   // if ((d2-d1)/d1 < broken_distance)
   //    return 0;
   // else
   //    return 1;
   if (d2-d1 < broken_distance)
      return 0;
   else
      return 1;
}

__device__ __forceinline__ float breakage(float3 p1, float3 p2,
   float3 q1, float3 q2)
{  float3 diff;
   float d1, d2;

   diff = p1 - p2;
   d1 = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
   diff = q1 - q2;
   d2 = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
   //return (d2-d1)/d1;
   return (d2-d1);
}

__device__ __forceinline__ double mat_det(float *m)
{
   return  +(double)m[0]*(m[4]*m[8]-m[7]*m[5])
           -(double)m[1]*(m[3]*m[8]-m[6]*m[5])
           +(double)m[2]*(m[3]*m[7]-m[6]*m[4]);
}

__device__ __forceinline__ void compute_normals(float3 *nout, float3 *vin)
{  float3 e0, e1, n;

   e0 = vin[0] - vin[1];
   e1 = vin[2] - vin[1];
   e0 = normalize(e0);
   e1 = normalize(e1);
   n  = cross(e0, e1);
   nout[0] = n;
   nout[1] = n;
   nout[2] = n;
}

#define DAMAGE  0.01
__device__ __forceinline__ float cut_function(float3 p, float *cut_plane)
{
   return cut_plane[0]*p.x + cut_plane[1]*p.y + cut_plane[2]*p.z +
      cut_plane[3];
   //return 0.25*sin(p.x*2*M_PI)+0.5-p.y;
   //return n.x*p.x + n.y*p.y + n.z*p.z - 0.35;
}

// Compute cut weights between cut_function and tetrahedral edges.
__global__ void compute_cut_weights(float3* __restrict__ positions,
   float3* __restrict__ sim_positions,
   int* __restrict__ tetras, int n_tetras,
   float* __restrict__ cut_weights, unsigned char* __restrict__ broken_edges,
   float* cut_plane)
{  int block_id, idx, j, k, l, part_id;
   float3 t_vertices[4], t_sim_vertices[4], *p, *s;
   unsigned char *e, *ecp;
   float *cw, d1, d2;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_vertices[j] = positions[part_id];
      t_sim_vertices[j] = sim_positions[part_id];
   }

   cw = cut_weights+6*idx;
   e = broken_edges+6*idx;
   p = t_vertices;
   s = t_sim_vertices;
   ecp = edge_connected_points;
   for (j=0; j < 6; j++) {
      k = ecp[2*j];
      l = ecp[2*j+1];
      d1 = cut_function(p[k], cut_plane);
      d2 = cut_function(p[l], cut_plane);
      if (d1 >=0 && d2 < 0 || d1 < 0 && d2 >= 0) {
         cw[j] = -d2/(d1-d2);
         e[j] = 1;
      }
   }
}

// Routine to extract fracture surface with sliding cut-points
__global__ void march_tetra_with_cut_weights(float3* __restrict__ positions,
   float3* __restrict__ sim_positions,
   int* __restrict__ labels,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   unsigned char* __restrict__ boundary_flags,
   int* __restrict__ tetras, int n_tetras,
   int* __restrict__ n_bonds,
   float* __restrict__ cut_weights,
   unsigned char* __restrict__ broken_edges,
   int* __restrict__ accum_tri_counts,
   int label_component,
   float3* __restrict__ out_vertices, float3* __restrict__ out_normals,
   float3* __restrict__ out_uvs,
   int* __restrict__ out_tri_labels)
{  int block_id, idx, j, k, l, m, n;
   int label_id, part_id, point_id;
   int t_labels[4], t_part_ids[4], case_id, tri_counts;
   float3 t_vertices[4], s_vertices[N_MIDPOINTS];
   unsigned char *e, *ecp, *s_idx, **b_idx, *vc_idx, *b;
   float I[9]={1,0,0, 0,1,0, 0,0,1}, *used_R, *cw, w;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   tri_counts = accum_tri_counts[idx];
   b = boundary_flags + 4*idx;
   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_vertices[j] = positions[part_id];
      t_labels[j] = labels[part_id];
      t_part_ids[j] = part_id;
   }

   e  = broken_edges+6*idx;
   cw = cut_weights+6*idx;
   case_id = 0;
   for (j=0; j < 6; j++)
      case_id |= e[j] << (5-j);

   vc_idx = v_compute[case_id];
   s_idx  = inner_surface[case_id];
   b_idx  = boundary_surface[case_id];

   // compute all necessary vertices and mid-points.
   for (j=1; j <= vc_idx[0]; j++) {
      m = vc_idx[j];
      s_vertices[m].x = s_vertices[m].y = s_vertices[m].z = 0;
      for (k=1; k <= surface_midpoints[m][0]; k++)
         s_vertices[m] += surface_weights[m][k-1] *
                             t_vertices[surface_midpoints[m][k]];
   }

   // recompute cut-points.
   ecp = edge_connected_points;
   for (j=0; j < sizeof(edge_cut_points)/sizeof(edge_cut_points[0]); j++) {
      m = edge_cut_points[j][0];
      n = edge_cut_points[j][1];
      if (n >= 6) {
         n -= 6;
         k = ecp[2*n+1];
         l = ecp[2*n];
         w = 1.-cw[n];
      }
      else {
         k = ecp[2*n];
         l = ecp[2*n+1];
         w = cw[n];
      }
      s_vertices[m] = w*t_vertices[k] + (1.-w)*t_vertices[l];
   }

   // Loop through all triangles, one at a time.
   for (k=0; k < s_idx[0]; k++) {
      label_id = t_labels[surface_midpoints[s_idx[1+3*k]][1]];
      m = 3*tri_counts;
      // Loop through all triangle's vertices.
      for (l=0; l < 3; l++) {
         // For s_idx[1+...], we need skip one slot (s_idx[0]),
         //    which indicates a number of triangles.
         // surf_midpoints[s_idx[1+3*k+l]][1] gives a dominant vertex.
         point_id = surface_midpoints[s_idx[1+3*k+l]][1];
         part_id = t_part_ids[point_id];
         used_R = R+9*part_id;
         if (label_component)
            out_tri_labels[m+l] = label_id;
         else
            out_tri_labels[m+l] = 1;  // interior face
         out_uvs[m+l] = s_vertices[s_idx[1+3*k+l]];
         vertex_xform(out_vertices+m+l, s_vertices[s_idx[1+3*k+l]],
            T0[part_id], T1[part_id], used_R);
      }
      compute_normals(out_normals+m, out_vertices+m);
      tri_counts++;
   }

   // Loop through all 4 faces of a tet.
   for (j=0; j < 4; j++) {
      if (b[j]) {
         // Loop through all boundary triangles.
         // b_idx[j] ~ a list of triangles of the face j.
         // b_idx[j][0] ~ a number of triangles of the face j.
         for (k=0; k < b_idx[j][0]; k++) {
            label_id = t_labels[surface_midpoints[b_idx[j][1+3*k]][1]];
            m = 3*tri_counts;
            // Loop through all triangle's vertices.
            for (l=0; l < 3; l++) {
               point_id = surface_midpoints[b_idx[j][1+3*k+l]][1];
               part_id = t_part_ids[point_id];
               used_R = R+9*part_id;
               if (label_component)
                  out_tri_labels[m+l] = label_id;
               else
                  out_tri_labels[m+l] = 0;   // exterior face
               out_uvs[m+l] = s_vertices[b_idx[j][1+3*k+l]];
               vertex_xform(out_vertices+m+l, s_vertices[b_idx[j][1+3*k+l]],
                  T0[part_id], T1[part_id], used_R);
            }
            compute_normals(out_normals+m, out_vertices+m);
            tri_counts++;
         }
      }
   }
}

// Routine to extract fracture surface with sliding cut-points
// Also generating triangle split side of a cutting plane
__global__ void march_tetra_with_cut_weights_and_split(
   float3* __restrict__ positions,
   float3* __restrict__ sim_positions,
   int* __restrict__ labels,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   unsigned char* __restrict__ boundary_flags,
   int* __restrict__ tetras, int n_tetras,
   int* __restrict__ n_bonds,
   float* __restrict__ cut_weights,
   unsigned char* __restrict__ broken_edges,
   int* __restrict__ accum_tri_counts,
   int label_component,
   float3* __restrict__ out_vertices, float3* __restrict__ out_normals,
   float3* __restrict__ out_uvs,
   int* __restrict__ out_tri_labels,
   float* cut_plane, unsigned char* __restrict__ tri_cut_sides)
{  int block_id, idx, j, k, l, m, n, side0_cnt;
   int label_id, part_id, point_id;
   int t_labels[4], t_part_ids[4], case_id, tri_counts;
   float3 t_vertices[4], s_vertices[N_MIDPOINTS];
   unsigned char *e, *ecp, *s_idx, **b_idx, *vc_idx, *b;
   float I[9]={1,0,0, 0,1,0, 0,0,1}, *used_R, *cw, w, dist;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   tri_counts = accum_tri_counts[idx];
   b = boundary_flags + 4*idx;
   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_vertices[j] = positions[part_id];
      t_labels[j] = labels[part_id];
      t_part_ids[j] = part_id;
   }

   e  = broken_edges+6*idx;
   cw = cut_weights+6*idx;
   case_id = 0;
   for (j=0; j < 6; j++)
      case_id |= e[j] << (5-j);

   vc_idx = v_compute[case_id];
   s_idx  = inner_surface[case_id];
   b_idx  = boundary_surface[case_id];

   // compute all necessary vertices and mid-points.
   for (j=1; j <= vc_idx[0]; j++) {
      m = vc_idx[j];
      s_vertices[m].x = s_vertices[m].y = s_vertices[m].z = 0;
      for (k=1; k <= surface_midpoints[m][0]; k++)
         s_vertices[m] += surface_weights[m][k-1] *
                             t_vertices[surface_midpoints[m][k]];
   }

   // recompute cut-points.
   ecp = edge_connected_points;
   for (j=0; j < sizeof(edge_cut_points)/sizeof(edge_cut_points[0]); j++) {
      m = edge_cut_points[j][0];
      n = edge_cut_points[j][1];
      if (n >= 6) {
         n -= 6;
         k = ecp[2*n+1];
         l = ecp[2*n];
         w = 1.-cw[n];
      }
      else {
         k = ecp[2*n];
         l = ecp[2*n+1];
         w = cw[n];
      }
      s_vertices[m] = (w+0.0001)*t_vertices[k] + (1.-w-0.0001)*t_vertices[l];
   }

   // Loop through all triangles, one at a time.
   for (k=0; k < s_idx[0]; k++) {
      label_id = t_labels[surface_midpoints[s_idx[1+3*k]][1]];
      m = 3*tri_counts;
      // Loop through all triangle's vertices.
      side0_cnt = 0;
      for (l=0; l < 3; l++) {
         // For s_idx[1+...], we need skip one slot (s_idx[0]),
         //    which indicates a number of triangles.
         // surf_midpoints[s_idx[1+3*k+l]][1] gives a dominant vertex.
         point_id = surface_midpoints[s_idx[1+3*k+l]][1];
         part_id = t_part_ids[point_id];
         used_R = R+9*part_id;
         if (label_component)
            out_tri_labels[m+l] = label_id;
         else
            out_tri_labels[m+l] = 1;  // interior face
         out_uvs[m+l] = s_vertices[s_idx[1+3*k+l]];
         dist = cut_function(s_vertices[s_idx[1+3*k+l]], cut_plane);
         if (dist < 0)
            side0_cnt++;
         vertex_xform(out_vertices+m+l, s_vertices[s_idx[1+3*k+l]],
            T0[part_id], T1[part_id], used_R);
      }
      if (side0_cnt > 0)
         tri_cut_sides[m] = tri_cut_sides[m+1] = tri_cut_sides[m+2] = 0;
      else
         tri_cut_sides[m] = tri_cut_sides[m+1] = tri_cut_sides[m+2] = 1;
      compute_normals(out_normals+m, out_vertices+m);
      tri_counts++;
   }

   // Loop through all 4 faces of a tet.
   for (j=0; j < 4; j++) {
      if (b[j]) {
         // Loop through all boundary triangles.
         // b_idx[j] ~ a list of triangles of the face j.
         // b_idx[j][0] ~ a number of triangles of the face j.
         for (k=0; k < b_idx[j][0]; k++) {
            label_id = t_labels[surface_midpoints[b_idx[j][1+3*k]][1]];
            m = 3*tri_counts;
            // Loop through all triangle's vertices.
            side0_cnt = 0;
            for (l=0; l < 3; l++) {
               point_id = surface_midpoints[b_idx[j][1+3*k+l]][1];
               part_id = t_part_ids[point_id];
               used_R = R+9*part_id;
               if (label_component)
                  out_tri_labels[m+l] = label_id;
               else
                  out_tri_labels[m+l] = 0;   // exterior face
               out_uvs[m+l] = s_vertices[b_idx[j][1+3*k+l]];
               dist = cut_function(s_vertices[b_idx[j][1+3*k+l]], cut_plane);
               if (dist < 0)
                  side0_cnt++;
               vertex_xform(out_vertices+m+l, s_vertices[b_idx[j][1+3*k+l]],
                  T0[part_id], T1[part_id], used_R);
            }
            if (side0_cnt > 0)
               tri_cut_sides[m] = tri_cut_sides[m+1] = tri_cut_sides[m+2] = 0;
            else
               tri_cut_sides[m] = tri_cut_sides[m+1] = tri_cut_sides[m+2] = 1;
            compute_normals(out_normals+m, out_vertices+m);
            tri_counts++;
         }
      }
   }
}

// similar to march_tetra() except combining all routines into one function.
__global__ void march_tetra_cuts(float3* __restrict__ positions,
   float3* __restrict__ sim_positions,
   int* __restrict__ labels,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   // tetras contains a list of particle ids of tets.
   // bounders contains a list of opposite tet ids of a tet's vertex?
   int* __restrict__ tetras, int* __restrict__ bounders,
   int n_tetras, int* __restrict__ n_bonds,
   float* __restrict__ cut_weights,
   unsigned char* __restrict__ broken_edges, float broken_distance,
   int label_component,
   float3* __restrict__ out_vertices, int* __restrict__ tri_counts,
   int* __restrict__ out_tri_labels,
   int tris_per_tet)
{  int block_id, idx, j, k, l, m, n, bnd_found;
   int label_id, part_id, point_id;
   int t_labels[4], t_part_ids[4], case_id, *t, bnd0, bnd1, bnd2;
   int chk_list[12], chk_count;
   float3 t_vertices[4], t_sim_vertices[4], s_vertices[N_MIDPOINTS], *p, *s;
   unsigned char *e, *ecp, *s_idx, **b_idx, *vc_idx;
   float I[9]={1,0,0, 0,1,0, 0,0,1}, *used_R, *cw, w;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   tri_counts[idx] = 0;
   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_vertices[j] = positions[part_id];
      t_sim_vertices[j] = sim_positions[part_id];
      t_labels[j] = labels[part_id];
      t_part_ids[j] = part_id;
   }

   cw = cut_weights+6*idx;
   e = broken_edges+6*idx;
   t = t_labels;
   p = t_vertices;
   s = t_sim_vertices;
   ecp = edge_connected_points;

   case_id = 0;
   for (j=0; j < 6; j++) {
      k = ecp[2*j];
      l = ecp[2*j+1];
      e[j] |= is_broken(t[k], t[l], p[k], p[l], s[k], s[l], broken_distance);
      case_id |= e[j] << (5-j);
   }

   vc_idx = v_compute[case_id];
   s_idx  = inner_surface[case_id];
   b_idx  = boundary_surface[case_id];

   // compute all necessary vertices and mid-points.
   for (j=1; j <= vc_idx[0]; j++) {
      m = vc_idx[j];
      s_vertices[m].x = s_vertices[m].y = s_vertices[m].z = 0;
      for (k=1; k <= surface_midpoints[m][0]; k++)
         s_vertices[m] += surface_weights[m][k-1] *
                             t_vertices[surface_midpoints[m][k]];
   }

   // recompute cut-points.
   for (j=0; j < sizeof(edge_cut_points)/sizeof(edge_cut_points[0]); j++) {
      m = edge_cut_points[j][0];
      n = edge_cut_points[j][1];
      if (n >= 6) {
         n -= 6;
         k = ecp[2*n+1];
         l = ecp[2*n];
         w = 1.-cw[n];
      }
      else {
         k = ecp[2*n];
         l = ecp[2*n+1];
         w = cw[n];
      }
      s_vertices[m] = (w+0.0001)*t_vertices[k] + (1.-w)*t_vertices[l];
   }

   // Loop through all triangles, one at a time.
   for (k=0; k < s_idx[0]; k++) {
      label_id = t_labels[surface_midpoints[s_idx[1+3*k]][1]];
      m = idx*tris_per_tet*3 + 3*tri_counts[idx];
      // Loop through all triangle's vertices.
      for (l=0; l < 3; l++) {
         // For s_idx[1+...], we need skip one slot (s_idx[0]),
         //    which indicates a number of triangles.
         // surf_midpoints[s_idx[1+3*k+l]][1] gives a dominant vertex.
         point_id = surface_midpoints[s_idx[1+3*k+l]][1];
         part_id = t_part_ids[point_id];
         used_R = R+9*part_id;
         if (label_component)
            out_tri_labels[m+l] = label_id;
         else
            out_tri_labels[m+l] = 1;  // interior face
         vertex_xform(out_vertices+m+l, s_vertices[s_idx[1+3*k+l]],
            T0[part_id], T1[part_id], used_R);
      }
      tri_counts[idx]++;
   }

   chk_count = 0;
   // Loop through all vertices of the tet in the mesh file.
   //    If a boundary value of a vertex is -1, store the corresponding
   //    particle id (assumed to be on a boundary) in chk_list.
   // See also a mesh file format.
   for (j=0; j < 4; j++) {
      if (bounders[idx*4 + j] == -1)
         for (k=0; k < 4; k++)
            if (k != j)
               chk_list[chk_count++] = tetras[idx*4 + k];
   }
   if (chk_count == 0)
      return;
   // Loop through all 4 faces of a tet.
   for (j=0; j < 4; j++) {
      // bnd0,1,2 are particle ids of a face.
      bnd0 = tetras[idx*4 + face_check[j][0]];
      bnd1 = tetras[idx*4 + face_check[j][1]];
      bnd2 = tetras[idx*4 + face_check[j][2]];
      bnd_found = 0;
      // If all 3 particle ids from a face we are about to generate are the
      //    same as ones from the mesh file, these particles are on a boundary.
      for (k=0; k < chk_count; k+=3)
         if ((bnd0==chk_list[k]||bnd0==chk_list[k+1]||bnd0==chk_list[k+2]) &&
             (bnd1==chk_list[k]||bnd1==chk_list[k+1]||bnd1==chk_list[k+2]) &&
             (bnd2==chk_list[k]||bnd2==chk_list[k+1]||bnd2==chk_list[k+2])) {
            bnd_found = 1;
            break;
         }
      // Generating boundary triangles.
      if (bnd_found == 1) {
         // Loop through all boundary triangles.
         // b_idx[j] ~ a list of triangles of the face j.
         // b_idx[j][0] ~ a number of triangles of the face j.
         for (k=0; k < b_idx[j][0]; k++) {
            label_id = t_labels[surface_midpoints[b_idx[j][1+3*k]][1]];
            m = idx*tris_per_tet*3 + 3*tri_counts[idx];
            // Loop through all triangle's vertices.
            for (l=0; l < 3; l++) {
               point_id = surface_midpoints[b_idx[j][1+3*k+l]][1];
               part_id = t_part_ids[point_id];
               used_R = R+9*part_id;
               if (label_component)
                  out_tri_labels[m+l] = label_id;
               else
                  out_tri_labels[m+l] = 0;   // exterior face
               vertex_xform(out_vertices+m+l, s_vertices[b_idx[j][1+3*k+l]],
                  T0[part_id], T1[part_id], used_R);
            }
            tri_counts[idx]++;
         }
      }
   }
}

// Standard routine to extract fracture surface
// using per particle transformation
__global__ void march_tetra(float3* __restrict__ positions,
   float3* __restrict__ sim_positions, int n_active_particles,
   int* __restrict__ labels,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   unsigned char* __restrict__ boundary_flags,
   int* __restrict__ tetras, int n_tetras,
   int* __restrict__ n_bonds,
   unsigned char* __restrict__ broken_edges,
   int* __restrict__ accum_tri_counts,
   int label_component,
   float3* __restrict__ out_vertices, float3* __restrict__ out_normals,
   float3* __restrict__ out_uvs,
   int* __restrict__ out_tri_labels)
{  int block_id, idx, j, k, l, m;
   int label_id, part_id, point_id;
   int t_labels[4], t_part_ids[4], case_id, tri_counts;
   float3 t_vertices[4], s_vertices[N_MIDPOINTS];
   unsigned char *e, *s_idx, **b_idx, *vc_idx, *b;
   float I[9]={1,0,0, 0,1,0, 0,0,1}, *used_R;
   double t_abs_dets[4];

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   tri_counts = accum_tri_counts[idx];
   b = boundary_flags + 4*idx;

   if (tetras[idx*4] >= n_active_particles ||
       tetras[idx*4+1] >= n_active_particles ||
       tetras[idx*4+2] >= n_active_particles ||
       tetras[idx*4+3] >= n_active_particles)
      return;

   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_vertices[j] = positions[part_id];
      t_labels[j] = labels[part_id];
      t_part_ids[j] = part_id;
      t_abs_dets[j] = fabs(mat_det(A+9*part_id));
   }

   e = broken_edges+6*idx;
   case_id = 0;
   for (j=0; j < 6; j++)
      case_id |= e[j] << (5-j);

   vc_idx = v_compute[case_id];
   s_idx  = inner_surface[case_id];
   b_idx  = boundary_surface[case_id];

   // compute all necessary vertices and mid-points.
   for (j=1; j <= vc_idx[0]; j++) {
      m = vc_idx[j];
      s_vertices[m].x = s_vertices[m].y = s_vertices[m].z = 0;
      for (k=1; k <= surface_midpoints[m][0]; k++)
         s_vertices[m] += surface_weights[m][k-1] *
                             t_vertices[surface_midpoints[m][k]];
   }

   // Loop through all triangles, one at a time.
   for (k=0; k < s_idx[0]; k++) {
      label_id = t_labels[surface_midpoints[s_idx[1+3*k]][1]];
      m = 3*tri_counts;
      // Loop through all triangle's vertices.
      for (l=0; l < 3; l++) {
         // For s_idx[1+...], we need skip one slot (s_idx[0]),
         //    which indicates a number of triangles.
         // surf_midpoints[s_idx[1+3*k+l]][1] gives a dominant vertex.
         point_id = surface_midpoints[s_idx[1+3*k+l]][1];
         part_id = t_part_ids[point_id];
         used_R = R+9*part_id;
         // if (n_bonds[part_id] < 2) {
         //    used_R = I;
         //    out_tri_labels[m+l] = -2;
         // }
         // else if (t_abs_dets[point_id] < DOUBLE_EPSILON) {
         //    used_R = I;
         //    out_tri_labels[m+l] = -1;
         // }
         // else {
         //    used_R = R+9*part_id;
         //    out_tri_labels[m+l] = label_id;
         // }
         if (label_component)
            out_tri_labels[m+l] = label_id;
         else
            out_tri_labels[m+l] = 1;  // interior face
         out_uvs[m+l] = s_vertices[s_idx[1+3*k+l]];
         vertex_xform(out_vertices+m+l, s_vertices[s_idx[1+3*k+l]],
            T0[part_id], T1[part_id], used_R);
      }
      compute_normals(out_normals+m, out_vertices+m);
      tri_counts++;
   }

   // Loop through all 4 faces of a tet.
   for (j=0; j < 4; j++) {
      if (b[j]) {
         // Loop through all boundary triangles.
         // b_idx[j] ~ a list of triangles of the face j.
         // b_idx[j][0] ~ a number of triangles of the face j.
         for (k=0; k < b_idx[j][0]; k++) {
            label_id = t_labels[surface_midpoints[b_idx[j][1+3*k]][1]];
            m = 3*tri_counts;
            // Loop through all triangle's vertices.
            for (l=0; l < 3; l++) {
               point_id = surface_midpoints[b_idx[j][1+3*k+l]][1];
               part_id = t_part_ids[point_id];
               used_R = R+9*part_id;
               // if (n_bonds[part_id] < 2) {
               //    used_R = I;
               //    out_tri_labels[m+l] = -2;
               // }
               // else if (t_abs_dets[point_id] < DOUBLE_EPSILON) {
               //    used_R = I;
               //    out_tri_labels[m+l] = -1;
               // }
               // else {
               //    used_R = R+9*part_id;
               //    out_tri_labels[m+l] = label_id;
               // }
               if (label_component)
                  out_tri_labels[m+l] = label_id;
               else
                  out_tri_labels[m+l] = 0;   // exterior face
               out_uvs[m+l] = s_vertices[b_idx[j][1+3*k+l]];
               // if (case_id != 0)
               //    out_tri_labels[m+l] = -1;
               vertex_xform(out_vertices+m+l, s_vertices[b_idx[j][1+3*k+l]],
                  T0[part_id], T1[part_id], used_R);
            }
            compute_normals(out_normals+m, out_vertices+m);
            tri_counts++;
         }
      }
   }
}

// Standard routine to extract fracture surface
// using per component transformation
__global__ void march_tetra_per_component_xform(float3* __restrict__ positions,
   float3* __restrict__ sim_positions,
   int* __restrict__ labels,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   unsigned char* __restrict__ boundary_flags,
   int* __restrict__ tetras, int n_tetras,
   int* __restrict__ n_bonds,
   unsigned char* __restrict__ broken_edges,
   int* __restrict__ accum_tri_counts,
   int label_component,
   float3* __restrict__ out_vertices, float3* __restrict__ out_normals,
   float3* __restrict__ out_uvs,
   int* __restrict__ out_tri_labels)
{  int block_id, idx, j, k, l, m;
   int label_id, part_id;
   int t_labels[4], case_id, tri_counts;
   float3 t_vertices[4], s_vertices[N_MIDPOINTS];
   unsigned char *e, *s_idx, **b_idx, *vc_idx, *b;
   float *used_R;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   tri_counts = accum_tri_counts[idx];
   b = boundary_flags + 4*idx;
   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_vertices[j] = positions[part_id];
      t_labels[j] = labels[part_id];
   }

   e = broken_edges+6*idx;
   case_id = 0;
   for (j=0; j < 6; j++)
      case_id |= e[j] << (5-j);

   vc_idx = v_compute[case_id];
   s_idx  = inner_surface[case_id];
   b_idx  = boundary_surface[case_id];

   // compute all necessary vertices and mid-points.
   for (j=1; j <= vc_idx[0]; j++) {
      m = vc_idx[j];
      s_vertices[m].x = s_vertices[m].y = s_vertices[m].z = 0;
      for (k=1; k <= surface_midpoints[m][0]; k++)
         s_vertices[m] += surface_weights[m][k-1] *
                             t_vertices[surface_midpoints[m][k]];
   }

   // Loop through all triangles, one at a time.
   for (k=0; k < s_idx[0]; k++) {
      label_id = t_labels[surface_midpoints[s_idx[1+3*k]][1]];
      m = 3*tri_counts;
      // Loop through all triangle's vertices.
      for (l=0; l < 3; l++) {
         used_R = R+9*label_id;
         if (label_component)
            out_tri_labels[m+l] = label_id;
         else
            out_tri_labels[m+l] = 1;  // interior face
         out_uvs[m+l] = s_vertices[s_idx[1+3*k+l]];
         vertex_xform(out_vertices+m+l, s_vertices[s_idx[1+3*k+l]],
            T0[label_id], T1[label_id], used_R);
      }
      compute_normals(out_normals+m, out_vertices+m);
      tri_counts++;
   }

   // Loop through all 4 faces of a tet.
   for (j=0; j < 4; j++) {
      if (b[j]) {
         for (k=0; k < b_idx[j][0]; k++) {
            label_id = t_labels[surface_midpoints[b_idx[j][1+3*k]][1]];
            m = 3*tri_counts;
            // Loop through all triangle's vertices.
            for (l=0; l < 3; l++) {
               used_R = R+9*label_id;
               if (label_component)
                  out_tri_labels[m+l] = label_id;
               else
                  out_tri_labels[m+l] = 0;   // exterior face
               out_uvs[m+l] = s_vertices[b_idx[j][1+3*k+l]];
               vertex_xform(out_vertices+m+l, s_vertices[b_idx[j][1+3*k+l]],
                  T0[label_id], T1[label_id], used_R);
            }
            compute_normals(out_normals+m, out_vertices+m);
            tri_counts++;
         }
      }
   }
}

// Collect boundary tet faces and mark them in boundary_flags
__global__ void find_boundary(int* __restrict__ tetras,
   int* __restrict__ bounders, int n_tetras,
   unsigned char* __restrict__ boundary_flags)
{  int block_id, idx, j, k;
   int chk_list[12], chk_count, bnd_found, bnd0, bnd1, bnd2;
   unsigned char *b;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   b = boundary_flags + 4*idx;
   b[0] = b[1] = b[2] = b[3] = 0;

   chk_count = 0;
   // Loop through all vertices of the tet in the mesh file.
   //    If a boundary value of a vertex is -1, store the corresponding
   //    particle id (assumed to be on a boundary) in chk_list.
   // See also a mesh file format.
   for (j=0; j < 4; j++) {
      if (bounders[idx*4 + j] == -1)
         for (k=0; k < 4; k++)
            if (k != j)
               chk_list[chk_count++] = tetras[idx*4 + k];
   }
   if (chk_count == 0)
      return;
   // Loop through all 4 faces of a tet.
   for (j=0; j < 4; j++) {
      // bnd0,1,2 are particle ids of a face.
      bnd0 = tetras[idx*4 + face_check[j][0]];
      bnd1 = tetras[idx*4 + face_check[j][1]];
      bnd2 = tetras[idx*4 + face_check[j][2]];
      bnd_found = 0;
      // If all 3 particle ids from a face we are about to generate are the
      //    same as ones from the mesh file, these particles are on a boundary.
      for (k=0; k < chk_count; k+=3)
         if ((bnd0==chk_list[k]||bnd0==chk_list[k+1]||bnd0==chk_list[k+2]) &&
             (bnd1==chk_list[k]||bnd1==chk_list[k+1]||bnd1==chk_list[k+2]) &&
             (bnd2==chk_list[k]||bnd2==chk_list[k+1]||bnd2==chk_list[k+2])) {
            bnd_found = 1;
            break;
         }
      b[j] = bnd_found;
   }
}

// Use geometric distance to determine edge breaking in the tetrahedron.
__global__ void breaking_edge_check(float3* __restrict__ positions,
   float3* __restrict__ sim_positions, int* __restrict__ tetras, int n_tetras,
   unsigned char* __restrict__ breaking_edges,
   unsigned char* __restrict__ broken_edges, float broken_distance)
{  int block_id, idx, j, k, l, part_id;
   float3 t_vertices[4], t_sim_vertices[4], *p, *s;
   unsigned char *e, *b, *ecp;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_vertices[j] = positions[part_id];
      t_sim_vertices[j] = sim_positions[part_id];
   }

   e = breaking_edges+6*idx;
   b = broken_edges+6*idx;
   p = t_vertices;
   s = t_sim_vertices;
   ecp = edge_connected_points;
   for (j=0; j < 6; j++) {
      e[j] = 0;
      if (b[j])  // if this edge was already broken, skip this edge.
         continue;
      k = ecp[2*j];
      l = ecp[2*j+1];
      e[j] = is_broken(0, 0, p[k], p[l], s[k], s[l], broken_distance);
   }
}

// Calculate the number of triangles based on the broken edges.
__global__ void tri_counts(unsigned char* __restrict__ boundary_flags,
   int n_tetras, unsigned char* __restrict__ broken_edges,
   int* __restrict__ tri_counts)
{  int block_id, idx, j, case_id;
   unsigned char *e, *s_idx, **b_idx, *b;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   tri_counts[idx] = 0;
   b = boundary_flags + 4*idx;
   e = broken_edges+6*idx;
   case_id = 0;
   for (j=0; j < 6; j++)
      case_id |= e[j] << (5-j);

   s_idx  = inner_surface[case_id];
   b_idx  = boundary_surface[case_id];

   tri_counts[idx] += s_idx[0];
   for (j=0; j < 4; j++) {
      if (b[j])
         tri_counts[idx] += b_idx[j][0];
   }
}

// Use geometric distance to determine edge breaks in the tetrahedron.
__global__ void broken_edge_check_tri_counts(float3* __restrict__ positions,
   float3* __restrict__ sim_positions, int n_active_particles,
   int* __restrict__ labels,
   unsigned char* __restrict__ boundary_flags,
   int* __restrict__ tetras, int n_tetras,
   unsigned char* __restrict__ broken_edges, float broken_distance,
   int* __restrict__ tri_counts)
{  int block_id, idx, j, k, l, part_id;
   int case_id, *t, t_labels[4];
   float3 t_vertices[4], t_sim_vertices[4], *p, *s;
   unsigned char *e, *ecp, *s_idx, **b_idx, *b;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   tri_counts[idx] = 0;

   if (tetras[idx*4] >= n_active_particles ||
       tetras[idx*4+1] >= n_active_particles ||
       tetras[idx*4+2] >= n_active_particles ||
       tetras[idx*4+3] >= n_active_particles)
      return;

   b = boundary_flags + 4*idx;
   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_vertices[j] = positions[part_id];
      t_sim_vertices[j] = sim_positions[part_id];
      t_labels[j] = labels[part_id];
   }

   e = broken_edges+6*idx;
   t = t_labels;
   p = t_vertices;
   s = t_sim_vertices;
   ecp = edge_connected_points;
   case_id = 0;
   for (j=0; j < 6; j++) {
      k = ecp[2*j];
      l = ecp[2*j+1];
      e[j] |= is_broken(t[k], t[l], p[k], p[l], s[k], s[l], broken_distance);
      case_id |= e[j] << (5-j);
   }

   s_idx  = inner_surface[case_id];
   b_idx  = boundary_surface[case_id];

   tri_counts[idx] += s_idx[0];
   for (j=0; j < 4; j++) {
      if (b[j])
         tri_counts[idx] += b_idx[j][0];
   }
}

// Use bond information to determine edge breaks in the tetrahedron.
__global__ void bond_broken_edge_tri_counts(float3* __restrict__ positions,
   float3* __restrict__ sim_positions, int n_active_particles,
   int* __restrict__ bondlist, int* __restrict__ n_bonds, int maxbonds,
   int* __restrict__ labels,
   unsigned char* __restrict__ boundary_flags,
   int* __restrict__ tetras, int n_tetras,
   unsigned char* __restrict__ broken_edges,
   int* __restrict__ tri_counts)
{  int block_id, idx, j, k, l, m, part_id, source, target;
   int case_id, *t, t_labels[4], t_part_ids[4];
   float3 t_vertices[4], t_sim_vertices[4], *p, *s;
   unsigned char *e, *ecp, *s_idx, **b_idx, *b;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   tri_counts[idx] = 0;

   if (tetras[idx*4] >= n_active_particles ||
       tetras[idx*4+1] >= n_active_particles ||
       tetras[idx*4+2] >= n_active_particles ||
       tetras[idx*4+3] >= n_active_particles)
      return;

   b = boundary_flags + 4*idx;
   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_part_ids[j] = part_id;
      t_vertices[j] = positions[part_id];
      t_sim_vertices[j] = sim_positions[part_id];
      t_labels[j] = labels[part_id];
   }

   e = broken_edges+6*idx;
   p = t_vertices;
   s = t_sim_vertices;
   ecp = edge_connected_points;
   case_id = 0;
   for (j=0; j < 6; j++) {
      k = ecp[2*j];
      l = ecp[2*j+1];
      if (e[j] == 0) {
         source = t_part_ids[k];
         target = t_part_ids[l];
         t = bondlist+source*maxbonds;
         for(m=0; m < n_bonds[source]; m++)
            if (t[m] == target)
               break;
         if (m == n_bonds[source])
            e[j] = 1;
      }
      case_id |= e[j] << (5-j);
   }

   s_idx  = inner_surface[case_id];
   b_idx  = boundary_surface[case_id];

   tri_counts[idx] += s_idx[0];
   for (j=0; j < 4; j++) {
      if (b[j])
         tri_counts[idx] += b_idx[j][0];
   }
}

// similar to march_tetra() except combining all routines into one function.
__global__ void march_tetra_per_particle_xform(float3* __restrict__ positions,
   float3* __restrict__ sim_positions,
   int* __restrict__ labels,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   // tetras contains a list of particle ids of tets.
   // bounders contains a list of opposite tet ids of a tet's vertex?
   int* __restrict__ tetras, int* __restrict__ bounders,
   int n_tetras, int* __restrict__ n_bonds,
   unsigned char* __restrict__ broken_edges, float broken_distance,
   int label_component,
   float3* __restrict__ out_vertices, int* __restrict__ tri_counts,
   int* __restrict__ out_tri_labels,
   int tris_per_tet)
{  int block_id, idx, j, k, l, m, bnd_found;
   int label_id, part_id, point_id;
   int t_labels[4], t_part_ids[4], case_id, *t, bnd0, bnd1, bnd2;
   int chk_list[12], chk_count;
   float3 t_vertices[4], t_sim_vertices[4], s_vertices[N_MIDPOINTS], *p, *s;
   unsigned char *e, *ecp, *s_idx, **b_idx, *vc_idx;
   float I[9]={1,0,0, 0,1,0, 0,0,1}, *used_R;
   double t_abs_dets[4];

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_tetras)
      return;

   tri_counts[idx] = 0;
   for (j=0; j < 4; j++) {
      part_id = tetras[idx*4 + j];
      t_vertices[j] = positions[part_id];
      t_sim_vertices[j] = sim_positions[part_id];
      t_labels[j] = labels[part_id];
      t_part_ids[j] = part_id;
      t_abs_dets[j] = fabs(mat_det(A+9*part_id));
   }

   e = broken_edges+6*idx;
   t = t_labels;
   p = t_vertices;
   s = t_sim_vertices;
   ecp = edge_connected_points;
   case_id = 0;
   for (j=0; j < 6; j++) {
      k = ecp[2*j];
      l = ecp[2*j+1];
      e[j] |= is_broken(t[k], t[l], p[k], p[l], s[k], s[l], broken_distance);
      case_id |= e[j] << (5-j);
   }

   vc_idx = v_compute[case_id];
   s_idx  = inner_surface[case_id];
   b_idx  = boundary_surface[case_id];

#if 0   // use equal weights!
   // give all with equal weights.
   float surf_weights[88][4] = {{1.}, {1.}, {1.}, {1.}, {0.}};
   for (j=0; j < sizeof(surface_midpoints)/sizeof(surface_midpoints[0]);
        j++) {
      for (k=0; k < surface_midpoints[j][0]; k++)
         surf_weights[j][k] = 1./surface_midpoints[j][0];
   }
   // compute all necessary vertices and mid-points.
   for (j=1; j <= vc_idx[0]; j++) {
      m = vc_idx[j];
      s_vertices[m].x = s_vertices[m].y = s_vertices[m].z = 0;
      for (k=1; k <= surface_midpoints[m][0]; k++)
         s_vertices[m] += surf_weights[m][k-1] *
                             t_vertices[surface_midpoints[m][k]];
   }
#else
   // compute all necessary vertices and mid-points.
   for (j=1; j <= vc_idx[0]; j++) {
      m = vc_idx[j];
      s_vertices[m].x = s_vertices[m].y = s_vertices[m].z = 0;
      for (k=1; k <= surface_midpoints[m][0]; k++)
         s_vertices[m] += surface_weights[m][k-1] *
                             t_vertices[surface_midpoints[m][k]];
   }
#endif

   // Loop through all triangles, one at a time.
   for (k=0; k < s_idx[0]; k++) {
      label_id = t_labels[surface_midpoints[s_idx[1+3*k]][1]];
      m = idx*tris_per_tet*3 + 3*tri_counts[idx];
      // Loop through all triangle's vertices.
      for (l=0; l < 3; l++) {
         // For s_idx[1+...], we need skip one slot (s_idx[0]),
         //    which indicates a number of triangles.
         // surf_midpoints[s_idx[1+3*k+l]][1] gives a dominant vertex.
         point_id = surface_midpoints[s_idx[1+3*k+l]][1];
         part_id = t_part_ids[point_id];
         used_R = R+9*part_id;
         // if (n_bonds[part_id] < 2) {
         //    used_R = I;
         //    out_tri_labels[m+l] = -2;
         // }
         // else if (t_abs_dets[point_id] < DOUBLE_EPSILON) {
         //    used_R = I;
         //    out_tri_labels[m+l] = -1;
         // }
         // else {
         //    used_R = R+9*part_id;
         //    out_tri_labels[m+l] = label_id;
         // }
         if (label_component)
            out_tri_labels[m+l] = label_id;
         else
            out_tri_labels[m+l] = 1;  // interior face
         vertex_xform(out_vertices+m+l, s_vertices[s_idx[1+3*k+l]],
            T0[part_id], T1[part_id], used_R);
      }
      tri_counts[idx]++;
   }

   chk_count = 0;
   // Loop through all vertices of the tet in the mesh file.
   //    If a boundary value of a vertex is -1, store the corresponding
   //    particle id (assumed to be on a boundary) in chk_list.
   // See also a mesh file format.
   for (j=0; j < 4; j++) {
      if (bounders[idx*4 + j] == -1)
         for (k=0; k < 4; k++)
            if (k != j)
               chk_list[chk_count++] = tetras[idx*4 + k];
   }
   if (chk_count == 0)
      return;
   // Loop through all 4 faces of a tet.
   for (j=0; j < 4; j++) {
      // bnd0,1,2 are particle ids of a face.
      bnd0 = tetras[idx*4 + face_check[j][0]];
      bnd1 = tetras[idx*4 + face_check[j][1]];
      bnd2 = tetras[idx*4 + face_check[j][2]];
      bnd_found = 0;
      // If all 3 particle ids from a face we are about to generate are the
      //    same as ones from the mesh file, these particles are on a boundary.
      for (k=0; k < chk_count; k+=3)
         if ((bnd0==chk_list[k]||bnd0==chk_list[k+1]||bnd0==chk_list[k+2]) &&
             (bnd1==chk_list[k]||bnd1==chk_list[k+1]||bnd1==chk_list[k+2]) &&
             (bnd2==chk_list[k]||bnd2==chk_list[k+1]||bnd2==chk_list[k+2])) {
            bnd_found = 1;
            break;
         }
      // Generating boundary triangles.
      if (bnd_found == 1) {
         // Loop through all boundary triangles.
         // b_idx[j] ~ a list of triangles of the face j.
         // b_idx[j][0] ~ a number of triangles of the face j.
         for (k=0; k < b_idx[j][0]; k++) {
            label_id = t_labels[surface_midpoints[b_idx[j][1+3*k]][1]];
            m = idx*tris_per_tet*3 + 3*tri_counts[idx];
            // Loop through all triangle's vertices.
            for (l=0; l < 3; l++) {
               point_id = surface_midpoints[b_idx[j][1+3*k+l]][1];
               part_id = t_part_ids[point_id];
               used_R = R+9*part_id;
               // if (n_bonds[part_id] < 2) {
               //    used_R = I;
               //    out_tri_labels[m+l] = -2;
               // }
               // else if (t_abs_dets[point_id] < DOUBLE_EPSILON) {
               //    used_R = I;
               //    out_tri_labels[m+l] = -1;
               // }
               // else {
               //    used_R = R+9*part_id;
               //    out_tri_labels[m+l] = label_id;
               // }
               if (label_component)
                  out_tri_labels[m+l] = label_id;
               else
                  out_tri_labels[m+l] = 0;   // exterior face
               vertex_xform(out_vertices+m+l, s_vertices[b_idx[j][1+3*k+l]],
                  T0[part_id], T1[part_id], used_R);
            }
            tri_counts[idx]++;
         }
      }
   }
}

__global__ void pack_vertices_labels(float3* __restrict__ vertices,
   int* __restrict__ tri_counts, int* __restrict__ tri_labels,
   float3* __restrict__ packed_vertices, float3* __restrict__ packed_normals,
   int* __restrict__ packed_labels, int tris_per_cube, int max_index)
{  int i, block_id, idx, cnt, k, l;
   float3 e0, e1, n;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= max_index)
      return;
   cnt = 0;
   for (i=0; i < idx; i++)
      cnt += tri_counts[i];
   for (i=0; i < tri_counts[idx]; i++) {
      k = 3*cnt + 3*i;  l = idx*tris_per_cube*3 + 3*i;
      packed_vertices[k]   = vertices[l];
      packed_vertices[k+1] = vertices[l+1];
      packed_vertices[k+2] = vertices[l+2];
      e0 = packed_vertices[k]   - packed_vertices[k+1];
      e1 = packed_vertices[k+2] - packed_vertices[k+1];
      e0 = normalize(e0);
      e1 = normalize(e1);
      n  = cross(e0, e1);
      packed_normals[k]   = n;
      packed_normals[k+1] = n;
      packed_normals[k+2] = n;
      packed_labels[k]    = tri_labels[l];
      packed_labels[k+1]  = tri_labels[l+1];
      packed_labels[k+2]  = tri_labels[l+2];
   }
}

__global__ void pack_vertices(float3* __restrict__ vertices,
   int* __restrict__ tri_counts, float3* __restrict__ packed_vertices,
   float3* __restrict__ packed_normals, int tris_per_cube, int max_index)
{  int i, block_id, idx, cnt, k, l;
   float3 e0, e1, n;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= max_index)
      return;
   cnt = 0;
   for (i=0; i < idx; i++)
      cnt += tri_counts[i];
   for (i=0; i < tri_counts[idx]; i++) {
      k = 3*cnt + 3*i;  l = idx*tris_per_cube*3 + 3*i;
      packed_vertices[k]   = vertices[l];
      packed_vertices[k+1] = vertices[l+1];
      packed_vertices[k+2] = vertices[l+2];
      e0 = packed_vertices[k]   - packed_vertices[k+1];
      e1 = packed_vertices[k+2] - packed_vertices[k+1];
      e0 = normalize(e0);
      e1 = normalize(e1);
      n  = cross(e0, e1);
      packed_normals[k]   = n;
      packed_normals[k+1] = n;
      packed_normals[k+2] = n;
   }
}

__global__ void label_particles(int* __restrict__ bondlist,
   int* __restrict__ n_bonds, int n_particles, int maxbonds,
   int* __restrict__ labels, int* __restrict__ n_updates)
{  int i, block_id, idx, min_label;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_particles)
      return;

   min_label = labels[idx];
   for (i=0; i < n_bonds[idx]; i++) {
      if (min_label > labels[bondlist[idx*maxbonds+i]])
         min_label = labels[bondlist[idx*maxbonds+i]];
   }
   if (min_label != labels[idx]) {
      atomicAdd(n_updates, 1);
      labels[idx] = min_label;
   }
}

__global__ void mark_labels(int n_particles, int* __restrict__ labels,
   unsigned char* __restrict__ marked_labels)
{  int i, block_id, idx;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_particles)
      return;

   marked_labels[idx] = 0;
   for (i=0; i < n_particles; i++) {
      if (labels[i] == idx) {
         marked_labels[idx] = 1;
         break;
      }
   }
}

__global__ void renum_labels(int n_particles, int* __restrict__ labels,
   unsigned char* __restrict__ marked_labels,
   int* __restrict__ out_labels)
{  int i, block_id, idx, num;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_particles)
      return;

   num = 0;
   for (i=0; i < labels[idx]; i++) {
      if (marked_labels[i] == 0)
         num++;
   }
   out_labels[idx] = labels[idx] - num;
}

/* Taken from http://www.ngs.noaa.gov/gps-toolbox/sp3intrp/svdfit.c */
/* Slightly modified versions of routines from
 * Press, William H., Brian P. Flannery, Saul A Teukolsky and
 *   William T. Vetterling, 1986, "Numerical Recipes: The Art of
 *   Scientific Computing" (Fortran), Cambrigde University Press.
 *
 * svd  on pp. 60-64.
 */
__device__ int svd(float *A, float *W, float *V)
{
    /*
       Give a matrix A, with physical dimensions M by N, this routine computes its singular value decomposition, A = U * W * transpose V. The matrix U replaces A on output. The diagonal matrix of singular values, W, is output as a vector W. The matrix V (not the transpose of V) is output as
       V. M must be greater or equal to N. If it is smaller then A should be filled up to square with zero rows.
    */

    unsigned int M=3, N=3;
    double rv1[3];

    /* Householder reduction to bidiagonal form. */
    int NM;
    double C, F, G = 0.0, H, S, X, Y, Z, Scale = 0.0, ANorm = 0.0, tmp;
    int flag, i, its, j, jj, k, l;

    for( i = 0; i < N; ++i ) {
        l = i + 1;
        rv1[i] = Scale * G;
        G = 0.0;
        S = 0.0;
        Scale = 0.0;
        if( i < M ) {
            for( k = i; k < M; ++k ) {
                Scale = Scale + fabs( A[3*k+i] );
            }
            if( Scale != 0.0 ) {
                for( k = i; k < M; ++k ) {
                    A[3*k+i] = A[3*k+i] / Scale;
                    S = S + A[3*k+i] * A[3*k+i];
                }
                F = A[3*i+i];
                G = sqrt(S);
                if( F > 0.0 ) {
                    G = -G;
                }
                H = F * G - S;
                A[3*i+i] = F - G;
                if( i != (N-1) ) {
                    for( j = l; j < N; ++j ) {
                        S = 0.0;
                        for( k = i; k < M; ++k ) {
                            S = S + A[3*k+i] * A[3*k+j];
                        }
                        F = S / H;
                        for( k = i; k < M; ++k ) {
                            A[3*k+j] = A[3*k+j] + F * A[3*k+i];
                        }
                    }
                }
                for( k = i; k < M; ++k ) {
                    A[3*k+i] = Scale * A[3*k+i];
                }
            }
        }

        W[i] = Scale * G;
        G = 0.0;
        S = 0.0;
        Scale = 0.0;
        if( (i < M) && (i != (N-1)) ) {
            for( k = l; k < N; ++k ) {
                Scale = Scale + fabs( A[3*i+k] );
            }
            if( Scale != 0.0 ) {
                for( k = l; k < N; ++k ) {
                    A[3*i+k] = A[3*i+k] / Scale;
                    S = S + A[3*i+k] * A[3*i+k];
                }
                F = A[3*i+l];
                G = sqrt(S);
                if( F > 0.0 ) {
                    G = -G;
                }
                H = F * G - S;
                A[3*i+l] = F - G;
                for( k = l; k < N; ++k ) {
                    rv1[k] = A[3*i+k] / H;
                }
                if( i != (M-1) ) {
                    for( j = l; j < M; ++j ) {
                        S = 0.0;
                        for( k = l; k < N; ++k ) {
                            S = S + A[3*j+k] * A[3*i+k];
                        }
                        for( k = l; k < N; ++k ) {
                            A[3*j+k] = A[3*j+k] + S * rv1[k];
                        }
                    }
                }
                for( k = l; k < N; ++k ) {
                    A[3*i+k] = Scale * A[3*i+k];
                }
            }
        }
        tmp = fabs( W[i] ) + fabs( rv1[i] );
        if( tmp > ANorm )
            ANorm = tmp;
    }

    /* Accumulation of right-hand transformations. */
    for( i = N-1; i >= 0; --i ) {
        if( i < (N-1) ) {
            if( G != 0.0 ) {
                for( j = l; j < N; ++j ) {
                    V[3*j+i] = (A[3*i+j] / A[3*i+l]) / G;
                }
                for( j = l; j < N; ++j ) {
                    S = 0.0;
                    for( k = l; k < N; ++k ) {
                        S = S + A[3*i+k] * V[3*k+j];
                    }
                    for( k = l; k < N; ++k ) {
                        V[3*k+j] = V[3*k+j] + S * V[3*k+i];
                    }
                }
            }
            for( j = l; j < N; ++j ) {
                V[3*i+j] = 0.0;
                V[3*j+i] = 0.0;
            }
        }
        V[3*i+i] = 1.0;
        G = rv1[i];
        l = i;
    }

    /* Accumulation of left-hand transformations. */
    for( i = N-1; i >= 0; --i ) {
        l = i + 1;
        G = W[i];
        if( i < (N-1) ) {
            for( j = l; j < N; ++j ) {
                A[3*i+j] = 0.0;
            }
        }
        if( G != 0.0 ) {
            G = 1.0 / G;
            if( i != (N-1) ) {
                for( j = l; j < N; ++j ) {
                    S = 0.0;
                    for( k = l; k < M; ++k ) {
                        S = S + A[3*k+i] * A[3*k+j];
                    }
                    F = (S / A[3*i+i]) * G;
                    for( k = i; k < M; ++k ) {
                        A[3*k+j] = A[3*k+j] + F * A[3*k+i];
                    }
                }
            }
            for( j = i; j < M; ++j ) {
                A[3*j+i] = A[3*j+i] * G;
            }
        } else {
            for( j = i; j < M; ++j ) {
                A[3*j+i] = 0.0;
            }
        }
        A[3*i+i] = A[3*i+i] + 1.0;
    }

    /* Diagonalization of the bidiagonal form.
       Loop over singular values. */
    for( k = (N-1); k >= 0; --k ) {
        /* Loop over allowed iterations. */
        for( its = 1; its <= 300; ++its ) {
            /* Test for splitting.
               Note that rv1[0] is always zero. */
            flag = true;
            for( l = k; l >= 0; --l ) {
                NM = l - 1;
                if( (fabs(rv1[l]) + ANorm) == ANorm ) {
                    flag = false;
                    break;
                } else if( (fabs(W[NM]) + ANorm) == ANorm ) {
                    break;
                }
            }

            /* Cancellation of rv1[l], if l > 0; */
            if( flag ) {
                C = 0.0;
                S = 1.0;
                for( i = l; i <= k; ++i ) {
                    F = S * rv1[i];
                    if( (fabs(F) + ANorm) != ANorm ) {
                        G = W[i];
                        H = sqrt( F * F + G * G );
                        W[i] = H;
                        H = 1.0 / H;
                        C = ( G * H );
                        S = -( F * H );
                        for( j = 0; j < M; ++j ) {
                            Y = A[3*j+NM];
                            Z = A[3*j+i];
                            A[3*j+NM] = (Y * C) + (Z * S);
                            A[3*j+i] = -(Y * S) + (Z * C);
                        }
                    }
                }
            }
            Z = W[k];
            /* Convergence. */
            if( l == k ) {
                /* Singular value is made nonnegative. */
                if( Z < 0.0 ) {
                    W[k] = -Z;
                    for( j = 0; j < N; ++j ) {
                        V[3*j+k] = -V[3*j+k];
                    }
                }
                break;
            }

            if( its >= 300 )
                return 0;

            X = W[l];
            NM = k - 1;
            Y = W[NM];
            G = rv1[NM];
            H = rv1[k];
            F = ((Y-Z)*(Y+Z) + (G-H)*(G+H)) / (2.0*H*Y);
            G = sqrt( F * F + 1.0 );
            tmp = G;
            if( F < 0.0 )
                tmp = -tmp;
            F = ((X-Z)*(X+Z) + H*((Y/(F+tmp))-H)) / X;

            /* Next QR transformation. */
            C = 1.0;
            S = 1.0;
            for( j = l; j <= NM; ++j ) {
                i = j + 1;
                G = rv1[i];
                Y = W[i];
                H = S * G;
                G = C * G;
                Z = sqrt( F * F + H * H );
                rv1[j] = Z;
                C = F / Z;
                S = H / Z;
                F = (X * C) + (G * S);
                G = -(X * S) + (G * C);
                H = Y * S;
                Y = Y * C;
                for( jj = 0; jj < N; ++jj ) {
                    X = V[3*jj+j];
                    Z = V[3*jj+i];
                    V[3*jj+j] = (X * C) + (Z * S);
                    V[3*jj+i] = -(X * S) + (Z * C);
                }
                Z = sqrt( F * F + H * H );
                W[j] = Z;

                /* Rotation can be arbitrary if Z = 0. */
                if( Z != 0.0 ) {
                    Z = 1.0 / Z;
                    C = F * Z;
                    S = H * Z;
                }
                F = (C * G) + (S * Y);
                X = -(S * G) + (C * Y);
                for( jj = 0; jj < M; ++jj ) {
                    Y = A[3*jj+j];
                    Z = A[3*jj+i];
                    A[3*jj+j] = (Y * C) + (Z * S);
                    A[3*jj+i] = -(Y * S) + (Z * C);
                }
            }
            rv1[l] = 0.0;
            rv1[k] = F;
            W[k] = X;
        }
    }

    return 1;
}

__device__ void add_inner_product(float *U, float3 Po, float3 Qo)
{
   U[0] += Po.x * Qo.x;
   U[1] += Po.x * Qo.y;
   U[2] += Po.x * Qo.z;
   U[3] += Po.y * Qo.x;
   U[4] += Po.y * Qo.y;
   U[5] += Po.y * Qo.z;
   U[6] += Po.z * Qo.x;
   U[7] += Po.z * Qo.y;
   U[8] += Po.z * Qo.z;
}

__global__ void compute_per_particle_procrustes(
   float3* __restrict__ org_vertices,
   float3* __restrict__ sim_vertices, int n_particles,
   int* __restrict__ bondlist, int* __restrict__ n_bonds, int maxbonds,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   float3* __restrict__  random_scale)
{  int i, block_id, idx;
   float3 Cp, Cq, Po, Qo;
   float3 p0, s0;
   float *M, *N, U[9], S[3], V[9];

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_particles)
      return;

   p0 = org_vertices[idx];
   s0 = sim_vertices[idx];

   // T0[idx] = make_float3(0,0,0);
   // T1[idx] = make_float3(0,0,0);
   // R[idx*9+1] = R[idx*9+2] = R[idx*9+3] = R[idx*9+5] = R[idx*9+6] =
   //    R[idx*9+7] = 0;
   // R[idx*9]   = 1.0;
   // R[idx*9+4] = 1.0;
   // R[idx*9+8] = 1.0;
   // return;

   float limit = 0.1;
   float scale = 0.98;
   if (n_bonds[idx] == 0) {
      T0[idx] = p0;
      T1[idx] = s0;
      // if (R[idx*9] > limit || R[idx*9+4] > limit || R[idx*9+8] > limit) {
      //   R[idx*9+1] = R[idx*9+2] = R[idx*9+3] = R[idx*9+5] = R[idx*9+6] =
      //      R[idx*9+7] = 0;
      //   R[idx*9] *= scale;
      //   R[idx*9+4] *= scale;
      //   R[idx*9+8] *= scale;
      // }
      // R[idx*9] = random_scale[idx].x;
      // R[idx*9+4] = random_scale[idx].y;
      // R[idx*9+8] = random_scale[idx].z;
      return;
   }
   T0[idx] = Cp = p0;
   T1[idx] = Cq = s0;
   if (n_bonds[idx] < 3) {
      // R[idx*9+1] = R[idx*9+2] = R[idx*9+3] = R[idx*9+5] = R[idx*9+6] =
      //    R[idx*9+7] = 0;
      // R[idx*9] = R[idx*9+4] = R[idx*9+8] = 1;
      return;
   }
   M = R+idx*9;
   U[0] = U[1] = U[2] = U[3] = U[4] = U[5] = U[6] = U[7] = U[8] = 0;
   for (i=0; i < n_bonds[idx]; i++) {
      Po = org_vertices[bondlist[idx*maxbonds+i]] - Cp;
      Qo = sim_vertices[bondlist[idx*maxbonds+i]] - Cq;
      add_inner_product(U, Po, Qo);
   }
   N = A+idx*9;
   N[0] = U[0]; N[1] = U[1]; N[2] = U[2];
   N[3] = U[3]; N[4] = U[4]; N[5] = U[5];
   N[6] = U[6]; N[7] = U[7]; N[8] = U[8];
   if (!svd(U, S, V)) {
      // R[idx*9+1] = R[idx*9+2] = R[idx*9+3] = R[idx*9+5] = R[idx*9+6] =
      //    R[idx*9+7] = 0;
      // R[idx*9] = R[idx*9+4] = R[idx*9+8] = 1;
      return;
   }
   // M = U x V.T
   M[0] = U[0]*V[0] + U[1]*V[1] + U[2]*V[2];
   M[1] = U[0]*V[3] + U[1]*V[4] + U[2]*V[5];
   M[2] = U[0]*V[6] + U[1]*V[7] + U[2]*V[8];
   M[3] = U[3]*V[0] + U[4]*V[1] + U[5]*V[2];
   M[4] = U[3]*V[3] + U[4]*V[4] + U[5]*V[5];
   M[5] = U[3]*V[6] + U[4]*V[7] + U[5]*V[8];
   M[6] = U[6]*V[0] + U[7]*V[1] + U[8]*V[2];
   M[7] = U[6]*V[3] + U[7]*V[4] + U[8]*V[5];
   M[8] = U[6]*V[6] + U[7]*V[7] + U[8]*V[8];
}

__global__ void compute_per_particle_procrustes_cem(
   float3* __restrict__ org_vertices,
   float3* __restrict__ sim_vertices, int n_particles,
   int* __restrict__ bondlist, int* __restrict__ n_bonds, int maxbonds,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   float3* __restrict__  random_scale)
{  int i, block_id, idx, cnt;
   float3 Cp, Cq, Po, Qo;
   float3 p0, s0;
   float *M, *N, U[9], S[3], V[9];

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_particles)
      return;

   p0 = org_vertices[idx];
   s0 = sim_vertices[idx];

   float limit = 0.1;
   float scale = 0.98;
   if (n_bonds[idx] == 0) {
      T0[idx] = p0;
      T1[idx] = s0;
      R[idx*9] *= 0.1;
      R[idx*9+4] = 0.1;
      R[idx*9+8] = 0.1;
      return;
   }
   T0[idx] = Cp = p0;
   T1[idx] = Cq = s0;
   if (n_bonds[idx] < 3)
      return;
   M = R+idx*9;
   U[0] = U[1] = U[2] = U[3] = U[4] = U[5] = U[6] = U[7] = U[8] = 0;
   for (i=cnt=0; i < n_bonds[idx]; i++) {
      Po = org_vertices[bondlist[idx*maxbonds+i]] - Cp;
      Qo = sim_vertices[bondlist[idx*maxbonds+i]] - Cq;
      if (sqrt(dot(Qo, Qo)) > 0.005*0.005)
         continue;
      add_inner_product(U, Po, Qo);
      cnt++;
   }
   if (cnt < 5) {
      n_bonds[idx] = 0;
      R[idx*9] *= 0.1;
      R[idx*9+4] = 0.1;
      R[idx*9+8] = 0.1;
      return;
   }
   N = A+idx*9;
   N[0] = U[0]; N[1] = U[1]; N[2] = U[2];
   N[3] = U[3]; N[4] = U[4]; N[5] = U[5];
   N[6] = U[6]; N[7] = U[7]; N[8] = U[8];
   if (!svd(U, S, V))
      return;
   // M = U x V.T
   M[0] = U[0]*V[0] + U[1]*V[1] + U[2]*V[2];
   M[1] = U[0]*V[3] + U[1]*V[4] + U[2]*V[5];
   M[2] = U[0]*V[6] + U[1]*V[7] + U[2]*V[8];
   M[3] = U[3]*V[0] + U[4]*V[1] + U[5]*V[2];
   M[4] = U[3]*V[3] + U[4]*V[4] + U[5]*V[5];
   M[5] = U[3]*V[6] + U[4]*V[7] + U[5]*V[8];
   M[6] = U[6]*V[0] + U[7]*V[1] + U[8]*V[2];
   M[7] = U[6]*V[3] + U[7]*V[4] + U[8]*V[5];
   M[8] = U[6]*V[6] + U[7]*V[7] + U[8]*V[8];
}

__global__ void compute_per_component_procrustes(
   float3* __restrict__ org_vertices,
   float3* __restrict__ sim_vertices, int n_particles,
   int* __restrict__ labels, int max_labels,
   int* __restrict__ bondlist, int* __restrict__ n_bonds, int maxbonds,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   float3* __restrict__  random_scale)
{  int i, block_id, idx, cnt;
   float3 Cp, Cq, Po, Qo;
   float3 p0, s0;
   float *M, *N, U[9], S[3], V[9];

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= max_labels)
      return;

   p0 = org_vertices[idx];
   s0 = sim_vertices[idx];
   if (n_bonds[idx] == 0) {
      T0[idx] = p0;
      T1[idx] = s0;
      //if (R[idx*9] > 0.5 || R[idx*9+4] > 0.5 || R[idx*9+8] > 0.5) {
      //   R[idx*9+1] = R[idx*9+2] = R[idx*9+3] = R[idx*9+5] = R[idx*9+6] =
      //      R[idx*9+7] = 0;
      //   R[idx*9] *= 0.5;
      //   R[idx*9+4] *= 0.5;
      //   R[idx*9+8] *= 0.5;
      //}
      // R[idx*9] = random_scale[idx].x;
      // R[idx*9+4] = random_scale[idx].y;
      // R[idx*9+8] = random_scale[idx].z;
      return;
   }
   Cp = p0;
   Cq = s0;
   cnt = 1;
   for (i=0; i < max_labels; i++) {
      if (labels[i] != idx)
         continue;
      Cp += org_vertices[i];
      Cq += sim_vertices[i];
      cnt++;
   }
   if (cnt > 0) {
      Cp *= (1./cnt);
      Cq *= (1./cnt);
   }
   T0[idx] = Cp;
   T1[idx] = Cq;
   if (cnt < 3)
      return;

   M = R+idx*9;
   U[0] = U[1] = U[2] = U[3] = U[4] = U[5] = U[6] = U[7] = U[8] = 0;
   add_inner_product(U, p0-Cp, s0-Cq);
   for (i=0; i < max_labels; i++) {
      if (labels[i] != idx)
         continue;
      Po = org_vertices[i] - Cp;
      Qo = sim_vertices[i] - Cq;
      add_inner_product(U, Po, Qo);
   }
   N = A+idx*9;
   N[0] = U[0]; N[1] = U[1]; N[2] = U[2];
   N[3] = U[3]; N[4] = U[4]; N[5] = U[5];
   N[6] = U[6]; N[7] = U[7]; N[8] = U[8];
   if (!svd(U, S, V))
      return;

   // M = U x V.T
   M[0] = U[0]*V[0] + U[1]*V[1] + U[2]*V[2];
   M[1] = U[0]*V[3] + U[1]*V[4] + U[2]*V[5];
   M[2] = U[0]*V[6] + U[1]*V[7] + U[2]*V[8];
   M[3] = U[3]*V[0] + U[4]*V[1] + U[5]*V[2];
   M[4] = U[3]*V[3] + U[4]*V[4] + U[5]*V[5];
   M[5] = U[3]*V[6] + U[4]*V[7] + U[5]*V[8];
   M[6] = U[6]*V[0] + U[7]*V[1] + U[8]*V[2];
   M[7] = U[6]*V[3] + U[7]*V[4] + U[8]*V[5];
   M[8] = U[6]*V[6] + U[7]*V[7] + U[8]*V[8];
}

__global__ void compute_per_particle_procrustes_about_centroid(
   float3* __restrict__ org_vertices,
   float3* __restrict__ sim_vertices, int n_particles,
   int* __restrict__ bondlist, int* __restrict__ n_bonds, int maxbonds,
   float3* __restrict__ T0, float3* __restrict__ T1,
   float* __restrict__ R, float* __restrict__ A,
   float3* __restrict__  random_scale)
{  int i, block_id, idx;
   float3 Cp, Cq, Po, Qo;
   float3 p0, s0;
   float *M, *N, U[9], S[3], V[9];

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_particles)
      return;

   p0 = org_vertices[idx];
   s0 = sim_vertices[idx];
   if (n_bonds[idx] == 0) {
      T0[idx] = p0;
      T1[idx] = s0;
      R[idx*9+1] = R[idx*9+2] = R[idx*9+3] = R[idx*9+5] = R[idx*9+6] =
         R[idx*9+7] = 0;
      R[idx*9] = R[idx*9+4] = R[idx*9+8] = 1;
      // R[idx*9] = random_scale[idx].x;
      // R[idx*9+4] = random_scale[idx].y;
      // R[idx*9+8] = random_scale[idx].z;
      return;
   }
   Cp = p0;
   Cq = s0;
   for (i=0; i < n_bonds[idx]; i++) {
      Cp += org_vertices[bondlist[idx*maxbonds+i]];
      Cq += sim_vertices[bondlist[idx*maxbonds+i]];
   }
   if (n_bonds[idx] > 0) {
      Cp *= (1./(n_bonds[idx]+1));
      Cq *= (1./(n_bonds[idx]+1));
   }
   T0[idx] = Cp;
   T1[idx] = Cq;
   if (n_bonds[idx]+1 < 3) {
      R[idx*9+1] = R[idx*9+2] = R[idx*9+3] = R[idx*9+5] = R[idx*9+6] =
         R[idx*9+7] = 0;
      R[idx*9] = R[idx*9+4] = R[idx*9+8] = 1;
      return;
   }
   M = R+idx*9;
   U[0] = U[1] = U[2] = U[3] = U[4] = U[5] = U[6] = U[7] = U[8] = 0;
   add_inner_product(U, p0-Cp, s0-Cq);
   for (i=0; i < n_bonds[idx]; i++) {
      Po = org_vertices[bondlist[idx*maxbonds+i]] - Cp;
      Qo = sim_vertices[bondlist[idx*maxbonds+i]] - Cq;
      add_inner_product(U, Po, Qo);
   }
   N = A+idx*9;
   N[0] = U[0]; N[1] = U[1]; N[2] = U[2];
   N[3] = U[3]; N[4] = U[4]; N[5] = U[5];
   N[6] = U[6]; N[7] = U[7]; N[8] = U[8];
   if (!svd(U, S, V)) {
      R[idx*9+1] = R[idx*9+2] = R[idx*9+3] = R[idx*9+5] = R[idx*9+6] =
         R[idx*9+7] = 0;
      R[idx*9] = R[idx*9+4] = R[idx*9+8] = 1;
      return;
   }
   // M = U x V.T
   M[0] = U[0]*V[0] + U[1]*V[1] + U[2]*V[2];
   M[1] = U[0]*V[3] + U[1]*V[4] + U[2]*V[5];
   M[2] = U[0]*V[6] + U[1]*V[7] + U[2]*V[8];
   M[3] = U[3]*V[0] + U[4]*V[1] + U[5]*V[2];
   M[4] = U[3]*V[3] + U[4]*V[4] + U[5]*V[5];
   M[5] = U[3]*V[6] + U[4]*V[7] + U[5]*V[8];
   M[6] = U[6]*V[0] + U[7]*V[1] + U[8]*V[2];
   M[7] = U[6]*V[3] + U[7]*V[4] + U[8]*V[5];
   M[8] = U[6]*V[6] + U[7]*V[7] + U[8]*V[8];
}

// Compute FTLEs.
__global__ void compute_FTLE(float3* __restrict__ positions,
   float3* __restrict__ next_positions, int n_particles,
   int* __restrict__ bondlist, int* __restrict__ n_bonds,
   int* __restrict__ accum_n_bonds, int maxbonds,
   float inv_ftle_tau,
   float3* __restrict__ locations, float* __restrict__ ftles)
{  int block_id, idx, j, k, l, m;
   float3 diff0, diff1;
   float d0, d1;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_particles)
      return;

   m = accum_n_bonds[idx];
   for (j=0; j < n_bonds[idx]; j++) {
      k = bondlist[idx*maxbonds+j];
      diff0 = positions[idx] - positions[k];
      diff1 = next_positions[idx] - next_positions[k];
      d0 = sqrt(dot(diff0, diff0));
      d1 = sqrt(dot(diff1, diff1));
      locations[m] = 0.5*(positions[idx]+positions[k]);
      ftles[m] = inv_ftle_tau*log(d1/d0);
      m++;
   }
}

__global__ void remove_duplicates1(float3* __restrict__ vertices,
   int* __restrict__ faces, int* __restrict__ out_faces,
   int n_vertices)
{  int i, block_id, idx;
   float3 d, v;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_vertices)   // idx refers to vertex id.
      return;

   v = vertices[faces[idx]];
   for (i=0; i < idx; i++) {
       d = vertices[faces[i]] - v;
       if (dot(d, d) < EPSILON && out_faces[idx] > faces[i]) {
          out_faces[idx] = faces[i];
          break;
       }
   }
}

__global__ void remove_duplicates(float3* __restrict__ vertices,
   int* __restrict__ faces, int n_slices,
   int* __restrict__ out_faces,
   int n_vertices, int *actual_n_vertices)
{  int i, block_id, idx, min_vert_id, vert_id;
   int slice_id, per_slice, end_slice;
   float3 d, v;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_slices*n_vertices)   // idx refers to vertex id.
      return;

   slice_id = idx % n_slices;
   vert_id = idx / n_slices;
   per_slice = n_vertices / n_slices;
   min_vert_id = faces[vert_id];
   v = vertices[faces[min_vert_id]];
   if (slice_id == n_slices-1)
      end_slice = n_vertices;
   else
      end_slice = (slice_id+1)*per_slice;
   for (i=slice_id*per_slice; i < end_slice; i++) {
      d = vertices[faces[i]] - v;
      if (dot(d, d) < EPSILON && min_vert_id > faces[i])
         min_vert_id = faces[i];
   }
   atomicMin(out_faces+vert_id, min_vert_id);
   __syncthreads();
   // if (min_vert_id == idx) {
   //    vert_id = atomicAdd(actual_n_vertices, 1);
   //    out_faces[idx] = vert_id;
   //    out_vertices[vert_id] = vertices[idx];
   // }
   // else
   //    out_faces[idx] = vert_id;
}

__global__ void count_vertices(int* __restrict__ faces,
   int* __restrict__ counts, int n_vertices, int *actual_n_vertices)
{  int i, block_id, idx, cnt;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_vertices)
      return;

   cnt = 0;
   for (i=0; i < n_vertices; i++) {
      if (faces[i] == idx) {
         cnt++;
         break;
      }
   }
   counts[idx] = cnt;
   if (cnt > 0)
      atomicAdd(actual_n_vertices, 1);
}

__global__ void accum_zero_counts(int* __restrict__ counts,
   int* __restrict__ accum_zero_counts, int n_vertices)
{  int i, block_id, idx, cnt;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_vertices)
      return;

   cnt = 0;
   for (i=0; i < idx; i++) {
      if (counts[i] == 0)
         cnt++;
   }
   accum_zero_counts[idx] = cnt;
}

__global__ void remove_vertices(float3* __restrict__ vertices,
   float3* __restrict__ out_vertices, int* __restrict__ faces,
   int* __restrict__ counts, int* __restrict__ accum_zero_counts,
   int* __restrict__ out_faces, int n_vertices)
{  int block_id, idx;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_vertices)
      return;

   out_faces[idx] = faces[idx] - accum_zero_counts[idx];
   out_vertices[out_faces[idx]] = vertices[faces[idx]];
}

__global__ void smooth_normals(float3* __restrict__ vertices, int n_vertices,
   int* __restrict__ faces, int n_faces, float3* __restrict__ normals)
{  int i, block_id, idx;
   float3 e0, e1, n;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   if (idx >= n_vertices)
      return;

   n.x = n.y = n.z = 0;
   for (i=0; i < n_faces; i++) {
      if (faces[3*i] == idx || faces[3*i+1] == idx || faces[3*i+2] == idx) {
         e0 = vertices[faces[3*i]]   - vertices[faces[3*i+1]];
         e1 = vertices[faces[3*i+2]] - vertices[faces[3*i+1]];
         e0 = normalize(e0);
         e1 = normalize(e1);
         n  = n + cross(e0, e1);
      }
   }
   if (dot(n, n) < EPSILON)
      normals[idx].x = normals[idx].y = normals[idx].z = 0;
   else
      normals[idx] = normalize(n);
}

__global__ void print_threads()
{  int block_id, idx;

   block_id = blockIdx.y * gridDim.x + blockIdx.x;
   idx = block_id * blockDim.x + threadIdx.x;
   printf("Block: (%d, %d, %d), "
          "Thread: (%d, %d, %d), "
          "BDim: (%d,%d,%d), "
          "GDim: (%d,%d,%d), idx: %d\n",
          blockIdx.x, blockIdx.y, blockIdx.z,
          threadIdx.x, threadIdx.y, threadIdx.z,
          blockDim.x, blockDim.y, blockDim.z,
          gridDim.x, gridDim.y, gridDim.z, idx);
}

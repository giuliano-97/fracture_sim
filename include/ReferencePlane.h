#include <Eigen/Core>
#include <vector>

class ReferencePlane {
public:
	ReferencePlane() : size(10), color(Eigen::RowVector3d(0.25, 0.25, 0.25)) {
		//// for set_edges
		//int num_edges = 2 * (2 * size)*(2 * size) + (2 * size) * 2;
		//int num_points = (2 * size + 1)*(2 * size + 1);

		//points = Eigen::MatrixXd(num_points, 3);
		//edges = Eigen::MatrixXi(num_edges, 2);

		//int i = 0;
		//for (int z = -size; z <= size; ++z) {
		//	for (int x = -size; x <= size; ++x) {
		//		points.row(i++) = Eigen::RowVector3d(x, -5, z);
		//	}
		//}

		//i = 0;
		//for (int z = 0; z <= 2*size; ++z) {
		//	for (int x = 0; x <= 2*size; ++x) {
		//		int p1 = z * (2 * size + 1) + x;
		//		int p2 = p1 + 1;
		//		int p3 = p1 + (2 * size + 1);
		//		if (x < 2*size)
		//			edges.row(i++) = Eigen::RowVector2i(p1, p2);
		//		if (z < 2 * size)
		//			edges.row(i++) = Eigen::RowVector2i(p1, p3);
		//	}
		//}

		int num_edges = 2 * (2 * size)*(2 * size) + (2 * size) * 2;
		start = Eigen::MatrixXd(num_edges, 3);
		end = Eigen::MatrixXd(num_edges, 3);

		int e = 0;
		for (int z = -size; z <= size; ++z)
			for (int x = -size; x <= size; ++x) {
				if (x < size) {
					start.row(e) = Eigen::RowVector3d(x, 0, z);
					end.row(e++) = Eigen::RowVector3d(x + 1, 0, z);
				}
				if (z < size) {
					start.row(e) = Eigen::RowVector3d(x, 0, z);
					end.row(e++) = Eigen::RowVector3d(x, 0, z + 1);
				}
			}
	}

	//Eigen::MatrixXd points;
	//Eigen::MatrixXi edges;
	Eigen::MatrixXd start;
	Eigen::MatrixXd end;
	Eigen::RowVector3d color;
	int size;
};
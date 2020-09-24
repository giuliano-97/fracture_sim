#ifndef MACGRID2_H
#define MACGRID2_H

#include <igl/colormap.h>
#include <Eigen/Core>
#include "Array2T.h"

class MACGrid2 {
public:
	MACGrid2(int res_x, int res_y) {
		m_res_x = res_x;
		m_res_y = res_y;
		m_x = Array2d(res_x + 1, res_y);
		m_y = Array2d(res_x, res_y + 1);
		buildGrid();
	}

	Array2d& x() { return m_x; }
	Array2d& y() { return m_y; }

	const Eigen::MatrixXd& s() { return m_start; }
	const Eigen::MatrixXd& e() { return m_end; }
	const Eigen::MatrixXd& vs() { return m_vs; }
	const Eigen::MatrixXd& ve() { return m_ve; }
	const Eigen::MatrixXd& vc() { return m_vc; }

	void buildGrid() {
		int num_edges = 2 * m_res_x * m_res_y + m_res_x + m_res_y;
		m_start = Eigen::MatrixXd(num_edges, 3);
		m_end = Eigen::MatrixXd(num_edges, 3);

		int i = 0;
		for (int y = 0; y <= m_res_y; ++y)
			for (int x = 0; x <= m_res_x; ++x) {
				if (x < m_res_x) {
					m_start.row(i) = Eigen::RowVector3d(x, y, 0);
					m_end.row(i++) = Eigen::RowVector3d(x + 1, y, 0);
				}
				if (y < m_res_y) {
					m_start.row(i) = Eigen::RowVector3d(x, y, 0);
					m_end.row(i++) = Eigen::RowVector3d(x, y + 1, 0);
				}
			}
	}

	void updateEdges(double scale = 1) {
		int num_edges = (m_res_x + 1) * m_res_y + m_res_x * (m_res_y + 1) + m_res_x * m_res_y;
		if (m_vs.rows() == 0) {
			m_vs = Eigen::MatrixXd(num_edges, 3);
			m_ve = Eigen::MatrixXd(num_edges, 3);
			m_vc = Eigen::MatrixXd(num_edges, 3);
		}

		int i = 0;
		for (int y = 0; y <= m_res_y; ++y)
			for (int x = 0; x <= m_res_x; ++x) {
				if (y < m_res_y) {
					m_vc.row(i) = Eigen::RowVector3d(1, 0, 0);
					m_vs.row(i) = Eigen::RowVector3d(x, y + 0.5, 0);
					m_ve.row(i++) = Eigen::RowVector3d(x + m_x(x, y) * scale, y + 0.5, 0);
				}

				if (x < m_res_x) {
					m_vc.row(i) = Eigen::RowVector3d(0, 0, 1);
					m_vs.row(i) = Eigen::RowVector3d(x + 0.5, y, 0);
					m_ve.row(i++) = Eigen::RowVector3d(x + 0.5, y + m_y(x, y) * scale, 0);
				}

				if (y < m_res_y && x < m_res_x) {
					double vx = (m_x(x, y) + m_x(x + 1, y)) * 0.5;
					double vy = (m_y(x, y) + m_x(x, y + 1)) * 0.5;
					m_vc.row(i) = Eigen::RowVector3d(0, 1, 0);
					m_vs.row(i) = Eigen::RowVector3d(x + 0.5, y + 0.5, 0);
					m_ve.row(i++) = Eigen::RowVector3d(x + 0.5 + vx * scale, y + 0.5 + vy * scale, 0);
				}
			}
	}
	
	void reset() {		
		m_x.zero();
		m_y.zero();
	}

protected:
	int m_res_x, m_res_y;
	Array2d m_x;
	Array2d m_y;
	Eigen::MatrixXd m_start;
	Eigen::MatrixXd m_end;
	Eigen::MatrixXd m_vs;
	Eigen::MatrixXd m_ve;
	Eigen::MatrixXd m_vc;
};

#endif // MACGRID2_H
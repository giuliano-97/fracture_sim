#ifndef AABB_H
#define AABB_H

#include <Eigen/Core>
#include "RigidObject.h"

class AABB {
   public:
    AABB()
        : m_minCoord(
              Eigen::Vector3d::Constant(std::numeric_limits<double>::min())),
          m_maxCoord(
              Eigen::Vector3d::Constant(std::numeric_limits<double>::max())) {}

    AABB(const Eigen::Vector3d& minCoord, const Eigen::Vector3d& maxCoord)
        : m_minCoord(minCoord), m_maxCoord(maxCoord) {}

    const Eigen::Vector3d& getMinCoord() const { return m_minCoord; };
    const Eigen::Vector3d& getMaxCoord() const { return m_maxCoord; };

    void setMinCoord(const Eigen::Vector3d& minCoord) { m_minCoord = minCoord; }
    void setMaxCoord(const Eigen::Vector3d& maxCoord) { m_maxCoord = maxCoord; }

    bool testCollision(const AABB& other) const {
        if (m_maxCoord.x() < other.m_minCoord.x() ||
            other.m_maxCoord.x() < m_minCoord.x())
            return false;
        if (m_maxCoord.y() < other.m_minCoord.y() ||
            other.m_maxCoord.y() < m_minCoord.y())
            return false;
        if (m_maxCoord.z() < other.m_minCoord.z() ||
            other.m_maxCoord.z() < m_minCoord.z())
            return false;
        return true;
    }

    void computeAABB(const RigidObject & obj) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        obj.getMesh(V, F);

        Eigen::Vector3d minCoord = V.colwise().minCoeff();
        setMinCoord(minCoord);
        Eigen::Vector3d maxCoord = V.colwise().maxCoeff();
        setMaxCoord(maxCoord);
    }

    void clear() {
        m_minCoord =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::min());
        m_maxCoord =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    }

   private:
    Eigen::Vector3d m_minCoord;
    Eigen::Vector3d m_maxCoord;
};

#endif // AABB_H
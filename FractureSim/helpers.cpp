#include "helpers.h"

namespace d3_helpers {

	double norm(const double3& v)
	{
		return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	}

	void setZero(double3& v) {
		v.x = v.y = v.z = 0;
	}

	double3 operator+(const double3& a, const double3& b) {
		return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
	}

	double3 operator-(const double3& a, const double3& b) {
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}

	double3 operator*(double s, const double3& v) {
		return make_double3(s * v.x, s * v.y, s * v.z);
	}

	double3 operator*(const double3& v, double s) {
		return make_double3(s * v.x, s * v.y, s * v.z);
	}

	std::ostream& operator<<(std::ostream& os, const double3& v)
	{
		os << v.x << ' ' << v.y << ' ' << v.z;
		return os;
	}

}
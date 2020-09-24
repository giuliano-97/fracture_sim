#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include <sstream>

namespace d3_helpers {

	double norm(const double3& v);

	void setZero(double3& v);

	double3 operator+(const double3& a, const double3& b);

	double3 operator-(const double3& a, const double3& b);

	double3 operator*(double s, const double3& v);

	double3 operator*(const double3& v, double s);

	std::ostream& operator<<(std::ostream& os, const double3& v);
}

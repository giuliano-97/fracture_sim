//=============================================================================
//  Physically-based Simulation in Computer Graphics
//  ETH Zurich
//
//  Author: Christian Schumacher
//=============================================================================

#pragma once

#include <vector>
#include <assert.h>

// Simple 2D array
template <typename SCALAR>
class Array2T
{
public:
	// Default constructor
	Array2T()
	{
		m_size[0] = 0;
		m_size[1] = 0;
	}

	// Constructor with given size
	Array2T(int size0, int size1, SCALAR value = (SCALAR)0)
	{
		resize(size0, size1, value);
	}

	// Copy constructor
	Array2T(const Array2T<SCALAR> &m)
	{
		*this = m;
	}

	// Resize array
	void resize(int size0, int size1, SCALAR value = (SCALAR)0)
	{
		m_size[0] = size0;
		m_size[1] = size1;

		m_data.resize(size0 * size1, value);
	}

	// Fill array with scalar s
	void fill(SCALAR s)
	{
		std::fill(m_data.begin(), m_data.end(), s);
	}

	// Fill array with 0
	void zero()
	{
		fill(0);
	}

	// Read & write element access
	SCALAR& operator()(unsigned int i, unsigned int j)
	{
		assert(i >= 0 && i < m_size[0] && j >= 0 && j < m_size[1]);
		return m_data[i * m_size[1] + j];
	}

	// Read only element access
	const SCALAR& operator()(unsigned int i, unsigned int j) const
	{
		assert(i >= 0 && i < m_size[0] && j >= 0 && j < m_size[1]);
		return m_data[i * m_size[1] + j];
	}

	// Dimension
	int size(int dimension) const
	{
		assert(dimension >= 0 && dimension < 2);
		return (int)m_size[dimension];
	}

	// Assignment
	Array2T<SCALAR> &operator=(const Array2T<SCALAR> &m2)
	{
		if (&m2 != this)
		{
			resize(m2.size(0), m2.size(1));

			int n = (int)m_data.size();
			for (int i = 0; i < n; i++)
				m_data[i] = m2.m_data[i];
		}

		return *this;
	}

protected:
	unsigned int		m_size[2];
	std::vector<SCALAR>	m_data;
};

typedef Array2T<double> Array2d;

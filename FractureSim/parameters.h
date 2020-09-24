#pragma once

const double g_timestep = 1e-03;
const double g_lambda = 0.5;		// minimum distance between particles
const int g_N = 4;					// N * lambda as the distance of neighbor query to create springs
const double g_K =  5e+8;				// bulk
const double g_Kc = 1e+8;			// Collision response constant
const double g_G = 3.10;				// material fracturing energy

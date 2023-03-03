#pragma once
#include <SFML/Graphics.hpp>
#include <thrust/device_vector.h>
#include "cuda_runtime.h"

/// <summary>
/// Particle representation
/// </summary>
struct Particles
{
	float2		 position;
	float2       nextPosition;
	float		 angle;
	int			 neighbors;
	int			 closeNeighbors;
};

class CudaParticleManager
{
public:
	CudaParticleManager(float particleSpeed = 0.0f, float alpha = 0.0f, float beta = 0.0f, float reactRadius = 0.0f, float2 simulationBound = float2{ 0.0f,0.0f }, int cpuThreadNumber = 1, int blockNumber = 1, int threadPerBlock = 1);

	void draw(sf::RenderWindow& window);

	void spawnCells(sf::Vector2f position, int cellNumber);

	void update();

	void reduceSimulationBound();

	void increaseSimulationBound();

	float2 getSimulationBound() const;

	void changeMode();

private:
	std::vector<Particles>				m_cpuParticles;			//!< Particles vector
	thrust::device_vector<Particles>	m_gpuParticles;			//!< Particles vector on GPU
	float								m_particleSpeed;		//!< Particle speed
	float								m_alpha;				//!< Alpha degree
	float								m_beta;					//!< Beta degree
	float								m_reactionRadius;		//!< Reaction radius
	float2								m_simulationBound;		//!< Simulation bound
	int									m_cpuThreadNumber;		//!< Maximum number of cpu threads
	int									m_blockNumber;			//!< Number of blocks (GPU)
	int									m_threadPerBlock;		//!< Number of threads per block (GPU)
	bool								m_cpuMode{ true };		//!< Update mode GPU or CPU
};

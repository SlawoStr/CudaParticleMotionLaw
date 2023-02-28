#pragma once
#include <SFML/Graphics.hpp>
#include <thrust/device_vector.h>
#include "cuda_runtime.h"

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
	std::vector<Particles>				m_cpuParticles;
	thrust::device_vector<Particles>	m_gpuParticles;
	float								m_particleSpeed;
	float								m_alpha;
	float								m_beta;
	float								m_reactionRadius;
	float2								m_simulationBound;
	int									m_cpuThreadNumber;
	int									m_blockNumber;
	int									m_threadPerBlock;
	bool								m_cpuMode{ true };
};

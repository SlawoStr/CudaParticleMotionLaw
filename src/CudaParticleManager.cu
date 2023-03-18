#include "CudaParticleManager.cuh"
#include "device_launch_parameters.h"
#include "RNGGenerator.h"
#include "CudaError.h"
#include <math.h>
#define PI 3.14159265

/// <summary>
/// Calculate signum
/// </summary>
/// <param name="val">Value</param>
/// <returns>Signum of value</returns>
template <typename T> __host__ __device__ int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

/// <summary>
/// Transfer radians to degrees
/// </summary>
/// <param name="a">Radians</param>
/// <returns>Degrees</returns>
inline __host__ __device__ float degree(float a)
{
	return static_cast<float>(a * (180 / PI));
}

/// <summary>
/// Transfer degree to radians
/// </summary>
/// <param name="a"></param>
/// <returns></returns>
inline __host__ __device__ float radians(float a)
{
	return static_cast<float>(0.017453292 * a);
}

/// <summary>
/// Get distance between two points
/// </summary>
/// <param name="fp">First point</param>
/// <param name="sp">Second point</param>
/// <returns>Distance between points</returns>
__host__ __device__ float getDistance(float2 fp, float2 sp)
{
	return static_cast<float>(sqrtf(powf(sp.x - fp.x, 2) + powf(sp.y - fp.y, 2)));
}

/// <summary>
/// Check if point is on the right side of line
/// </summary>
/// <param name="a">First point of line</param>
/// <param name="b">Second point of line</param>
/// <param name="c">Point</param>
/// <returns>Is point on the right side</returns>
__host__ __device__ bool isRight(float2 a, float2 b, float2 c)
{
	return((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) > 0;
}

////////////////////////////////////////////////////////////
__global__ void updateAgents(Particles* particles, float particleSpeed, float alpha, float beta, float reactionRadius,float2 simulationBound, int taskSize)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < taskSize; i += blockDim.x * gridDim.x)
	{
		int Nt{};
		int Rt{};
		int Lt{};
		int Ct{};
		// Angle line coordinates
		float2 posStart{ particles[i].position };
		float2 posEnd{ particles[i].position.x + cos(particles[i].angle),particles[i].position.y + sin(particles[i].angle) };
		for (int j = 0; j < taskSize; ++j)
		{
			if (i == j)
				continue;
			float distance{ getDistance(particles[i].position, particles[j].position) };
			if (distance < reactionRadius)
			{
				Nt++;
				if (isRight(posStart, posEnd, particles[j].position))
				{
					Rt++;
				}
				else
				{
					Nt++;
				}
				if (distance < 1.3f)
				{
					Ct++;
				}
			}
		}
		// Update particle data
		float rotationAngle = alpha + beta * Nt * sgn(Rt - Lt);
		particles[i].neighbors = Nt;
		particles[i].closeNeighbors = Ct;
		particles[i].angle += rotationAngle;
		particles[i].nextPosition = { posStart.x + particleSpeed * cos(particles[i].angle),posStart.y + particleSpeed * sin(particles[i].angle) };
		if (particles[i].nextPosition.x < 0.0f || particles[i].nextPosition.x > simulationBound.x || particles[i].nextPosition.y < 0.0f || particles[i].nextPosition.y > simulationBound.y)
		{
			particles[i].nextPosition = particles[i].position;
		}
	}
}

////////////////////////////////////////////////////////////
__global__ void updatePos(Particles* particles,int taskSize)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < taskSize; i += blockDim.x * gridDim.x)
	{
		particles[i].position = particles[i].nextPosition;
	}
}

////////////////////////////////////////////////////////////
CudaParticleManager::CudaParticleManager(float particleSpeed, float alpha, float beta, float reactRadius, float2 simulationBound, int cpuThreadNumber, int threadPerBlock)
	: m_particleSpeed{ particleSpeed }, m_alpha{ radians(alpha) }, m_beta{ radians(beta) }, m_reactionRadius{ reactRadius }, m_simulationBound{ simulationBound },
	m_cpuThreadNumber{ cpuThreadNumber }, m_threadPerBlock{ threadPerBlock }
{
}

////////////////////////////////////////////////////////////
void CudaParticleManager::draw(sf::RenderWindow& window)
{
	sf::CircleShape shape;
	shape.setRadius(1.0f);
	shape.setOrigin(shape.getRadius(), shape.getRadius());

	sf::Vertex line[2];
	line[0].color = sf::Color::Red;
	line[1].color = sf::Color::Red;

	for (const auto& particle : m_cpuParticles)
	{
		shape.setFillColor(sf::Color::Green);
		int n = particle.neighbors;
		if (n > 35)
		{
			shape.setFillColor(sf::Color::Yellow);
		}
		else if (n > 15 && n <= 35)
		{
			shape.setFillColor(sf::Color::Blue);
		}
		else if (n >= 13 && n <= 15)
		{
			shape.setFillColor(sf::Color(136, 69, 19));
		}
		if (particle.closeNeighbors > 15)
		{
			shape.setFillColor(sf::Color::Magenta);
		}
		float2 particlePostion = particle.position;
		shape.setPosition(particlePostion.x, particlePostion.y);
		line[0].position = { particlePostion.x,particlePostion.y };
		line[1].position.x = particlePostion.x + cos(particle.angle);
		line[1].position.y = particlePostion.y + sin(particle.angle);
		window.draw(shape);
		window.draw(line, 2, sf::Lines);
	}
}

////////////////////////////////////////////////////////////
void CudaParticleManager::spawnCells(sf::Vector2f position, int cellNumber)
{
	Particles particle{};
	particle.position = { position.x,position.y };
	// Add new particles to cpu vector and copy it to gpu vector
	for (int i = 0; i < cellNumber; ++i)
	{
		particle.angle = RNGGenerator::randFloat(0.0f, 6.2831f);
		m_cpuParticles.push_back(particle);
	}
	size_t gpuSize = m_gpuParticles.size();
	m_gpuParticles.resize(m_cpuParticles.size());
	checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(m_gpuParticles.data()) + gpuSize, m_cpuParticles.data() + gpuSize, sizeof(Particles) * cellNumber, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////
void CudaParticleManager::update()
{
	if (m_cpuMode)
	{
		#pragma omp parallel for num_threads(m_cpuThreadNumber)
		for (int i = 0; i < m_cpuParticles.size(); ++i)
		{
			int Nt{};
			int Rt{};
			int Lt{};
			int Ct{};
			// Angle line coordinates
			float2 posStart{ m_cpuParticles[i].position };
			float2 posEnd{ m_cpuParticles[i].position.x + cos(m_cpuParticles[i].angle),m_cpuParticles[i].position.y + sin(m_cpuParticles[i].angle) };
			for (int j = 0; j < m_cpuParticles.size(); ++j)
			{
				if (i == j)
					continue;
				// If particle is in reaction radius
				float distance{ getDistance(m_cpuParticles[i].position, m_cpuParticles[j].position) };
				if (distance < m_reactionRadius)
				{
					Nt++;
					if (isRight(posStart, posEnd, m_cpuParticles[j].position))
					{
						Rt++;
					}
					else
					{
						Nt++;
					}
					if (distance < 1.3f)
					{
						Ct++;
					}
				}
			}
			// Update particle data
			float rotationAngle = m_alpha + m_beta * Nt * sgn(Rt - Lt);
			m_cpuParticles[i].neighbors = Nt;
			m_cpuParticles[i].closeNeighbors = Ct;
			m_cpuParticles[i].angle += rotationAngle;
			m_cpuParticles[i].nextPosition = { posStart.x + m_particleSpeed * cos(m_cpuParticles[i].angle),posStart.y + m_particleSpeed * sin(m_cpuParticles[i].angle) };
			if (m_cpuParticles[i].nextPosition.x < 0.0f || m_cpuParticles[i].nextPosition.x > m_simulationBound.x || m_cpuParticles[i].nextPosition.y < 0.0f || m_cpuParticles[i].nextPosition.y > m_simulationBound.y)
			{
				m_cpuParticles[i].nextPosition = m_cpuParticles[i].position;
			}
		}
		// Update particle positions
		for (int i = 0; i < m_cpuParticles.size(); ++i)
		{
			m_cpuParticles[i].position = m_cpuParticles[i].nextPosition;
		}
	}
	else
	{
		int blockNumber = (static_cast<int>(m_gpuParticles.size()) + m_threadPerBlock - 1) / m_threadPerBlock;
		updateAgents << <blockNumber, m_threadPerBlock >> > (thrust::raw_pointer_cast(m_gpuParticles.data()), m_particleSpeed, m_alpha, m_beta, m_reactionRadius, m_simulationBound, static_cast<int>(m_gpuParticles.size()));
		updatePos << <blockNumber, m_threadPerBlock >> > (thrust::raw_pointer_cast(m_gpuParticles.data()), static_cast<int>(m_gpuParticles.size()));
		checkCudaErrors(cudaMemcpy(m_cpuParticles.data(), thrust::raw_pointer_cast(m_gpuParticles.data()), sizeof(Particles) * m_cpuParticles.size(), cudaMemcpyDeviceToHost));
	}
}

////////////////////////////////////////////////////////////
void CudaParticleManager::reduceSimulationBound()
{
	m_simulationBound.x -= 1;
	m_simulationBound.y -= 1;

	for (int i = 0; i < m_cpuParticles.size(); ++i)
	{
		if (m_cpuParticles[i].position.x >= m_simulationBound.x)
		{
			m_cpuParticles[i].position.x -= 1;
		}
		if (m_cpuParticles[i].position.y >= m_simulationBound.y)
		{
			m_cpuParticles[i].position.y -= 1;
		}
	}
	checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(m_gpuParticles.data()), m_cpuParticles.data(), sizeof(Particles) * m_cpuParticles.size(), cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////
void CudaParticleManager::increaseSimulationBound()
{
	m_simulationBound.x += 1;
	m_simulationBound.y += 1;
}

////////////////////////////////////////////////////////////
float2 CudaParticleManager::getSimulationBound() const
{
	return m_simulationBound;
}

////////////////////////////////////////////////////////////
void CudaParticleManager::changeMode()
{
	if (m_cpuMode == true)
	{
		checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(m_gpuParticles.data()), m_cpuParticles.data(), sizeof(Particles) * m_cpuParticles.size(), cudaMemcpyHostToDevice));
	}
	else
	{
		checkCudaErrors(cudaMemcpy(m_cpuParticles.data(), thrust::raw_pointer_cast(m_gpuParticles.data()), sizeof(Particles) * m_cpuParticles.size(), cudaMemcpyDeviceToHost));
	}
	m_cpuMode = !m_cpuMode;
}

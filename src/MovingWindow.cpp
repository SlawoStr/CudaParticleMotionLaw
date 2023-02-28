#include <iostream>
#include "MovingWindow.h"
#include "Timer.h"
#include "ResourceManager.h"


////////////////////////////////////////////////////////////
MovingWindow::MovingWindow(unsigned windowWidth, unsigned windowHeight, std::string windowTitle, unsigned framerate) : m_camera(m_window)
{
	m_window.create(sf::VideoMode(windowWidth, windowHeight), windowTitle);
	m_window.setFramerateLimit(framerate);
	// Load configuration from file
	ResourcesManager resourceManager("Resources/ParticleConfig.lua");
	float particleSpeed = resourceManager.getFloat("ParticleSpeed");
	float alpha = resourceManager.getFloat("Alpha");
	float beta = resourceManager.getFloat("Beta");
	float reactRadius = resourceManager.getFloat("ReactRadius");
	float2 simulationBound = { resourceManager.getFloat("BoundX"),resourceManager.getFloat("BoundY") };
	int threadNumber = resourceManager.getInt("ThreadNumber");
	int blockNumber = resourceManager.getInt("BlockNumber");
	int threadPerBlock = resourceManager.getInt("ThreadPerBlock");
	m_manager = CudaParticleManager(particleSpeed, alpha, beta, reactRadius, simulationBound, threadNumber, blockNumber, threadPerBlock);

	m_background.setFillColor(sf::Color::White);
	m_background.setPosition(sf::Vector2f{ 0.0f,0.0f });
	m_background.setSize({ simulationBound.x,simulationBound.y });
}

////////////////////////////////////////////////////////////
void MovingWindow::run()
{
	Timer t;
	while (m_window.isOpen())
	{
		t.start();
		if (!m_pause)
		{
			update();
		}
		m_window.clear(sf::Color(128, 128, 128));
		draw();
		m_window.display();
		pollEvent();
		t.stop();
		double elapsedTime = t.measure();
		std::cout << "Frame time: " << elapsedTime << std::endl;
	}
}

////////////////////////////////////////////////////////////
void MovingWindow::pollEvent()
{
	sf::Event e;
	while (m_window.pollEvent(e))
	{
		if (m_camera.handleEvent(e))
		{
			break;
		}
		switch (e.type)
		{
			case sf::Event::Closed:
			{
				m_window.close();
				break;
			}
			case sf::Event::KeyPressed:
			{
				if (e.key.code == sf::Keyboard::Escape)
				{
					m_window.close();
				}
				else if (e.key.code == sf::Keyboard::P)
				{
					m_pause = !m_pause;
				}
				else if (e.key.code == sf::Keyboard::Up)
				{
					m_manager.increaseSimulationBound();
					m_background.setSize({ m_manager.getSimulationBound().x, m_manager.getSimulationBound().y });
				}
				else if (e.key.code == sf::Keyboard::Down)
				{
					m_manager.reduceSimulationBound();
					m_background.setSize({ m_manager.getSimulationBound().x, m_manager.getSimulationBound().y });
				}
				else if (e.key.code == sf::Keyboard::M)
				{
					m_manager.changeMode();
				}
				break;
			}
			case sf::Event::MouseButtonPressed:
			{
				if (e.key.code == sf::Mouse::Button::Left)
				{
					m_manager.spawnCells(m_window.mapPixelToCoords(sf::Mouse::getPosition(m_window)), 690);
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////
void MovingWindow::update()
{
	m_manager.update();
}

////////////////////////////////////////////////////////////
void MovingWindow::draw()
{
	m_window.draw(m_background);
	m_manager.draw(m_window);
}

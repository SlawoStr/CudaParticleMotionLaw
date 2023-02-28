#pragma once
#include "Simulation.h"
#include "CameraController.h"
#include "CudaParticleManager.cuh"

/// <summary>
/// Template for window with basic movement (camera movement, zooming)
/// </summary>
class MovingWindow : public Simulation
{
public:
	/// <summary>
	/// Create window of application
	/// </summary>
	/// <param name="windowWidth">Window width</param>
	/// <param name="windowHeight">Windot height</param>
	/// <param name="windowTitle">Window title</param>
	/// <param name="framerate">Window maximum framerate</param>
	MovingWindow(unsigned windowWidth, unsigned windowHeight, std::string windowTitle, unsigned framerate);
	/// <summary>
	/// Run simulation
	/// </summary>
	void run() override;
private:
	/// <summary>
	/// Handle user interactions
	/// </summary>
	void pollEvent() override;
	/// <summary>
	/// Update elements for next frame
	/// </summary>
	void update() override;
	/// <summary>
	/// Draw elements on screen
	/// </summary>
	void draw() override;

private:
	bool				m_pause{ false };		//!< Is simulation updated
	CameraController	m_camera;				//!< Camera handler for movement etc.
	CudaParticleManager m_manager;				//!< Particle manager
	sf::RectangleShape  m_background;			//!< Simulation background color
};
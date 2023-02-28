#pragma once
#include <SFML/Graphics.hpp>

/// <summary>
/// Abstract class for all simulations
/// </summary>
class Simulation
{
public:
	virtual ~Simulation() {}
	/// <summary>
	/// Run simulation
	/// </summary>
	virtual void run() = 0;
private:
	/// <summary>
	/// Handle user interactions
	/// </summary>
	virtual void pollEvent() = 0;
	/// <summary>
	/// Update elements for next frame
	/// </summary>
	virtual void update() = 0;
	/// <summary>
	/// Draw elements on screen
	/// </summary>
	virtual void draw() = 0;
protected:
	sf::RenderWindow m_window;		//!< Main window of application
};
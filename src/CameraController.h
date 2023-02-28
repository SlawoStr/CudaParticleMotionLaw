#pragma once
#include <SFML/Graphics.hpp>

/// <summary>
/// Handle user interaction with camera
/// </summary>
class CameraController
{
public:
	CameraController(sf::RenderWindow& window) : m_window(window) {}
	/// <summary>
	/// Handle camera events
	/// </summary>
	/// <param name="window">Reference to current window</param>
	/// <param name="e">Event to handle</param>
	/// <returns>True if event was handled by camera else false</returns>
	bool handleEvent(sf::Event e);
	/// <summary>
	/// Return view box currently seen by camera
	/// </summary>
	/// <param name="window">Reference to window</param>
	/// <returns>Top left and Bot Right positions of view box</returns>
	auto getViewBox();
private:
	sf::RenderWindow& m_window;				//!< Reference to window that will be handled by this controller
	float m_zoomLevel{ 1.0f };				//!< Currently zoom level
	bool m_panning{ false };				//!< Is camera in panning mode
	sf::Vector2i m_panningAnchor{ 0,0 };	//!< Panning anchor for camera movement
};
#include "CameraController.h"

////////////////////////////////////////////////////////////
bool CameraController::handleEvent(sf::Event e)
{
	switch (e.type)
	{
		case sf::Event::Resized:
		{
			sf::View view = m_window.getView();
			view.setSize(static_cast<float>(e.size.width), static_cast<float>(e.size.height));
			view.zoom(m_zoomLevel);
			m_window.setView(view);
			return true;
		}
		case sf::Event::MouseMoved:
		{
			if (m_panning)
			{
				sf::View view = m_window.getView();
				sf::Vector2i mousePos = sf::Mouse::getPosition(m_window);
				sf::Vector2f pos = sf::Vector2f(mousePos - this->m_panningAnchor);
				view.move(-1.0f * pos * m_zoomLevel);
				m_window.setView(view);
				m_panningAnchor = mousePos;
				return true;
			}
			break;
		}
		case sf::Event::MouseButtonPressed:
		{
			if (e.mouseButton.button == sf::Mouse::Middle)
			{
				m_panning = true;
				m_panningAnchor = sf::Mouse::getPosition(m_window);
				return true;
			}
			break;
		}
		case sf::Event::MouseButtonReleased:
		{
			if (e.mouseButton.button == sf::Mouse::Middle)
			{
				m_panning = false;
				return true;
			}
			break;
		}
		case sf::Event::MouseWheelMoved:
		{
			sf::View view = m_window.getView();
			if (e.mouseWheel.delta < 0)
			{
				view.zoom(2.0f);
				m_zoomLevel *= 2.0f;
			}
			else
			{
				view.zoom(0.5f);
				m_zoomLevel *= 0.5f;
			}
			m_window.setView(view);
			return true;
		}
	}
	return false;
}

////////////////////////////////////////////////////////////
auto CameraController::getViewBox()
{
	sf::View view = m_window.getView();
	sf::Vector2f topLeft = sf::Vector2f{ view.getCenter().x - view.getSize().x / 2, view.getCenter().y - view.getSize().y / 2 };
	sf::Vector2f botRight = sf::Vector2f{ view.getCenter().x + view.getSize().x / 2, view.getCenter().y + view.getSize().y / 2 };
	return std::make_pair(topLeft, botRight);
}

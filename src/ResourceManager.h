#pragma once
#include <string>

extern "C"
{
#include "Lua542/include/lua.h"
#include "Lua542/include/lauxlib.h"
#include "Lua542/include/lualib.h"
}
#include <iostream>
#ifdef _WIN32
#pragma comment(lib,"src/Lua542/liblua54.a")
#endif // _WIN32

/// <summary>
/// Lua files resource manager
/// </summary>
class ResourcesManager
{
public:
	/// <summary>
	/// Create connection to lua file
	/// </summary>
	/// <param name="filename"></param>
	ResourcesManager(std::string filename);

	/// <summary>
	/// Close connection to lue file
	/// </summary>
	~ResourcesManager();

	/// <summary>
	/// Get string from file
	/// </summary>
	/// <param name="name">Variable name</param>
	/// <returns>Variable value</returns>
	std::string getString(std::string name);

	/// <summary>
	/// Get int from file
	/// </summary>
	/// <param name="name">Variable name</param>
	/// <returns>Variable value</returns>
	int getInt(std::string name);

	/// <summary>
	/// Get float from file
	/// </summary>
	/// <param name="name">Variable name</param>
	/// <returns>Variable value</returns>
	float getFloat(std::string name);
private:
	lua_State* m_L;				//!< Lua state for file connection
};
